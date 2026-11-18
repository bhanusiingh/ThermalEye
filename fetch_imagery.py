"""
ThermalEye — Phase 4: Copernicus Sentinel-2 Imagery Retrieval
==============================================================

Fetches True Color (RGB) and SWIR (B12, B11, B8A) image chips from
the Copernicus Data Space Ecosystem (CDSE) Sentinel Hub APIs for
high-priority candidate events.

Architectural invariants:
  • All HTTP requests use verify=True — strict TLS, no exceptions.
  • Exponential-backoff retries on 401 (token expiry), 502, 504.
  • Valid STAC payload (no unsupported 'query' extensions).
  • Missing cloud-free imagery → paths stored as SQL NULL, not mocked.
  • Credentials loaded exclusively from .env via python-dotenv.
  • Resumable: queries WHERE imagery_fetched_at IS NULL, CAST(:event_id AS uuid).
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("thermaleye.fetch_imagery")

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(_script_dir, ".env"))

CDSE_CLIENT_ID: str = os.environ.get("CDSE_CLIENT_ID", "")
CDSE_CLIENT_SECRET: str = os.environ.get("CDSE_CLIENT_SECRET", "")
CDSE_TOKEN_URL: str = os.environ.get(
    "CDSE_TOKEN_URL",
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
)
CDSE_CATALOG_URL: str = os.environ.get(
    "CDSE_CATALOG_URL",
    "https://sh.dataspace.copernicus.eu/catalog/v1/search",
)
CDSE_PROCESS_URL: str = os.environ.get(
    "CDSE_PROCESS_URL",
    "https://sh.dataspace.copernicus.eu/api/v1/process",
)


def get_database_url() -> str:
    url = os.environ.get(
        "DATABASE_URL",
        "postgresql://thermaleye:thermaleye_secret@localhost:5432/thermaleye",
    )
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    if "+" not in url.split("://")[0]:
        try:
            import psycopg2
            url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
        except ImportError:
            try:
                import psycopg
                url = url.replace("postgresql://", "postgresql+psycopg://", 1)
            except ImportError:
                pass
    return url


DATABASE_URL: str = get_database_url()

# Imagery output directory
IMAGERY_DIR = Path(os.environ.get("IMAGERY_DIR", os.path.join(_script_dir, "static", "imagery")))

# Processing parameters
MAX_CLOUD_COVER: float = float(os.environ.get("MAX_CLOUD_COVER", "20"))
LOOKFORWARD_DAYS: int = int(os.environ.get("LOOKFORWARD_DAYS", "5"))
IMAGE_SIZE_PX: int = int(os.environ.get("IMAGE_SIZE_PX", "512"))
# Bounding-box half-width in degrees (~0.05° ≈ 5.5 km)
BBOX_HALF_DEG: float = float(os.environ.get("BBOX_HALF_DEG", "0.05"))
# Minimum priority_score to fetch imagery (saves API quota)
MIN_PRIORITY_SCORE: float = float(os.environ.get("MIN_PRIORITY_IMAGERY", "0"))
IMAGERY_BATCH_LIMIT: int = int(os.environ.get("IMAGERY_BATCH_LIMIT", "0"))

# HTTP timeouts (seconds)
HTTP_CONNECT_TIMEOUT: float = float(os.environ.get("HTTP_CONNECT_TIMEOUT", "15"))
HTTP_READ_TIMEOUT: float = float(os.environ.get("HTTP_READ_TIMEOUT", "90"))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if not CDSE_CLIENT_ID or not CDSE_CLIENT_SECRET:
    logger.critical(
        "CDSE_CLIENT_ID and CDSE_CLIENT_SECRET must be set in .env. Aborting."
    )
    sys.exit(1)

IMAGERY_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Retry predicate
# ---------------------------------------------------------------------------
def _is_retryable(exc: BaseException) -> bool:
    """Retry on transient gateway errors and expired tokens."""
    global _cached_token
    if isinstance(exc, requests.exceptions.HTTPError):
        code = getattr(exc.response, "status_code", 0)
        if code == 401:
            _cached_token = None  # force token refresh on retry
            return True
        return code in (429, 502, 503, 504)
    if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return True
    return False


# ---------------------------------------------------------------------------
# CDSE OAuth2 Token Management
# ---------------------------------------------------------------------------
_cached_token: Optional[str] = None
_token_expiry: datetime = datetime.min.replace(tzinfo=timezone.utc)


@retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _fetch_new_token() -> tuple[str, datetime]:
    """Obtain a fresh OAuth2 access token from CDSE."""
    logger.info("Requesting new CDSE OAuth2 token…")
    resp = requests.post(
        CDSE_TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": CDSE_CLIENT_ID,
            "client_secret": CDSE_CLIENT_SECRET,
        },
        timeout=(HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT),
        verify=True,
    )
    if resp.status_code >= 400:
        logger.error("CDSE token error HTTP %d: %s", resp.status_code, resp.text[:400])
    resp.raise_for_status()
    body = resp.json()
    access_token = body["access_token"]
    expires_in = int(body.get("expires_in", 300))
    expiry = datetime.now(timezone.utc) + timedelta(seconds=max(expires_in - 60, 30))
    logger.info("CDSE token obtained, expires at %s", expiry.isoformat())
    return access_token, expiry


def get_cdse_token() -> str:
    """Return a valid CDSE access token, refreshing if expired."""
    global _cached_token, _token_expiry
    if _cached_token and datetime.now(timezone.utc) < _token_expiry:
        return _cached_token
    _cached_token, _token_expiry = _fetch_new_token()
    return _cached_token


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {get_cdse_token()}"}


# ---------------------------------------------------------------------------
# STAC Catalog Search
# ---------------------------------------------------------------------------
@retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=3, max=60),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def search_sentinel2_scenes(
    lon: float,
    lat: float,
    date_start: datetime,
    date_end: datetime,
) -> list[dict]:
    """
    Query the CDSE STAC catalog for Sentinel-2 L2A scenes that
    intersect the event centroid within the date range and have
    cloud cover < MAX_CLOUD_COVER%.

    Sentinel Hub STAC does not support the 'query' extension in the payload;
    filtering by cloud cover is performed on the returned features.
    """
    bbox = [
        lon - BBOX_HALF_DEG,
        lat - BBOX_HALF_DEG,
        lon + BBOX_HALF_DEG,
        lat + BBOX_HALF_DEG,
    ]
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": f"{date_start.strftime('%Y-%m-%dT%H:%M:%SZ')}"
                    f"/{date_end.strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "limit": 10,
        "fields": {
            "include": [
                "id",
                "properties.datetime",
                "properties.eo:cloud_cover",
                "bbox",
            ],
        },
    }

    logger.info(
        "STAC search: bbox=%s, dates=%s to %s, max_cloud=%s%%",
        bbox,
        date_start.date(),
        date_end.date(),
        MAX_CLOUD_COVER,
    )

    resp = requests.post(
        CDSE_CATALOG_URL,
        json=payload,
        headers={
            **_auth_headers(),
            "Content-Type": "application/json",
        },
        timeout=(HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT),
        verify=True,
    )

    if resp.status_code >= 400:
        logger.error(
            "CDSE STAC search failed HTTP %d: %s",
            resp.status_code,
            resp.text[:500],
        )

    resp.raise_for_status()

    body = resp.json()
    raw_features = body.get("features", [])

    # Filter client-side by cloud cover
    features = [
        f
        for f in raw_features
        if f.get("properties", {}).get("eo:cloud_cover", 100) <= MAX_CLOUD_COVER
    ]

    # Sort by cloud cover ascending — best (clearest) scene first
    features.sort(
        key=lambda f: f.get("properties", {}).get("eo:cloud_cover", 100)
    )

    logger.info(
        "STAC returned %d total scenes, %d with cloud cover <= %s%%.",
        len(raw_features),
        len(features),
        MAX_CLOUD_COVER,
    )
    return features


# ---------------------------------------------------------------------------
# Evalscript definitions
# ---------------------------------------------------------------------------
EVALSCRIPT_TRUE_COLOR = """
//VERSION=3
function setup() {
    return {
        input: [{
            bands: ["B04", "B03", "B02"],
            units: "DN"
        }],
        output: {
            bands: 3,
            sampleType: "AUTO"
        }
    };
}

function evaluatePixel(sample) {
    return [
        sample.B04 / 3000,
        sample.B03 / 3000,
        sample.B02 / 3000
    ];
}
"""

EVALSCRIPT_SWIR = """
//VERSION=3
function setup() {
    return {
        input: [{
            bands: ["B12", "B11", "B8A"],
            units: "DN"
        }],
        output: {
            bands: 3,
            sampleType: "AUTO"
        }
    };
}

function evaluatePixel(sample) {
    return [
        sample.B12 / 4000,
        sample.B11 / 4000,
        sample.B8A / 4000
    ];
}
"""


# ---------------------------------------------------------------------------
# Process API — download image chip
# ---------------------------------------------------------------------------
@retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def download_image_chip(
    lon: float,
    lat: float,
    scene_datetime: str,
    evalscript: str,
    output_path: Path,
) -> bool:
    """
    Use the CDSE Sentinel Hub Process API to render and download
    a 512x512 PNG image chip.
    """
    bbox = [
        lon - BBOX_HALF_DEG,
        lat - BBOX_HALF_DEG,
        lon + BBOX_HALF_DEG,
        lat + BBOX_HALF_DEG,
    ]

    scene_dt = datetime.fromisoformat(scene_datetime.replace("Z", "+00:00"))
    from_time = (scene_dt - timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%SZ")
    to_time = (scene_dt + timedelta(hours=12)).strftime("%Y-%m-%dT%H:%M:%SZ")

    payload = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": from_time,
                            "to": to_time,
                        },
                        "maxCloudCoverage": MAX_CLOUD_COVER,
                    },
                }
            ],
        },
        "output": {
            "width": IMAGE_SIZE_PX,
            "height": IMAGE_SIZE_PX,
            "responses": [
                {
                    "identifier": "default",
                    "format": {"type": "image/png"},
                }
            ],
        },
        "evalscript": evalscript,
    }

    logger.info("Downloading image chip → %s", output_path.name)

    resp = requests.post(
        CDSE_PROCESS_URL,
        json=payload,
        headers={
            **_auth_headers(),
            "Content-Type": "application/json",
            "Accept": "image/png",
        },
        timeout=(HTTP_CONNECT_TIMEOUT, 120),
        verify=True,
    )
    if resp.status_code >= 400:
        logger.error(
            "CDSE Process API failed HTTP %d: %s",
            resp.status_code,
            resp.text[:500],
        )
    resp.raise_for_status()

    if resp.headers.get("Content-Type", "").startswith("image/"):
        output_path.write_bytes(resp.content)
        logger.info("Saved %d bytes → %s", len(resp.content), output_path)
        return True
    else:
        logger.warning(
            "Unexpected content-type %s for %s",
            resp.headers.get("Content-Type"),
            output_path.name,
        )
        return False


# ---------------------------------------------------------------------------
# Fetch imagery for a single event
# ---------------------------------------------------------------------------
def fetch_event_imagery(
    event_id: str,
    lon: float,
    lat: float,
    first_seen: datetime,
    last_seen: datetime,
) -> dict:
    """
    Search for Sentinel-2 scenes and download RGB + SWIR chips.
    Returns a dict with keys: rgb_path, swir_path, scene_id (None if not found).
    """
    search_start = first_seen
    search_end = last_seen + timedelta(days=LOOKFORWARD_DAYS)

    now = datetime.now(timezone.utc)
    if search_end > now:
        search_end = now

    result = {"rgb_path": None, "swir_path": None, "scene_id": None}

    try:
        scenes = search_sentinel2_scenes(lon, lat, search_start, search_end)
    except Exception:
        logger.exception("STAC search failed for event %s.", event_id)
        return result

    if not scenes:
        logger.warning(
            "No cloud-free Sentinel-2 scenes found for event %s "
            "(cloud_cover <= %s%%, %s to %s). Storing NULL.",
            event_id[:8],
            MAX_CLOUD_COVER,
            search_start.date(),
            search_end.date(),
        )
        return result

    best_scene = scenes[0]
    scene_id = best_scene.get("id", "unknown")
    scene_dt = best_scene.get("properties", {}).get("datetime", search_end.isoformat())
    cloud = best_scene.get("properties", {}).get("eo:cloud_cover", "?")
    logger.info(
        "Best scene for event %s: %s (cloud: %s%%)",
        event_id[:8],
        scene_id,
        cloud,
    )

    result["scene_id"] = scene_id

    # Download RGB chip
    rgb_path = IMAGERY_DIR / f"{event_id}_rgb.png"
    try:
        if download_image_chip(lon, lat, scene_dt, EVALSCRIPT_TRUE_COLOR, rgb_path):
            result["rgb_path"] = f"static/imagery/{event_id}_rgb.png"
    except Exception:
        logger.exception("Failed to download RGB chip for event %s.", event_id)

    # Download SWIR chip
    swir_path = IMAGERY_DIR / f"{event_id}_swir.png"
    try:
        if download_image_chip(lon, lat, scene_dt, EVALSCRIPT_SWIR, swir_path):
            result["swir_path"] = f"static/imagery/{event_id}_swir.png"
    except Exception:
        logger.exception("Failed to download SWIR chip for event %s.", event_id)

    return result


# ---------------------------------------------------------------------------
# Database update
# ---------------------------------------------------------------------------
def update_event_imagery(engine, event_id: str, imagery: dict) -> None:
    """
    Persist imagery paths and scene ID to candidate_events.
    Uses CAST(:event_id AS uuid) to avoid PostgreSQL/SQLAlchemy syntax collision.
    """
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE candidate_events
                SET imagery_rgb_path    = :rgb_path,
                    imagery_swir_path   = :swir_path,
                    sentinel2_scene_id  = :scene_id,
                    imagery_fetched_at  = NOW(),
                    updated_at          = NOW()
                WHERE id = CAST(:event_id AS uuid)
                """
            ),
            {
                "event_id": str(event_id),
                "rgb_path": imagery.get("rgb_path"),
                "swir_path": imagery.get("swir_path"),
                "scene_id": imagery.get("scene_id"),
            },
        )
    logger.info(
        "Updated DB for event %s: rgb=%s, swir=%s, scene=%s",
        str(event_id)[:8],
        imagery.get("rgb_path") or "NULL",
        imagery.get("swir_path") or "NULL",
        imagery.get("scene_id") or "NULL",
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_imagery_fetch(batch_limit: int = IMAGERY_BATCH_LIMIT) -> dict:
    """
    Main entry point. Queries candidate_events that lack imagery
    and fetches Sentinel-2 chips for each.
    """
    logger.info("=" * 60)
    logger.info("ThermalEye — Sentinel-2 imagery fetch starting")
    logger.info("=" * 60)

    engine = create_engine(DATABASE_URL, echo=False)

    query_sql = """
        SELECT
            id::text,
            ST_X(centroid) AS lon,
            ST_Y(centroid) AS lat,
            first_seen,
            last_seen,
            priority_score
        FROM candidate_events
        WHERE imagery_fetched_at IS NULL
          AND (priority_score >= :min_score OR priority_score IS NULL)
        ORDER BY priority_score DESC NULLS LAST
    """
    if batch_limit > 0:
        query_sql += f" LIMIT {batch_limit}"

    with engine.connect() as conn:
        rows = conn.execute(
            text(query_sql),
            {"min_score": MIN_PRIORITY_SCORE},
        ).fetchall()

    logger.info("Found %d events needing imagery.", len(rows))

    stats = {"processed": 0, "imagery_found": 0, "no_imagery": 0, "errors": 0}

    for row in rows:
        event_id = row[0]
        lon, lat = row[1], row[2]
        first_seen, last_seen = row[3], row[4]

        try:
            imagery = fetch_event_imagery(event_id, lon, lat, first_seen, last_seen)
            update_event_imagery(engine, event_id, imagery)

            if imagery.get("rgb_path") or imagery.get("swir_path"):
                stats["imagery_found"] += 1
            else:
                stats["no_imagery"] += 1
        except Exception:
            logger.exception("Error processing event %s.", event_id)
            # Record attempt so it is not re-queried indefinitely
            update_event_imagery(
                engine,
                event_id,
                {"rgb_path": None, "swir_path": None, "scene_id": None},
            )
            stats["errors"] += 1

        stats["processed"] += 1

    logger.info("Imagery fetch complete. Summary: %s", stats)
    return stats


if __name__ == "__main__":
    run_imagery_fetch()
