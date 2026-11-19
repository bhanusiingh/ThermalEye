"""
ThermalEye — Local-First OpenStreetMap Industrial Enrichment & Event Scoring
=============================================================================

Enriches candidate fire events with nearby industrial, energy, and mining
infrastructure using a high-performance local PostGIS spatial index
(`osm_industrial_features`), with graceful fallback to Overpass API only
when the local table is unpopulated.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("thermaleye.enrich_osm")

_script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(_script_dir, ".env"))


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


DB_URL = get_database_url()

# ---------------------------------------------------------------------------
# Overpass Fallback Configuration
# ---------------------------------------------------------------------------
OVERPASS_URL = os.getenv("OVERPASS_API_URL", "https://overpass.private.coffee/api/interpreter")
FALLBACK_URLS_RAW = os.getenv(
    "OVERPASS_FALLBACK_URLS",
    "https://overpass-api.de/api/interpreter,https://overpass.kumi.systems/api/interpreter",
)
OVERPASS_FALLBACK_URLS = [u.strip() for u in FALLBACK_URLS_RAW.split(",") if u.strip()]

OVERPASS_USER_AGENT = os.getenv(
    "OVERPASS_USER_AGENT",
    "ThermalEye/1.0 (https://github.com/thermaleye/thermaleye; contact@thermaleye.local)",
)
OVERPASS_CONNECT_TIMEOUT = float(os.getenv("OVERPASS_CONNECT_TIMEOUT", "5.0"))
OVERPASS_READ_TIMEOUT = float(os.getenv("OVERPASS_READ_TIMEOUT", "15.0"))
OVERPASS_QUERY_TIMEOUT = int(os.getenv("OVERPASS_QUERY_TIMEOUT", "12"))
OVERPASS_RATE_LIMIT_DELAY = float(os.getenv("OVERPASS_RATE_LIMIT_DELAY", "1.0"))

OSM_BATCH_LIMIT = int(os.getenv("OSM_BATCH_LIMIT", "0"))  # 0 = all un-enriched
CACHE_DISTANCE_METERS = float(os.getenv("OSM_CACHE_DISTANCE_M", "1500.0"))
DEFAULT_SEARCH_RADIUS = int(os.getenv("OSM_SEARCH_RADIUS_M", "5000"))

# State for rate limiting & circuit breaker
_last_request_time: float = 0.0
CIRCUIT_BREAKER_COOLDOWN_SEC = float(os.getenv("OVERPASS_CIRCUIT_BREAKER_COOLDOWN_SEC", "60.0"))
CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("OVERPASS_CIRCUIT_BREAKER_THRESHOLD", "2"))
_circuit_breaker_open: bool = False
_circuit_breaker_tripped_at: float = 0.0
_consecutive_failures: int = 0


def _parse_retry_after(response: requests.Response, default_wait: float = 5.0) -> float:
    """Parse Retry-After header if present, else return default wait time."""
    retry_header = response.headers.get("Retry-After")
    if retry_header:
        try:
            return max(float(retry_header), 1.0)
        except ValueError:
            pass
    return default_wait


# ---------------------------------------------------------------------------
# Spatial Utilities & Caching for Fallback
# ---------------------------------------------------------------------------
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great circle distance in meters between two coordinates."""
    r_earth = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r_earth * c


class SpatialCache:
    """In-memory spatial cache for Overpass fallback."""

    def __init__(self, cache_radius_m: float = CACHE_DISTANCE_METERS):
        self.cache_radius_m = cache_radius_m
        self.facilities: List[Dict[str, Any]] = []
        self.empty_zones: List[Tuple[float, float, float]] = []

    def add_facilities(self, elements: List[Dict[str, Any]]) -> None:
        for elem in elements:
            tags = elem.get("tags", {})
            name = tags.get("name", "Unnamed Facility")
            facility_type = tags.get("power", tags.get("landuse", "Industrial Zone"))
            lat = elem.get("lat") or elem.get("center", {}).get("lat")
            lon = elem.get("lon") or elem.get("center", {}).get("lon")
            if lat is not None and lon is not None:
                if not any(
                    haversine_distance(lat, lon, f["lat"], f["lon"]) < 50
                    for f in self.facilities
                ):
                    self.facilities.append(
                        {
                            "lat": float(lat),
                            "lon": float(lon),
                            "name": name,
                            "type": facility_type,
                        }
                    )

    def add_empty_zone(self, lat: float, lon: float, radius_m: float) -> None:
        self.empty_zones.append((lat, lon, radius_m))

    def find_nearest_facility(
        self, lat: float, lon: float, max_radius_m: float = DEFAULT_SEARCH_RADIUS
    ) -> Optional[Tuple[str, str, float]]:
        best_match = None
        min_dist = float("inf")
        for f in self.facilities:
            dist = haversine_distance(lat, lon, f["lat"], f["lon"])
            if dist <= max_radius_m and dist < min_dist:
                min_dist = dist
                best_match = (f["name"], f["type"], dist)
        return best_match

    def is_in_empty_zone(self, lat: float, lon: float) -> bool:
        for z_lat, z_lon, z_radius in self.empty_zones:
            dist = haversine_distance(lat, lon, z_lat, z_lon)
            if dist <= 500 and z_radius >= DEFAULT_SEARCH_RADIUS:
                return True
        return False


spatial_cache = SpatialCache()


# ---------------------------------------------------------------------------
# Scoring Heuristic
# ---------------------------------------------------------------------------
def compute_priority_score(
    max_frp: Optional[float],
    detection_count: int,
    nearest_type: Optional[str],
    rate_limited: bool = False,
) -> Tuple[float, str]:
    """
    Compute Investigation Priority Score (0-100) based on FRP, persistence,
    and infrastructure proximity.

    If nearest_type is present, infrastructure score is 30.0.
    If rate_limited is True, infrastructure score is 0 and note is
    '0/30 (Overpass Rate Limited)'.
    """
    frp_score = min(max_frp / 100.0 * 40.0, 40.0) if max_frp else 0.0
    persistence_score = min(detection_count * 10.0, 30.0)
    infra_score = 30.0 if (nearest_type and not rate_limited) else 0.0

    total_score = min(frp_score + persistence_score + infra_score, 100.0)

    if rate_limited:
        infra_note = "0/30 (Overpass Rate Limited)"
    else:
        infra_note = f"{infra_score:.0f}/30"

    priority_reasons = json.dumps(
        {
            "Thermal Intensity (FRP)": f"{frp_score:.1f}/40",
            "Persistence": f"{persistence_score:.0f}/30",
            "Infrastructure Proximity": infra_note,
        }
    )
    return total_score, priority_reasons


# ---------------------------------------------------------------------------
# Overpass API Calls (Fallback)
# ---------------------------------------------------------------------------
def _wait_rate_limit(min_delay: float = OVERPASS_RATE_LIMIT_DELAY) -> None:
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < min_delay:
        sleep_time = min_delay - elapsed + random.uniform(0.05, 0.25)
        time.sleep(sleep_time)
    _last_request_time = time.time()


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=2.0),
    retry=retry_if_exception_type((requests.exceptions.Timeout, requests.exceptions.HTTPError)),
    reraise=True,
)
def _call_overpass_endpoint(endpoint: str, query_str: str) -> Dict[str, Any]:
    global _circuit_breaker_open, _circuit_breaker_tripped_at, _consecutive_failures
    _wait_rate_limit()

    headers = {
        "User-Agent": OVERPASS_USER_AGENT,
        "Accept": "application/json",
    }
    try:
        resp = requests.post(
            endpoint,
            data={"data": query_str},
            headers=headers,
            timeout=(OVERPASS_CONNECT_TIMEOUT, OVERPASS_READ_TIMEOUT),
            verify=True,
        )
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        _consecutive_failures += 1
        if _consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD:
            _circuit_breaker_open = True
            _circuit_breaker_tripped_at = time.time()
        raise

    if resp.status_code == 429:
        _consecutive_failures += 1
        _circuit_breaker_open = True
        _circuit_breaker_tripped_at = time.time()
        raise requests.exceptions.HTTPError(
            f"429 Client Error: Too Many Requests for url: {endpoint}",
            response=resp,
        )

    if resp.status_code >= 400:
        if resp.status_code in (502, 503, 504):
            _consecutive_failures += 1
            if _consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD:
                _circuit_breaker_open = True
                _circuit_breaker_tripped_at = time.time()
        raise requests.exceptions.HTTPError(
            f"{resp.status_code} Error for url: {endpoint}",
            response=resp,
        )

    _consecutive_failures = 0
    return resp.json()


def fetch_osm_infrastructure(
    lat: float, lon: float, radius: int = DEFAULT_SEARCH_RADIUS
) -> Dict[str, Any]:
    """Fetch industrial infrastructure from Overpass API (fallback)."""
    global _circuit_breaker_open, _circuit_breaker_tripped_at, _consecutive_failures

    if _circuit_breaker_open:
        if time.time() - _circuit_breaker_tripped_at < CIRCUIT_BREAKER_COOLDOWN_SEC:
            resp_mock = requests.Response()
            resp_mock.status_code = 429
            raise requests.exceptions.HTTPError(
                "429 Client Error: Too Many Requests (Overpass circuit breaker active)",
                response=resp_mock,
            )
        else:
            _circuit_breaker_open = False
            _consecutive_failures = 0

    query = f"""
    [out:json][timeout:{OVERPASS_QUERY_TIMEOUT}];
    (
      node["landuse"="industrial"](around:{radius},{lat},{lon});
      way["landuse"="industrial"](around:{radius},{lat},{lon});
      node["power"="plant"](around:{radius},{lat},{lon});
      way["power"="plant"](around:{radius},{lat},{lon});
    );
    out center;
    """
    endpoints = [OVERPASS_URL] + [
        ep for ep in OVERPASS_FALLBACK_URLS if ep != OVERPASS_URL
    ]
    last_exc = None
    for endpoint in endpoints:
        try:
            return _call_overpass_endpoint(endpoint, query)
        except (
            requests.exceptions.Timeout,
            requests.exceptions.HTTPError,
            RetryError,
            requests.exceptions.RequestException,
        ) as exc:
            last_exc = exc
            if _circuit_breaker_open:
                break
            continue

    if last_exc:
        raise last_exc
    resp_mock = requests.Response()
    resp_mock.status_code = 429
    raise requests.exceptions.HTTPError(
        "429 Client Error: No Overpass endpoints available", response=resp_mock
    )


# ---------------------------------------------------------------------------
# Local PostGIS OSM Spatial Index Lookup
# ---------------------------------------------------------------------------
def check_local_osm_count(engine) -> int:
    """Return count of features in local osm_industrial_features table."""
    with engine.connect() as conn:
        table_exists = conn.execute(
            text(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'osm_industrial_features')"
            )
        ).scalar()
        if not table_exists:
            return 0
        return int(
            conn.execute(text("SELECT COUNT(*) FROM osm_industrial_features")).scalar()
            or 0
        )


def query_nearest_local_osm(
    conn, lon: float, lat: float, radius_m: float = 5000.0
) -> Optional[Dict[str, Any]]:
    """
    Query the nearest industrial/power/mining feature within radius_m meters
    using PostGIS spatial index.
    """
    query = text(
        """
        SELECT
            name,
            osm_type,
            category,
            ST_Distance(
                geom::geography,
                ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography
            ) AS dist_m
        FROM osm_industrial_features
        WHERE ST_DWithin(
            geom::geography,
            ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography,
            :radius_m
        )
        ORDER BY ST_Distance(
            geom::geography,
            ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography
        ) ASC
        LIMIT 1
        """
    )
    row = conn.execute(query, {"lon": lon, "lat": lat, "radius_m": radius_m}).fetchone()
    if not row:
        return None

    return {
        "name": row[0],
        "osm_type": row[1],
        "category": row[2],
        "dist_m": float(row[3]),
    }


# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------
def process_enrichment(
    batch_limit: int = OSM_BATCH_LIMIT,
    re_enrich_rate_limited: bool = False,
    db_url: Optional[str] = None,
) -> Dict[str, int]:
    """
    Enrich candidate fire events with OSM infrastructure.

    Local-first strategy:
      1. Inspect `osm_industrial_features`. If > 0 rows, use Local PostGIS Mode:
         - Query nearest feature within 5000m using ST_DWithin.
         - Populate nearest_osm_name, nearest_osm_type, nearest_osm_distance_km.
         - Compute priority_score with industrial proximity boost.
         - Mark enrichment status as 'enriched' (if match found) or 'none_nearby'.
      2. If local index has 0 rows, fall back to Overpass API.
    """
    logger.info("=" * 60)
    logger.info("ThermalEye — OSM Enrichment and Scoring Worker Starting")
    logger.info("Batch Limit: %s (0=unlimited)", batch_limit)
    logger.info("Re-enrich rate limited: %s", re_enrich_rate_limited)
    logger.info("=" * 60)

    resolved_db = db_url or get_database_url()
    engine = create_engine(resolved_db)

    local_count = check_local_osm_count(engine)
    logger.info("Local osm_industrial_features count: %d", local_count)

    stats = {
        "processed": 0,
        "successful": 0,
        "enriched": 0,
        "none_nearby": 0,
        "cached": 0,
        "rate_limited": 0,
        "failed": 0,
    }

    # Query events
    if re_enrich_rate_limited:
        query_sql = """
            SELECT id, ST_X(centroid) as lon, ST_Y(centroid) as lat, max_frp, detection_count
            FROM candidate_events
            WHERE priority_reasons IS NULL
               OR priority_reasons::text ILIKE '%Rate Limited%'
               OR nearest_osm_name IS NULL
            ORDER BY max_frp DESC NULLS LAST
        """
    else:
        query_sql = """
            SELECT id, ST_X(centroid) as lon, ST_Y(centroid) as lat, max_frp, detection_count
            FROM candidate_events
            WHERE priority_reasons IS NULL
            ORDER BY max_frp DESC NULLS LAST
        """

    if batch_limit > 0:
        query_sql += f" LIMIT {batch_limit}"

    with engine.connect() as conn:
        events = conn.execute(text(query_sql)).fetchall()

    total_to_process = len(events)
    logger.info("Found %d candidate events needing OSM enrichment.", total_to_process)
    if total_to_process == 0:
        return stats

    batch_buffer: List[Dict[str, Any]] = []

    def flush_batch() -> None:
        if not batch_buffer:
            return
        with engine.begin() as conn:
            for item in batch_buffer:
                conn.execute(
                    text(
                        """
                        UPDATE candidate_events
                        SET nearest_osm_name = :name,
                            nearest_osm_type = :type,
                            nearest_osm_distance_km = :dist_km,
                            priority_score = :score,
                            priority_reasons = CAST(:reasons AS jsonb),
                            updated_at = NOW()
                        WHERE id = CAST(:id AS uuid)
                        """
                    ),
                    item,
                )
        batch_buffer.clear()

    if local_count > 0:
        # -------------------------------------------------------------------
        # Mode A: Local PostGIS Mode (High Performance, No External Rate Limits)
        # -------------------------------------------------------------------
        logger.info("Using Local PostGIS Mode (Index size: %d features)...", local_count)
        with engine.connect() as read_conn:
            for idx, event in enumerate(events, 1):
                event_id = event.id
                lat, lon = float(event.lat), float(event.lon)

                match = query_nearest_local_osm(read_conn, lon, lat, radius_m=5000.0)

                if match:
                    nearest_name = match["name"]
                    nearest_type = match["osm_type"]
                    dist_km = round(match["dist_m"] / 1000.0, 3)
                    stats["enriched"] += 1
                else:
                    nearest_name = None
                    nearest_type = None
                    dist_km = None
                    stats["none_nearby"] += 1

                score, reasons = compute_priority_score(
                    event.max_frp,
                    event.detection_count,
                    nearest_type,
                    rate_limited=False,
                )

                batch_buffer.append(
                    {
                        "name": nearest_name,
                        "type": nearest_type,
                        "dist_km": dist_km,
                        "score": score,
                        "reasons": reasons,
                        "id": str(event_id),
                    }
                )
                stats["successful"] += 1
                stats["processed"] += 1

                if len(batch_buffer) >= 200:
                    flush_batch()

                if stats["processed"] % 1000 == 0 or stats["processed"] == total_to_process:
                    logger.info(
                        "Local Enrichment Progress: %d/%d processed (Enriched: %d, NoneNearby: %d)",
                        stats["processed"],
                        total_to_process,
                        stats["enriched"],
                        stats["none_nearby"],
                    )

        flush_batch()

    else:
        # -------------------------------------------------------------------
        # Mode B: Fallback to Overpass API
        # -------------------------------------------------------------------
        logger.warning(
            "Local osm_industrial_features is empty. Falling back to Overpass API."
        )
        for idx, event in enumerate(events, 1):
            event_id = event.id
            lat, lon = float(event.lat), float(event.lon)

            # Check spatial cache
            cached_facility = spatial_cache.find_nearest_facility(
                lat, lon, DEFAULT_SEARCH_RADIUS
            )
            if cached_facility:
                nearest_name, nearest_type, dist_m = cached_facility
                score, reasons = compute_priority_score(
                    event.max_frp,
                    event.detection_count,
                    nearest_type,
                    rate_limited=False,
                )
                batch_buffer.append(
                    {
                        "name": nearest_name,
                        "type": nearest_type,
                        "dist_km": round(dist_m / 1000.0, 3),
                        "score": score,
                        "reasons": reasons,
                        "id": str(event_id),
                    }
                )
                stats["cached"] += 1
                stats["successful"] += 1
                stats["processed"] += 1
                if len(batch_buffer) >= 100:
                    flush_batch()
                continue

            if spatial_cache.is_in_empty_zone(lat, lon):
                score, reasons = compute_priority_score(
                    event.max_frp, event.detection_count, None, rate_limited=False
                )
                batch_buffer.append(
                    {
                        "name": None,
                        "type": None,
                        "dist_km": None,
                        "score": score,
                        "reasons": reasons,
                        "id": str(event_id),
                    }
                )
                stats["cached"] += 1
                stats["successful"] += 1
                stats["processed"] += 1
                if len(batch_buffer) >= 100:
                    flush_batch()
                continue

            nearest_name = None
            nearest_type = None
            dist_km = None
            rate_limited = False

            try:
                osm_data = fetch_osm_infrastructure(
                    lat, lon, radius=DEFAULT_SEARCH_RADIUS
                )
                elements = (
                    osm_data.get("elements", []) if isinstance(osm_data, dict) else []
                )
                if elements:
                    spatial_cache.add_facilities(elements)
                    match_cache = spatial_cache.find_nearest_facility(
                        lat, lon, DEFAULT_SEARCH_RADIUS
                    )
                    if match_cache:
                        nearest_name, nearest_type, dist_m = match_cache
                        dist_km = round(dist_m / 1000.0, 3)
                        stats["enriched"] += 1
                    else:
                        stats["none_nearby"] += 1
                else:
                    spatial_cache.add_empty_zone(lat, lon, DEFAULT_SEARCH_RADIUS)
                    stats["none_nearby"] += 1
                stats["successful"] += 1

            except (
                requests.exceptions.Timeout,
                requests.exceptions.HTTPError,
                RetryError,
                requests.exceptions.RequestException,
            ) as exc:
                if stats["rate_limited"] == 0:
                    logger.warning(
                        "Overpass API unavailable/rate-limited for event %s (lat=%.3f, lon=%.3f): %s.",
                        event_id,
                        lat,
                        lon,
                        exc,
                    )
                nearest_name = None
                nearest_type = None
                dist_km = None
                rate_limited = True
                stats["rate_limited"] += 1

            score, reasons = compute_priority_score(
                event.max_frp,
                event.detection_count,
                nearest_type,
                rate_limited=rate_limited,
            )

            batch_buffer.append(
                {
                    "name": nearest_name,
                    "type": nearest_type,
                    "dist_km": dist_km,
                    "score": score,
                    "reasons": reasons,
                    "id": str(event_id),
                }
            )
            stats["processed"] += 1

            if len(batch_buffer) >= 100:
                flush_batch()

            if stats["processed"] % 500 == 0 or stats["processed"] == total_to_process:
                logger.info(
                    "Fallback Overpass Progress: %d/%d processed (Success: %d, Cached: %d, RateLimits: %d)",
                    stats["processed"],
                    total_to_process,
                    stats["successful"],
                    stats["cached"],
                    stats["rate_limited"],
                )

        flush_batch()

    logger.info("=" * 60)
    logger.info("OSM enrichment and scoring complete.")
    logger.info("Summary: %s", stats)
    logger.info("=" * 60)
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Enrich candidate events with nearby industrial/power infrastructure."
    )
    parser.add_argument(
        "--batch-limit",
        type=int,
        default=OSM_BATCH_LIMIT,
        help="Maximum candidate events to process (0 = all)",
    )
    parser.add_argument(
        "--re-enrich-rate-limited",
        action="store_true",
        help="Re-process all candidate events marked rate-limited or lacking OSM results",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=None,
        help="Optional database connection URL",
    )
    args = parser.parse_args()

    process_enrichment(
        batch_limit=args.batch_limit,
        re_enrich_rate_limited=args.re_enrich_rate_limited,
        db_url=args.db_url,
    )


if __name__ == "__main__":
    main()
