"""Sentinel-2 imagery retrieval using Copernicus Data Space Ecosystem (CDSE) APIs.

Verified API endpoints (October 2026):
  - OAuth2 token:  https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token
  - Catalogue:     https://sh.dataspace.copernicus.eu/catalog/v1/search
  - Processing:    https://sh.dataspace.copernicus.eu/process/v1
  - Collection ID: sentinel-2-l2a

Authentication uses OAuth2 client_credentials grant. Tokens are reused until expiry
and refreshed automatically on 401. Credentials are never logged or cached.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import requests

from thermaleye.config import (
    CDSE_TOKEN_URL,
    CDSE_CATALOG_URL,
    CDSE_PROCESS_URL,
    CDSE_S2_COLLECTION,
    DEFAULT_S2_DATE_RANGE_DAYS,
    DEFAULT_S2_MAX_CLOUD_PCT,
    DEFAULT_S2_PATCH_WIDTH_PX,
    DEFAULT_S2_PATCH_HEIGHT_PX,
    DEFAULT_S2_PATCH_HALF_DEG,
    DEFAULT_S2_OUTPUT_DIR,
    DEFAULT_S2_META_PATH,
    DEFAULT_S2_REQUEST_TIMEOUT,
    get_cdse_credentials,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# True-colour evalscript (Sentinel Hub Process API v3 format)
# Applies a brightness stretch of ×2.5 to bands B04/B03/B02 (R/G/B in L2A)
# ---------------------------------------------------------------------------
S2_TRUE_COLOUR_EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B02", "B03", "B04"], units: "REFLECTANCE" }],
    output: { bands: 3, sampleType: "UINT8" }
  };
}
function evaluatePixel(sample) {
  return [
    Math.round(Math.min(2.5 * sample.B04 * 255, 255)),
    Math.round(Math.min(2.5 * sample.B03 * 255, 255)),
    Math.round(Math.min(2.5 * sample.B02 * 255, 255))
  ];
}
"""


class SentinelHubConfigError(Exception):
    """Raised when CDSE credentials are missing or invalid."""


class SentinelHubAPIError(Exception):
    """Raised when a CDSE API call fails with a non-recoverable error."""


@dataclass
class TokenStore:
    """Holds a single OAuth2 access token with its expiry timestamp."""
    access_token: str = ""
    expires_at: float = 0.0      # Unix epoch seconds

    def is_valid(self, buffer_seconds: float = 60.0) -> bool:
        """Return True if the token is present and not expiring within `buffer_seconds`."""
        return bool(self.access_token) and (time.time() + buffer_seconds) < self.expires_at

    def clear(self) -> None:
        self.access_token = ""
        self.expires_at = 0.0


@dataclass
class S2SceneRecord:
    """Minimal metadata for a selected Sentinel-2 L2A scene."""
    scene_id: str
    acquisition_time: str         # ISO-8601 UTC
    cloud_cover_pct: Optional[float]
    bbox: List[float]             # [min_lon, min_lat, max_lon, max_lat]
    stac_datetime: str


@dataclass
class S2PatchResult:
    """Result of one Sentinel-2 patch retrieval."""
    event_id: str
    scene_id: str
    acquisition_time: str
    cloud_cover_pct: Optional[float]
    patch_width_px: int
    patch_height_px: int
    output_path: Optional[Path]
    success: bool
    image_fingerprint: str = ""
    source_lat: float = 0.0
    source_lon: float = 0.0
    error_message: Optional[str] = None

    def to_metadata_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "scene_id": self.scene_id,
            "acquisition_time": self.acquisition_time,
            "cloud_cover_pct": self.cloud_cover_pct,
            "patch_width_px": self.patch_width_px,
            "patch_height_px": self.patch_height_px,
            "output_path": str(self.output_path) if self.output_path else None,
            "success": self.success,
            "error_message": self.error_message,
        }


def compute_patch_bbox(
    lat: float,
    lon: float,
    half_deg: float = DEFAULT_S2_PATCH_HALF_DEG,
) -> List[float]:
    """Compute a square bounding box [min_lon, min_lat, max_lon, max_lat] around a point."""
    return [
        round(lon - half_deg, 6),
        round(lat - half_deg, 6),
        round(lon + half_deg, 6),
        round(lat + half_deg, 6),
    ]


def build_catalogue_search_payload(
    bbox: List[float],
    date_from: str,
    date_to: str,
    max_cloud_pct: float = DEFAULT_S2_MAX_CLOUD_PCT,
    limit: int = 10,
) -> Dict[str, Any]:
    """Build STAC-compatible POST body for CDSE Catalogue search."""
    payload: Dict[str, Any] = {
        "collections": [CDSE_S2_COLLECTION],
        "bbox": bbox,
        "datetime": f"{date_from}/{date_to}",
        "limit": limit,
        "filter-lang": "cql2-json",
        "filter": {
            "op": "<=",
            "args": [
                {"property": "eo:cloud_cover"},
                max_cloud_pct,
            ],
        },
    }
    return payload


def select_best_scene(features: List[Dict[str, Any]]) -> Optional[S2SceneRecord]:
    """Select the best Sentinel-2 scene from a STAC feature collection.

    Scene-selection rules (applied in order):
    1. Scenes with lower cloud cover are preferred over higher cloud cover.
    2. Among equal cloud cover, the most recent acquisition date is preferred.

    This approach maximises usable optical information while biasing
    toward the most temporally relevant observation relative to the event.
    Missing cloud metadata (None) is treated as maximum cloud cover (100%).
    """
    if not features:
        return None

    def sort_key(feat: Dict[str, Any]) -> Tuple[float, str]:
        props = feat.get("properties", {})
        cloud = props.get("eo:cloud_cover")
        cloud_val = float(cloud) if cloud is not None else 100.0
        dt = props.get("datetime") or props.get("created") or ""
        return (cloud_val, dt)

    # Sort ascending cloud cover, descending date (negate via string is unreliable;
    # sort ascending cloud then take latest among ties with max on datetime)
    sorted_features = sorted(features, key=lambda f: sort_key(f)[0])
    min_cloud = sort_key(sorted_features[0])[0]
    candidates = [f for f in sorted_features if sort_key(f)[0] == min_cloud]

    # Among tied cloud cover: pick most recent datetime
    best = max(candidates, key=lambda f: f.get("properties", {}).get("datetime", ""))
    props = best.get("properties", {})
    scene_id = best.get("id", "unknown")
    acq_time = props.get("datetime", "")
    cloud_val = props.get("eo:cloud_cover")
    geom_bbox = best.get("bbox", [])
    stac_dt = props.get("datetime", "")

    return S2SceneRecord(
        scene_id=scene_id,
        acquisition_time=acq_time,
        cloud_cover_pct=float(cloud_val) if cloud_val is not None else None,
        bbox=geom_bbox,
        stac_datetime=stac_dt,
    )


def build_process_request_body(
    bbox: List[float],
    date_from: str,
    date_to: str,
    width_px: int = DEFAULT_S2_PATCH_WIDTH_PX,
    height_px: int = DEFAULT_S2_PATCH_HEIGHT_PX,
    max_cloud_pct: float = DEFAULT_S2_MAX_CLOUD_PCT,
) -> Dict[str, Any]:
    """Build the JSON body for a Sentinel Hub Processing API request."""
    return {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
                },
            },
            "data": [
                {
                    "type": CDSE_S2_COLLECTION,
                    "dataFilter": {
                        "timeRange": {
                            "from": date_from,
                            "to": date_to,
                        },
                        "maxCloudCoverage": max_cloud_pct,
                        "mosaickingOrder": "leastCC",
                    },
                }
            ],
        },
        "output": {
            "width": width_px,
            "height": height_px,
            "responses": [
                {
                    "identifier": "default",
                    "format": {"type": "image/png"},
                }
            ],
        },
        "evalscript": S2_TRUE_COLOUR_EVALSCRIPT,
    }


class CDSEClient:
    """OAuth2-authenticated client for CDSE Catalogue and Processing APIs."""

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        token_url: str = CDSE_TOKEN_URL,
        catalog_url: str = CDSE_CATALOG_URL,
        process_url: str = CDSE_PROCESS_URL,
        timeout: int = DEFAULT_S2_REQUEST_TIMEOUT,
        result_cache_dir: Optional[Path] = None,
    ) -> None:
        cid, csec = get_cdse_credentials()
        self._client_id = client_id or cid
        self._client_secret = client_secret or csec

        if not self._client_id or not self._client_secret:
            raise SentinelHubConfigError(
                "CDSE_CLIENT_ID and CDSE_CLIENT_SECRET must be set in .env. "
                "See .env.example for the required keys."
            )

        self.token_url = token_url
        self.catalog_url = catalog_url
        self.process_url = process_url
        self.timeout = timeout
        self._token = TokenStore()
        self._result_cache_dir = result_cache_dir or DEFAULT_S2_OUTPUT_DIR

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------

    def _fetch_token(self) -> None:
        """Exchange client credentials for an OAuth2 access token."""
        resp = requests.post(
            self.token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise SentinelHubAPIError(
                f"Token request failed with HTTP {resp.status_code}. "
                "Check CDSE_CLIENT_ID and CDSE_CLIENT_SECRET."
            )
        payload = resp.json()
        self._token.access_token = payload["access_token"]
        # `expires_in` is given in seconds; record absolute expiry time
        self._token.expires_at = time.time() + int(payload.get("expires_in", 600))

    def _get_valid_token(self) -> str:
        """Return a valid token, fetching a new one if needed."""
        if not self._token.is_valid():
            self._fetch_token()
        return self._token.access_token

    def _auth_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._get_valid_token()}",
            "Content-Type": "application/json",
        }

    def _handle_401_retry(self, call):
        """Execute `call(headers)` once; on 401 refresh token and retry once."""
        headers = self._auth_headers()
        resp = call(headers)
        if resp.status_code == 401:
            logger.info("Token rejected (401), refreshing and retrying once.")
            self._token.clear()
            headers = self._auth_headers()
            resp = call(headers)
        return resp

    # ------------------------------------------------------------------
    # Catalogue search
    # ------------------------------------------------------------------

    def search_scenes(
        self,
        bbox: List[float],
        date_from: str,
        date_to: str,
        max_cloud_pct: float = DEFAULT_S2_MAX_CLOUD_PCT,
        limit: int = 10,
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """Query CDSE Catalogue for Sentinel-2 L2A scenes.

        Returns:
            (success, features_list). Features list may be empty when no
            scenes match the criteria (confirmed absence, not an error).
        """
        payload = build_catalogue_search_payload(bbox, date_from, date_to, max_cloud_pct, limit)

        def call(headers):
            return requests.post(
                self.catalog_url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )

        try:
            resp = self._handle_401_retry(call)
        except requests.RequestException as e:
            logger.warning("Catalogue network error: %s", e)
            return False, []

        if resp.status_code == 429:
            logger.warning("CDSE Catalogue rate limited (HTTP 429).")
            return False, []
        if resp.status_code not in (200, 201):
            logger.warning("Catalogue search failed: HTTP %d", resp.status_code)
            return False, []

        try:
            data = resp.json()
        except Exception as e:
            logger.warning("Catalogue response not JSON: %s", e)
            return False, []

        features = data.get("features", [])
        return True, features

    # ------------------------------------------------------------------
    # Processing API — image patch retrieval
    # ------------------------------------------------------------------

    def _patch_cache_key(self, bbox: List[float], date_from: str, date_to: str) -> str:
        raw = json.dumps({"bbox": bbox, "from": date_from, "to": date_to}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def fetch_patch(
        self,
        bbox: List[float],
        date_from: str,
        date_to: str,
        event_id: str,
        width_px: int = DEFAULT_S2_PATCH_WIDTH_PX,
        height_px: int = DEFAULT_S2_PATCH_HEIGHT_PX,
        max_cloud_pct: float = DEFAULT_S2_MAX_CLOUD_PCT,
    ) -> Tuple[bool, Optional[bytes]]:
        """Call the Processing API and return raw PNG bytes.

        Returns:
            (success, image_bytes_or_None).
        """
        payload = build_process_request_body(
            bbox, date_from, date_to, width_px, height_px, max_cloud_pct
        )

        def call(headers):
            return requests.post(
                self.process_url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )

        try:
            resp = self._handle_401_retry(call)
        except requests.RequestException as e:
            logger.warning("Processing API network error for %s: %s", event_id, e)
            return False, None

        if resp.status_code == 429:
            logger.warning("CDSE Processing API rate limited (HTTP 429) for %s.", event_id)
            return False, None
        if resp.status_code not in (200, 201):
            body_snippet = resp.text[:200] if resp.text else ""
            logger.warning(
                "Processing API failed: HTTP %d for %s. Body: %s",
                resp.status_code, event_id, body_snippet,
            )
            return False, None

        content_type = resp.headers.get("Content-Type", "")
        if "image" not in content_type and len(resp.content) < 100:
            logger.warning(
                "Processing API returned unexpected content-type '%s' for %s.",
                content_type, event_id,
            )
            return False, None

        return True, resp.content


def generate_image_fingerprint(centroid_lat: float, centroid_lon: float, date_range_days: int, patch_half_deg: float, width_px: int, height_px: int) -> str:
    """Generate a deterministic fingerprint representing the geospatial and configuration identity of the requested imagery."""
    # Normalize coordinates to 4 decimal places to prevent floating point jitter cache misses
    lat_norm = f"{centroid_lat:.4f}"
    lon_norm = f"{centroid_lon:.4f}"
    
    # Construct an identity string containing all parameters that affect the returned image pixels
    identity = f"{lat_norm}_{lon_norm}_{date_range_days}_{patch_half_deg}_{width_px}_{height_px}"
    
    # Generate SHA-256 fingerprint
    return hashlib.sha256(identity.encode('utf-8')).hexdigest()

def validate_strict_cache_metadata(
    metadata_path: Path,
    requested_fingerprint: str,
    centroid_lat: float,
    centroid_lon: float,
    date_range_days: int,
    patch_half_deg: float,
    width_px: int,
    height_px: int,
    ev_dt: datetime,
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """Validate a cached Sentinel-2 patch against its own sidecar metadata in strict mode.

    Returns:
        (is_valid, metadata_dict_or_None, rejection_reason_or_None)
    """
    if not metadata_path.exists():
        return False, None, f"Missing metadata sidecar: {metadata_path.name}"

    try:
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, None, f"Malformed metadata JSON in {metadata_path.name}: {e}"

    if not isinstance(meta, dict):
        return False, None, f"Metadata sidecar {metadata_path.name} is not a JSON object"

    # 1. Fingerprint verification
    cached_fp = meta.get("image_fingerprint")
    if cached_fp != requested_fingerprint:
        return False, None, f"Fingerprint mismatch: cached={cached_fp!r} != requested={requested_fingerprint!r}"

    # 2. Coordinates verification (4 decimal places tolerance)
    cached_lat = meta.get("centroid_lat")
    cached_lon = meta.get("centroid_lon")
    if cached_lat is None or cached_lon is None:
        return False, None, "Missing centroid coordinates in metadata sidecar"
    try:
        if abs(float(cached_lat) - centroid_lat) > 1e-4 or abs(float(cached_lon) - centroid_lon) > 1e-4:
            return False, None, f"Coordinate mismatch: ({cached_lat}, {cached_lon}) != ({centroid_lat}, {centroid_lon})"
    except (ValueError, TypeError):
        return False, None, "Invalid centroid coordinate types in metadata sidecar"

    # 3. Patch parameters verification
    if meta.get("date_range_days") != date_range_days:
        return False, None, f"date_range_days mismatch: cached={meta.get('date_range_days')} != requested={date_range_days}"
    if meta.get("patch_half_deg") != patch_half_deg:
        return False, None, f"patch_half_deg mismatch: cached={meta.get('patch_half_deg')} != requested={patch_half_deg}"
    if meta.get("width_px") != width_px or meta.get("height_px") != height_px:
        return False, None, f"Dimension mismatch: cached=({meta.get('width_px')}, {meta.get('height_px')}) != requested=({width_px}, {height_px})"

    # 4. Scene identity
    cached_scene_id = meta.get("scene_id")
    if not cached_scene_id or not isinstance(cached_scene_id, str):
        return False, None, "Missing or invalid scene_id in metadata sidecar"

    # 5. Acquisition timestamp and temporal eligibility
    cached_acq = meta.get("acquisition_time")
    if not cached_acq or not isinstance(cached_acq, str):
        return False, None, "Missing acquisition_time in metadata sidecar"
    try:
        cached_dt = datetime.fromisoformat(cached_acq.replace("Z", "+00:00"))
        if cached_dt.tzinfo is None:
            cached_dt = cached_dt.replace(tzinfo=timezone.utc)
    except ValueError as e:
        return False, None, f"Invalid acquisition_time format {cached_acq!r}: {e}"

    if cached_dt > ev_dt:
        return False, None, f"Temporal leakage: cached scene {cached_scene_id} is post-incident ({cached_dt} > {ev_dt})"

    return True, meta, None

def retrieve_sentinel2_patch(
    event_id: str,
    centroid_lat: float,
    centroid_lon: float,
    event_date: str,
    client: CDSEClient,
    output_dir: Path,
    date_range_days: int = DEFAULT_S2_DATE_RANGE_DAYS,
    patch_half_deg: float = DEFAULT_S2_PATCH_HALF_DEG,
    max_cloud_pct: float = DEFAULT_S2_MAX_CLOUD_PCT,
    width_px: int = DEFAULT_S2_PATCH_WIDTH_PX,
    height_px: int = DEFAULT_S2_PATCH_HEIGHT_PX,
    strict_eval_mode: bool = False,
) -> S2PatchResult:
    """Retrieve a Sentinel-2 L2A true-colour patch for a thermal event.

    Workflow:
      1. Compute patch bounding box.
      2. Build date window: [event_date - date_range_days, event_date + 1].
      3. Search catalogue for eligible scenes.
      4. Select best scene (lowest cloud cover, most recent among ties).
      5. Request processed PNG from Processing API.
      6. Persist image to disk; check existing cache first.

    Args:
        event_id: Stable event identifier (e.g., 'TE-EVT-00001').
        centroid_lat/lon: Event centroid coordinates.
        event_date: ISO date string (YYYY-MM-DD) of the event (last_detected).
        client: Authenticated CDSEClient instance.
        output_dir: Directory to save patch images.
        ...

    Returns:
        S2PatchResult with success flag, metadata, and output path.
    """
    bbox = compute_patch_bbox(centroid_lat, centroid_lon, patch_half_deg)

    # Build date window around event detection
    try:
        ev_dt = datetime.fromisoformat(event_date.replace("Z", "+00:00"))
        if ev_dt.tzinfo is None:
            ev_dt = ev_dt.replace(tzinfo=timezone.utc)
    except ValueError:
        try:
            ev_dt = datetime.strptime(event_date[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return S2PatchResult(
                event_id=event_id, scene_id="", acquisition_time="",
                cloud_cover_pct=None, patch_width_px=width_px, patch_height_px=height_px,
                output_path=None, success=False,
                error_message=f"Invalid event_date format: {event_date!r}",
                image_fingerprint="", source_lat=centroid_lat, source_lon=centroid_lon
            )

    from datetime import timedelta
    date_from = (ev_dt - timedelta(days=date_range_days)).strftime("%Y-%m-%dT00:00:00Z")
    if strict_eval_mode:
        date_to = ev_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        date_to = (ev_dt + timedelta(days=1)).strftime("%Y-%m-%dT23:59:59Z")

    # Check on-disk cache: skip API if patch file already exists
    safe_event_id = event_id.replace("/", "_").replace(":", "_")
    fingerprint = generate_image_fingerprint(
        centroid_lat, centroid_lon, date_range_days, patch_half_deg, width_px, height_px
    )
    if strict_eval_mode:
        fingerprint += "_STRICT"
    patch_filename = f"{safe_event_id}_{fingerprint}.png"
    patch_path = output_dir / patch_filename
    logger.info(f'Expected patch path: {patch_path}')

    if not strict_eval_mode and patch_path.exists():
        logger.info("Using cached patch for %s: %s", event_id, patch_path)
        return S2PatchResult(
            event_id=event_id, scene_id="cached", acquisition_time="",
            cloud_cover_pct=None, patch_width_px=width_px, patch_height_px=height_px,
            output_path=patch_path, success=True,
        )

    # Catalogue search
    cat_ok, features = client.search_scenes(bbox, date_from, date_to, max_cloud_pct)
    if not cat_ok:
        return S2PatchResult(
            event_id=event_id, scene_id="", acquisition_time="",
            cloud_cover_pct=None, patch_width_px=width_px, patch_height_px=height_px,
            output_path=None, success=False,
            error_message="Catalogue API request failed.",
            image_fingerprint=fingerprint, source_lat=centroid_lat, source_lon=centroid_lon
        )

    if not features:
        return S2PatchResult(
            event_id=event_id, scene_id="", acquisition_time="",
            cloud_cover_pct=None, patch_width_px=width_px, patch_height_px=height_px,
            output_path=None, success=False,
            error_message=f"No Sentinel-2 L2A scenes found within {date_range_days} days "
                          f"and cloud cover ≤{max_cloud_pct}%.",
        )

    if strict_eval_mode:
        valid_features = []
        for f in features:
            props = f.get("properties", {})
            dt_str = props.get("datetime") or props.get("created") or ""
            try:
                # Correctly handle Z, offsets, or timezone-naive
                acq_dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                if acq_dt.tzinfo is None:
                    acq_dt = acq_dt.replace(tzinfo=timezone.utc)
                    
                if acq_dt <= ev_dt:
                    valid_features.append(f)
            except ValueError as e:
                continue
        
        if not valid_features:
            return S2PatchResult(
                event_id=event_id, scene_id="", acquisition_time="",
                cloud_cover_pct=None, patch_width_px=width_px, patch_height_px=height_px,
                output_path=None, success=False,
                error_message="Temporal Leakage Prevention: No eligible pre-incident scenes found.",
                image_fingerprint=fingerprint, source_lat=centroid_lat, source_lon=centroid_lon
            )
        features = valid_features

    scene = select_best_scene(features)
    if scene is None:
        return S2PatchResult(
            event_id=event_id, scene_id="", acquisition_time="",
            cloud_cover_pct=None, patch_width_px=width_px, patch_height_px=height_px,
            output_path=None, success=False,
            error_message="Scene selection returned no result.",
            image_fingerprint=fingerprint, source_lat=centroid_lat, source_lon=centroid_lon
        )

    if strict_eval_mode and patch_path.exists():
        metadata_path = patch_path.with_suffix('.json')
        is_valid, meta, reason = validate_strict_cache_metadata(
            metadata_path=metadata_path,
            requested_fingerprint=fingerprint,
            centroid_lat=centroid_lat,
            centroid_lon=centroid_lon,
            date_range_days=date_range_days,
            patch_half_deg=patch_half_deg,
            width_px=width_px,
            height_px=height_px,
            ev_dt=ev_dt,
        )
        if is_valid and meta is not None:
            logger.info("Using cached patch after strict validation for %s: %s (scene_id=%s)", event_id, patch_path, meta["scene_id"])
            return S2PatchResult(
                event_id=event_id,
                scene_id=meta["scene_id"],
                acquisition_time=meta["acquisition_time"],
                cloud_cover_pct=meta.get("cloud_cover_pct"),
                patch_width_px=width_px,
                patch_height_px=height_px,
                output_path=patch_path,
                success=True,
                image_fingerprint=fingerprint,
                source_lat=centroid_lat,
                source_lon=centroid_lon,
            )
        else:
            logger.warning("Strict cache rejected for %s (%s): %s", event_id, patch_path, reason)

    # Processing API request
    proc_ok, image_bytes = client.fetch_patch(
        bbox=bbox,
        date_from=date_from,
        date_to=date_to,
        event_id=event_id,
        width_px=width_px,
        height_px=height_px,
        max_cloud_pct=max_cloud_pct,
    )

    if not proc_ok or not image_bytes:
        return S2PatchResult(
            event_id=event_id, scene_id=scene.scene_id,
            acquisition_time=scene.acquisition_time,
            cloud_cover_pct=scene.cloud_cover_pct,
            patch_width_px=width_px, patch_height_px=height_px,
            output_path=None, success=False,
            error_message="Processing API returned no image data.",
            image_fingerprint=fingerprint, source_lat=centroid_lat, source_lon=centroid_lon
        )

    # Save to disk
    output_dir.mkdir(parents=True, exist_ok=True)
    patch_path.write_bytes(image_bytes)
    
    # Write trustworthy scene metadata sidecar for strict mode identity validation
    import json
    metadata_path = patch_path.with_suffix('.json')
    metadata_path.write_text(json.dumps({
        "scene_id": scene.scene_id,
        "acquisition_time": scene.acquisition_time,
        "cloud_cover_pct": scene.cloud_cover_pct,
        "image_fingerprint": fingerprint,
        "centroid_lat": centroid_lat,
        "centroid_lon": centroid_lon,
        "date_range_days": date_range_days,
        "patch_half_deg": patch_half_deg,
        "width_px": width_px,
        "height_px": height_px,
    }))

    return S2PatchResult(
        event_id=event_id,
        scene_id=scene.scene_id,
        acquisition_time=scene.acquisition_time,
        cloud_cover_pct=scene.cloud_cover_pct,
        patch_width_px=width_px,
        patch_height_px=height_px,
        output_path=patch_path,
        success=True,
        image_fingerprint=fingerprint,
        source_lat=centroid_lat,
        source_lon=centroid_lon,
    )
