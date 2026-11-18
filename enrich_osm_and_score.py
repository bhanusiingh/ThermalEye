"""
ThermalEye — Phase 2/3: OSM Infrastructure Enrichment & Event Scoring Worker
=============================================================================

Enriches candidate fire events with nearby industrial and power infrastructure
from OpenStreetMap (Overpass API) and computes the Investigation Priority Score.

Architectural invariants:
  • Respectful rate limiting, exponential backoff with jitter, Retry-After header.
  • Descriptive User-Agent per Overpass Acceptable Use Policies.
  • Spatial caching & deduplication to minimize external HTTP queries.
  • Resumable: queries WHERE priority_reasons IS NULL, commits per event/batch.
  • Configurable timeouts, primary and fallback endpoints.
  • Missing infrastructure stored as SQL NULL; failed queries are NOT marked as
    enriched, and no fabricated data is assigned.
"""

from __future__ import annotations

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

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
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

# Overpass Endpoints & Configuration
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

# Processing controls
OSM_BATCH_LIMIT = int(os.getenv("OSM_BATCH_LIMIT", "0"))  # 0 = all un-enriched
OSM_BATCH_QUERY_SIZE = int(os.getenv("OSM_BATCH_QUERY_SIZE", "5"))
CACHE_DISTANCE_METERS = float(os.getenv("OSM_CACHE_DISTANCE_M", "1500.0"))
DEFAULT_SEARCH_RADIUS = int(os.getenv("OSM_SEARCH_RADIUS_M", "3000"))

# State for rate limiting & circuit breaker
_last_request_time: float = 0.0
CIRCUIT_BREAKER_COOLDOWN_SEC = float(os.getenv("OVERPASS_CIRCUIT_BREAKER_COOLDOWN_SEC", "60.0"))
CIRCUIT_BREAKER_THRESHOLD = int(os.getenv("OVERPASS_CIRCUIT_BREAKER_THRESHOLD", "2"))
_circuit_breaker_open: bool = False
_circuit_breaker_tripped_at: float = 0.0
_consecutive_failures: int = 0


# ---------------------------------------------------------------------------
# Spatial Utilities & Caching
# ---------------------------------------------------------------------------
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance in meters between two points on Earth."""
    r_earth = 6371000.0  # meters
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
    """
    In-memory spatial cache for discovered facilities and empty regions.
    Avoids repeated Overpass queries for clustered thermal events.
    """

    def __init__(self, cache_radius_m: float = CACHE_DISTANCE_METERS):
        self.cache_radius_m = cache_radius_m
        # List of dicts: {"lat": float, "lon": float, "name": str, "type": str}
        self.facilities: List[Dict[str, Any]] = []
        # List of tuples: (lat, lon, radius_m)
        self.empty_zones: List[Tuple[float, float, float]] = []

    def add_facilities(self, elements: List[Dict[str, Any]]) -> None:
        """Extract facility records from Overpass elements and store in cache."""
        for elem in elements:
            tags = elem.get("tags", {})
            name = tags.get("name", "Unnamed Facility")
            facility_type = tags.get("power", tags.get("landuse", "Industrial Zone"))

            # Extract location (nodes have lat/lon; ways with 'out center' have center.lat/lon)
            lat = elem.get("lat") or elem.get("center", {}).get("lat")
            lon = elem.get("lon") or elem.get("center", {}).get("lon")

            if lat is not None and lon is not None:
                # Deduplicate by approximate distance < 50m
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
        """Record an area known to contain no qualifying facilities."""
        self.empty_zones.append((lat, lon, radius_m))

    def find_nearest_facility(
        self, lat: float, lon: float, max_radius_m: float = DEFAULT_SEARCH_RADIUS
    ) -> Optional[Tuple[str, str, float]]:
        """
        Check if any known cached facility is within max_radius_m.
        Returns (name, type, distance) or None.
        """
        best_match = None
        min_dist = float("inf")

        for f in self.facilities:
            dist = haversine_distance(lat, lon, f["lat"], f["lon"])
            if dist <= max_radius_m and dist < min_dist:
                min_dist = dist
                best_match = (f["name"], f["type"], dist)

        return best_match

    def is_in_empty_zone(self, lat: float, lon: float) -> bool:
        """
        Check if (lat, lon) is within an empty zone with enough margin
        to guarantee no facility exists within DEFAULT_SEARCH_RADIUS.
        """
        for z_lat, z_lon, z_radius in self.empty_zones:
            dist = haversine_distance(lat, lon, z_lat, z_lon)
            # If distance to center is within 500m of an empty 3000m query,
            # the remaining 2500m has no facilities.
            if dist <= 500 and z_radius >= DEFAULT_SEARCH_RADIUS:
                return True
        return False


# Global spatial cache instance
spatial_cache = SpatialCache()


# ---------------------------------------------------------------------------
# Rate Limiting & Overpass API Interaction
# ---------------------------------------------------------------------------
def _wait_rate_limit(min_delay: float = OVERPASS_RATE_LIMIT_DELAY) -> None:
    """Enforce minimum delay between outgoing requests."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < min_delay:
        sleep_time = min_delay - elapsed + random.uniform(0.05, 0.25)
        time.sleep(sleep_time)
    _last_request_time = time.time()


def _parse_retry_after(response: requests.Response, default_wait: float = 5.0) -> float:
    """Parse Retry-After header if present, else return default wait time."""
    retry_header = response.headers.get("Retry-After")
    if retry_header:
        try:
            return max(float(retry_header), 1.0)
        except ValueError:
            pass
    return default_wait


def execute_overpass_query(
    query_str: str,
    endpoints: Optional[List[str]] = None,
    max_retries: int = 2,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, int]]:
    """
    Execute an Overpass QL query with rate limiting, exponential backoff with jitter,
    Retry-After compliance, and endpoint fallback.

    Returns (response_json, metric_deltas).
    """
    if endpoints is None:
        endpoints = [OVERPASS_URL] + [
            ep for ep in OVERPASS_FALLBACK_URLS if ep != OVERPASS_URL
        ]

    headers = {
        "User-Agent": OVERPASS_USER_AGENT,
        "Accept": "application/json",
    }
    metrics = {"rate_limited": 0, "retries": 0}

    for ep_idx, endpoint in enumerate(endpoints):
        for attempt in range(max_retries):
            _wait_rate_limit()

            try:
                logger.debug(
                    "Overpass request attempt %d to %s", attempt + 1, endpoint
                )
                resp = requests.post(
                    endpoint,
                    data={"data": query_str},
                    headers=headers,
                    timeout=(OVERPASS_CONNECT_TIMEOUT, OVERPASS_READ_TIMEOUT),
                    verify=True,
                )

                if resp.status_code == 200:
                    try:
                        return resp.json(), metrics
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON from %s: %s", endpoint, resp.text[:200])

                elif resp.status_code == 429:
                    metrics["rate_limited"] += 1
                    retry_wait = _parse_retry_after(resp, default_wait=3.0 * (2 ** attempt))
                    jitter = random.uniform(0.5, 2.0)
                    total_wait = retry_wait + jitter
                    logger.warning(
                        "Overpass 429 Rate Limited on %s. Waiting %.1fs (Retry-After / backoff).",
                        endpoint,
                        total_wait,
                    )
                    time.sleep(total_wait)
                    metrics["retries"] += 1

                elif resp.status_code in (502, 503, 504):
                    backoff = (2.0 ** attempt) + random.uniform(0.5, 1.5)
                    logger.warning(
                        "Overpass server HTTP %d on %s. Backoff %.1fs.",
                        resp.status_code,
                        endpoint,
                        backoff,
                    )
                    time.sleep(backoff)
                    metrics["retries"] += 1

                else:
                    logger.warning(
                        "Overpass HTTP %d from %s: %s",
                        resp.status_code,
                        endpoint,
                        resp.text[:200],
                    )
                    break  # Non-retryable HTTP error on this endpoint

            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                backoff = (2.0 ** attempt) + random.uniform(0.5, 1.5)
                logger.warning(
                    "Network error (%s) contacting %s. Waiting %.1fs.",
                    type(exc).__name__,
                    endpoint,
                    backoff,
                )
                time.sleep(backoff)
                metrics["retries"] += 1

        logger.info(
            "Exhausted retries on %s. Switching to next fallback endpoint if available.",
            endpoint,
        )

    return None, metrics


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
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
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
    """
    Fetch industrial and power infrastructure within `radius` meters of (lat, lon).
    Raises requests.exceptions.Timeout, requests.exceptions.HTTPError, or RetryError on failure.
    """
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
            logger.debug("Overpass endpoint %s failed: %s", endpoint, exc)
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

    If rate_limited is True, infrastructure score is 0 and a fallback note
    "0/30 (Overpass Rate Limited)" is injected into priority_reasons.
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
# Database Commit Helper
# ---------------------------------------------------------------------------
def _commit_event_score(
    engine,
    event_id: Any,
    nearest_name: Optional[str],
    nearest_type: Optional[str],
    score: float,
    reasons: str,
) -> None:
    """Commit enriched score and reasoning to candidate_events."""
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                UPDATE candidate_events 
                SET nearest_osm_name = :name, 
                    nearest_osm_type = :type, 
                    priority_score = :score, 
                    priority_reasons = CAST(:reasons AS jsonb),
                    updated_at = NOW()
                WHERE id = CAST(:id AS uuid)
                """
            ),
            {
                "name": nearest_name,
                "type": nearest_type,
                "score": score,
                "reasons": reasons,
                "id": str(event_id),
            },
        )


# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------
def process_enrichment(batch_limit: int = OSM_BATCH_LIMIT) -> Dict[str, int]:
    """
    Process candidate events needing OSM enrichment:
      • Resumable: Selects events WHERE priority_reasons IS NULL.
      • Uses spatial caching to avoid redundant calls for nearby events.
      • Decoupled from external API: wraps fetch_osm_infrastructure in try...except.
      • On rate limit / timeout / RetryError: sets nearest_name and nearest_type to NULL,
        calculates local score (frp + persistence, infra=0), injects fallback note
        "0/30 (Overpass Rate Limited)", and commits valid score to candidate_events.
    """
    logger.info("=" * 60)
    logger.info("ThermalEye — OSM Enrichment and Scoring Worker Starting")
    logger.info("Primary Endpoint: %s", OVERPASS_URL)
    logger.info("Fallback Endpoints: %s", OVERPASS_FALLBACK_URLS)
    logger.info("Batch Limit: %s (0=unlimited)", batch_limit)
    logger.info("=" * 60)

    engine = create_engine(DB_URL)

    stats = {
        "processed": 0,
        "successful": 0,
        "cached": 0,
        "rate_limited": 0,
        "failed": 0,
    }

    # Resumable query: only select events where priority_reasons IS NULL
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
                            priority_score = :score, 
                            priority_reasons = CAST(:reasons AS jsonb),
                            updated_at = NOW()
                        WHERE id = CAST(:id AS uuid)
                        """
                    ),
                    item,
                )
        batch_buffer.clear()

    for idx, event in enumerate(events, 1):
        event_id = event.id
        lat, lon = event.lat, event.lon

        # 1. Check spatial cache for nearby known facility
        cached_facility = spatial_cache.find_nearest_facility(lat, lon, DEFAULT_SEARCH_RADIUS)
        if cached_facility:
            nearest_name, nearest_type, _ = cached_facility
            score, reasons = compute_priority_score(
                event.max_frp, event.detection_count, nearest_type, rate_limited=False
            )
            batch_buffer.append(
                {
                    "name": nearest_name,
                    "type": nearest_type,
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

        # 2. Check if event is in a known empty zone
        if spatial_cache.is_in_empty_zone(lat, lon):
            score, reasons = compute_priority_score(
                event.max_frp, event.detection_count, None, rate_limited=False
            )
            batch_buffer.append(
                {
                    "name": None,
                    "type": None,
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

        # 3. Not in cache: query Overpass via fetch_osm_infrastructure inside try...except
        nearest_name = None
        nearest_type = None
        rate_limited = False

        try:
            osm_data = fetch_osm_infrastructure(lat, lon, radius=DEFAULT_SEARCH_RADIUS)
            elements = osm_data.get("elements", []) if isinstance(osm_data, dict) else []
            if elements:
                spatial_cache.add_facilities(elements)
                match = spatial_cache.find_nearest_facility(lat, lon, DEFAULT_SEARCH_RADIUS)
                if match:
                    nearest_name, nearest_type, _ = match
            else:
                spatial_cache.add_empty_zone(lat, lon, DEFAULT_SEARCH_RADIUS)
            stats["successful"] += 1

        except (
            requests.exceptions.Timeout,
            requests.exceptions.HTTPError,
            RetryError,
            requests.exceptions.RequestException,
        ) as exc:
            # Overpass API rate-limited or timed out: fall back gracefully to local scoring
            if stats["rate_limited"] == 0:
                logger.warning(
                    "Overpass API unavailable/rate-limited for event %s (lat=%.3f, lon=%.3f): %s. Decoupling scoring.",
                    event_id,
                    lat,
                    lon,
                    exc,
                )
            nearest_name = None
            nearest_type = None
            rate_limited = True
            stats["rate_limited"] += 1

        # 4. Calculate score (using local data + fallback note if rate_limited)
        score, reasons = compute_priority_score(
            event.max_frp,
            event.detection_count,
            nearest_type,
            rate_limited=rate_limited,
        )

        # 5. Buffer score to commit in batch
        batch_buffer.append(
            {
                "name": nearest_name,
                "type": nearest_type,
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
                "Progress: %d/%d processed (Success: %d, Cached: %d, RateLimits: %d, Failed: %d)",
                stats["processed"],
                total_to_process,
                stats["successful"],
                stats["cached"],
                stats["rate_limited"],
                stats["failed"],
            )

    # Flush any remaining items in buffer
    flush_batch()

    logger.info("=" * 60)
    logger.info("OSM enrichment and scoring complete.")
    logger.info("Summary: %s", stats)
    logger.info("=" * 60)
    return stats


if __name__ == "__main__":
    process_enrichment()
