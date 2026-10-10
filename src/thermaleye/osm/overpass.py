"""OpenStreetMap industrial infrastructure context module using public Overpass API."""

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import requests
import pandas as pd

from thermaleye.config import (
    OVERPASS_API_URL,
    DEFAULT_OSM_SEARCH_RADIUS_M,
    DEFAULT_OSM_TIMEOUT,
    DEFAULT_OSM_CACHE_PATH,
    DEFAULT_OSM_OUTPUT_PATH,
    DEFAULT_MAX_OSM_EVENTS,
    DEFAULT_OSM_REQUEST_DELAY,
)
from thermaleye.temporal.grouping import haversine_distance_km

logger = logging.getLogger(__name__)

OSM_FEATURE_COLUMNS = [
    "nearest_industrial_feature_km",
    "nearest_power_plant_km",
    "industrial_features_500m",
    "industrial_features_1km",
    "industrial_landuse_present",
    "osm_data_available",
]


class DiskCache:
    """Local JSON cache for Overpass queries to minimize external network requests."""

    def __init__(self, cache_path: Optional[Path] = DEFAULT_OSM_CACHE_PATH) -> None:
        self.cache_path = cache_path
        self._cache: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self.cache_path and self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except Exception as e:
                logger.warning("Could not read OSM cache at %s: %s", self.cache_path, e)
                self._cache = {}

    def get(self, key: str) -> Optional[Any]:
        return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        self._cache[key] = value

    def save(self) -> None:
        if self.cache_path:
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = self.cache_path.with_suffix(".tmp")
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(self._cache, f, indent=2)
                temp_path.replace(self.cache_path)
            except Exception as e:
                logger.warning("Could not save OSM cache to %s: %s", self.cache_path, e)


def build_overpass_query(lat: float, lon: float, radius_m: int = DEFAULT_OSM_SEARCH_RADIUS_M, timeout: int = DEFAULT_OSM_TIMEOUT) -> str:
    """Construct Overpass QL query searching for industrial and power infrastructure around a coordinate."""
    return f"""[out:json][timeout:{timeout}];
(
  nwr["landuse"="industrial"](around:{radius_m},{lat},{lon});
  nwr["industrial"](around:{radius_m},{lat},{lon});
  nwr["power"="plant"](around:{radius_m},{lat},{lon});
  nwr["power"="generator"](around:{radius_m},{lat},{lon});
  nwr["man_made"="works"](around:{radius_m},{lat},{lon});
  nwr["man_made"="chimney"](around:{radius_m},{lat},{lon});
);
out center tags;
"""


class OverpassClient:
    """Client for the public Overpass API with caching and rate limiting."""

    def __init__(
        self,
        api_url: str = OVERPASS_API_URL,
        timeout: int = DEFAULT_OSM_TIMEOUT,
        request_delay: float = DEFAULT_OSM_REQUEST_DELAY,
        cache_path: Optional[Path] = DEFAULT_OSM_CACHE_PATH,
    ) -> None:
        self.api_url = api_url
        self.timeout = timeout
        self.request_delay = request_delay
        self.cache = DiskCache(cache_path)
        self._last_request_time: float = 0.0

    def _cache_key(self, lat: float, lon: float, radius_m: int) -> str:
        return f"{round(lat, 4)}_{round(lon, 4)}_{int(radius_m)}"

    def query_nearby_industrial(
        self,
        lat: float,
        lon: float,
        radius_m: int = DEFAULT_OSM_SEARCH_RADIUS_M,
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """Query Overpass API for industrial infrastructure around coordinate.

        Returns:
            Tuple of (success_bool, elements_list).
            If query fails or times out, returns (False, []).
        """
        key = self._cache_key(lat, lon, radius_m)
        cached_result = self.cache.get(key)
        if cached_result is not None:
            return True, cached_result

        # Respect Overpass polite usage policy with rate-limiting delay
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)

        ql_query = build_overpass_query(lat, lon, radius_m, self.timeout)

        headers = {
            "User-Agent": "ThermalEye/0.1.0 (https://github.com/thermaleye; thermal-anomaly-analysis)",
        }

        try:
            self._last_request_time = time.time()
            resp = requests.post(
                self.api_url,
                data={"data": ql_query},
                headers=headers,
                timeout=self.timeout + 5,
            )

            if resp.status_code == 200:
                try:
                    payload = resp.json()
                    elements = payload.get("elements", [])
                    self.cache.set(key, elements)
                    self.cache.save()
                    return True, elements
                except Exception as e:
                    logger.warning("Failed to decode JSON from Overpass API: %s", e)
                    return False, []
            elif resp.status_code == 429:
                logger.warning("Overpass API rate limit reached (HTTP 429).")
                return False, []
            elif resp.status_code in (504, 408):
                logger.warning("Overpass API query timed out (HTTP %d).", resp.status_code)
                return False, []
            else:
                logger.warning("Overpass API query failed with HTTP %d.", resp.status_code)
                return False, []

        except requests.RequestException as e:
            logger.warning("Network failure during Overpass query: %s", e)
            return False, []


def extract_features_from_elements(
    centroid_lat: float,
    centroid_lon: float,
    elements: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Calculate geospatial proximity and counts for industrial and power facilities."""
    seen_features: Set[Tuple[str, Any]] = set()

    industrial_distances: List[float] = []
    power_plant_distances: List[float] = []
    industrial_500m_count = 0
    industrial_1km_count = 0
    has_industrial_landuse = False

    for elem in elements:
        elem_type = elem.get("type", "")
        elem_id = elem.get("id", "")
        feat_key = (elem_type, elem_id)

        # Avoid counting the same feature multiple times
        if feat_key in seen_features:
            continue
        seen_features.add(feat_key)

        # Retrieve coordinate (node lat/lon or way/relation center)
        e_lat: Optional[float] = None
        e_lon: Optional[float] = None

        if "lat" in elem and "lon" in elem:
            e_lat = float(elem["lat"])
            e_lon = float(elem["lon"])
        elif "center" in elem and isinstance(elem["center"], dict):
            center = elem["center"]
            if "lat" in center and "lon" in center:
                e_lat = float(center["lat"])
                e_lon = float(center["lon"])

        if e_lat is None or e_lon is None:
            continue

        dist_km = haversine_distance_km(centroid_lat, centroid_lon, e_lat, e_lon)
        tags = elem.get("tags", {})

        is_power = tags.get("power") in ("plant", "generator")
        is_landuse_ind = tags.get("landuse") == "industrial"
        is_general_ind = (
            is_power
            or is_landuse_ind
            or ("industrial" in tags)
            or tags.get("man_made") in ("works", "chimney")
        )

        if is_general_ind:
            industrial_distances.append(dist_km)
            if dist_km <= 0.5:
                industrial_500m_count += 1
            if dist_km <= 1.0:
                industrial_1km_count += 1

        if is_power:
            power_plant_distances.append(dist_km)

        if is_landuse_ind:
            has_industrial_landuse = True

    nearest_industrial_km = (
        round(min(industrial_distances), 3) if industrial_distances else None
    )
    nearest_power_km = (
        round(min(power_plant_distances), 3) if power_plant_distances else None
    )

    return {
        "nearest_industrial_feature_km": nearest_industrial_km,
        "nearest_power_plant_km": nearest_power_km,
        "industrial_features_500m": industrial_500m_count,
        "industrial_features_1km": industrial_1km_count,
        "industrial_landuse_present": has_industrial_landuse,
        "osm_data_available": True,
    }


def enrich_events_with_osm(
    events_df: pd.DataFrame,
    client: Optional[OverpassClient] = None,
    max_events: Optional[int] = DEFAULT_MAX_OSM_EVENTS,
    radius_m: int = DEFAULT_OSM_SEARCH_RADIUS_M,
) -> pd.DataFrame:
    """Enrich candidate thermal events DataFrame with OSM industrial context features."""
    if events_df.empty:
        empty_df = events_df.copy()
        for col in OSM_FEATURE_COLUMNS:
            empty_df[col] = pd.Series(dtype="object")
        return empty_df

    if client is None:
        client = OverpassClient()

    enriched = events_df.copy()

    # Pre-populate empty feature columns with None
    for col in OSM_FEATURE_COLUMNS:
        enriched[col] = None
    enriched["osm_data_available"] = False

    total_events = len(enriched)
    limit = max_events if max_events is not None else total_events

    for idx in range(total_events):
        if idx >= limit:
            # Event is skipped due to max_events limit
            enriched.at[idx, "osm_data_available"] = False
            continue

        lat = float(enriched.at[idx, "centroid_lat"])
        lon = float(enriched.at[idx, "centroid_lon"])

        success, elements = client.query_nearby_industrial(lat, lon, radius_m=radius_m)

        if success:
            feat_dict = extract_features_from_elements(lat, lon, elements)
            for k, v in feat_dict.items():
                enriched.at[idx, k] = v
        else:
            # Query failed, timed out, or hit rate limits
            enriched.at[idx, "nearest_industrial_feature_km"] = None
            enriched.at[idx, "nearest_power_plant_km"] = None
            enriched.at[idx, "industrial_features_500m"] = None
            enriched.at[idx, "industrial_features_1km"] = None
            enriched.at[idx, "industrial_landuse_present"] = None
            enriched.at[idx, "osm_data_available"] = False

    return enriched


def run_osm_enrichment(
    input_events_path: Path,
    output_events_path: Path = DEFAULT_OSM_OUTPUT_PATH,
    max_events: Optional[int] = DEFAULT_MAX_OSM_EVENTS,
    radius_m: int = DEFAULT_OSM_SEARCH_RADIUS_M,
    client: Optional[OverpassClient] = None,
) -> pd.DataFrame:
    """Execute OSM enrichment pipeline on an events CSV and save to data/processed/."""
    if not input_events_path.exists():
        raise FileNotFoundError(f"Input events CSV does not exist: {input_events_path}")

    events_df = pd.read_csv(input_events_path)
    enriched_df = enrich_events_with_osm(
        events_df=events_df,
        client=client,
        max_events=max_events,
        radius_m=radius_m,
    )

    output_events_path.parent.mkdir(parents=True, exist_ok=True)
    enriched_df.to_csv(output_events_path, index=False)
    return enriched_df
