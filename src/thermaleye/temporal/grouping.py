"""Spatial-temporal grouping and feature extraction for FIRMS thermal detections."""

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import pandas as pd
import numpy as np

from thermaleye.temporal.persistence import calculate_persistence_score

DEFAULT_SPATIAL_THRESHOLD_KM = 2.5
DEFAULT_MAX_TIME_GAP_HOURS = 36.0
DEFAULT_PROCESSED_DATA_DIR = Path("data/processed")
DEFAULT_EVENTS_OUTPUT_PATH = DEFAULT_PROCESSED_DATA_DIR / "thermal_events.csv"
DEFAULT_GROUPED_OUTPUT_PATH = DEFAULT_PROCESSED_DATA_DIR / "grouped_detections.csv"


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two coordinates in kilometers using Haversine formula."""
    if lat1 == lat2 and lon1 == lon2:
        return 0.0

    R = 6371.0  # Earth's mean spherical radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    # Ensure numerical stability
    a = min(1.0, max(0.0, a))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def parse_confidence_score(val: Any) -> Optional[float]:
    """Parse FIRMS confidence into numeric percentage [0.0, 100.0].

    VIIRS NRT categoricals:
        'l' / 'low'     -> 30.0
        'n' / 'nominal' -> 60.0
        'h' / 'high'    -> 90.0
    Numeric values (0-100) are returned directly.
    """
    if pd.isna(val):
        return None

    # Handle direct numeric values
    if isinstance(val, (int, float, np.number)):
        try:
            num = float(val)
            if 0.0 <= num <= 100.0:
                return num
        except (ValueError, TypeError):
            pass

    # Handle string values
    val_str = str(val).strip().lower()
    if not val_str:
        return None

    mapping = {
        "l": 30.0,
        "low": 30.0,
        "n": 60.0,
        "nominal": 60.0,
        "h": 90.0,
        "high": 90.0,
    }
    if val_str in mapping:
        return mapping[val_str]

    try:
        num = float(val_str)
        if 0.0 <= num <= 100.0:
            return num
    except ValueError:
        pass

    return None


def format_acq_datetime(date_val: Any, time_val: Any) -> Optional[pd.Timestamp]:
    """Combine date and time fields into a UTC Timestamp."""
    if pd.isna(date_val):
        return None

    date_str = str(date_val).strip()

    # Normalize time into HHMM string
    if pd.isna(time_val):
        time_str = "0000"
    else:
        try:
            # Handle float or int representation (e.g. 730 or 730.0)
            time_int = int(float(time_val))
            time_str = f"{time_int:04d}"
        except (ValueError, TypeError):
            time_str = str(time_val).strip().zfill(4)

    if len(time_str) < 4:
        time_str = time_str.zfill(4)
    elif len(time_str) > 4:
        time_str = time_str[:4]

    combined = f"{date_str} {time_str[:2]}:{time_str[2:]}:00"
    try:
        return pd.to_datetime(combined)
    except Exception:
        # Fallback to date only
        try:
            return pd.to_datetime(date_str)
        except Exception:
            return None


class DisjointSet:
    """Disjoint Set Union (Union-Find) for connected component clustering."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, i: int) -> int:
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int) -> None:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i == root_j:
            return
        if self.rank[root_i] < self.rank[root_j]:
            self.parent[root_i] = root_j
        elif self.rank[root_i] > self.rank[root_j]:
            self.parent[root_j] = root_i
        else:
            self.parent[root_j] = root_i
            self.rank[root_i] += 1


@dataclass
class EventFeatures:
    """Calculated temporal and spatial features for a candidate thermal event."""

    event_id: str
    centroid_lat: float
    centroid_lon: float
    detection_count: int
    active_days: int
    first_detected: str
    last_detected: str
    duration_days: float
    mean_frp: Optional[float]
    max_frp: Optional[float]
    mean_firms_confidence: Optional[float]
    persistence_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "centroid_lat": self.centroid_lat,
            "centroid_lon": self.centroid_lon,
            "detection_count": self.detection_count,
            "active_days": self.active_days,
            "first_detected": self.first_detected,
            "last_detected": self.last_detected,
            "duration_days": self.duration_days,
            "mean_frp": self.mean_frp,
            "max_frp": self.max_frp,
            "mean_firms_confidence": self.mean_firms_confidence,
            "persistence_score": self.persistence_score,
        }


EVENT_COLUMNS = [
    "event_id",
    "centroid_lat",
    "centroid_lon",
    "detection_count",
    "active_days",
    "first_detected",
    "last_detected",
    "duration_days",
    "mean_frp",
    "max_frp",
    "mean_firms_confidence",
    "persistence_score",
]


def clean_detections(df: pd.DataFrame) -> pd.DataFrame:
    """Validate coordinates, deduplicate observations, and build timestamps."""
    if df.empty:
        return df.copy()

    # Verify minimum required columns
    required_cols = ["latitude", "longitude", "acq_date"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Input DataFrame missing required column: '{col}'")

    cleaned = df.copy()

    # Drop explicit duplicates
    dedup_subset = [c for c in ["latitude", "longitude", "acq_date", "acq_time", "satellite"] if c in cleaned.columns]
    if dedup_subset:
        cleaned = cleaned.drop_duplicates(subset=dedup_subset).reset_index(drop=True)
    else:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # Coerce and validate latitude and longitude
    cleaned["latitude"] = pd.to_numeric(cleaned["latitude"], errors="coerce")
    cleaned["longitude"] = pd.to_numeric(cleaned["longitude"], errors="coerce")

    valid_coords = (
        cleaned["latitude"].notna()
        & cleaned["longitude"].notna()
        & (cleaned["latitude"] >= -90.0)
        & (cleaned["latitude"] <= 90.0)
        & (cleaned["longitude"] >= -180.0)
        & (cleaned["longitude"] <= 180.0)
    )
    cleaned = cleaned[valid_coords].copy()

    if cleaned.empty:
        return cleaned
        
    from thermaleye.geo.boundary import filter_points_by_country
    cleaned = filter_points_by_country(cleaned, lat_col="latitude", lon_col="longitude", country_name="India")
    if cleaned.empty:
        return cleaned

    # Build acq_datetime
    time_col = cleaned["acq_time"] if "acq_time" in cleaned.columns else [None] * len(cleaned)
    timestamps = [
        format_acq_datetime(d, t)
        for d, t in zip(cleaned["acq_date"], time_col)
    ]
    cleaned["acq_datetime"] = timestamps
    cleaned = cleaned[cleaned["acq_datetime"].notna()].copy()
    cleaned = cleaned.sort_values(by="acq_datetime").reset_index(drop=True)

    return cleaned


def group_thermal_detections(
    df: pd.DataFrame,
    spatial_distance_km: float = DEFAULT_SPATIAL_THRESHOLD_KM,
    max_time_gap_hours: float = DEFAULT_MAX_TIME_GAP_HOURS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Group nearby detections into candidate thermal events and calculate temporal features.

    Args:
        df: Input FIRMS detections DataFrame.
        spatial_distance_km: Maximum geospatial distance (km) between detections.
        max_time_gap_hours: Maximum elapsed hours between consecutive linked detections.

    Returns:
        Tuple of (events_df, enriched_detections_df).
    """
    if df.empty:
        empty_events = pd.DataFrame(columns=EVENT_COLUMNS)
        empty_detections = df.copy()
        empty_detections["event_id"] = pd.Series(dtype="object")
        return empty_events, empty_detections

    cleaned = clean_detections(df)
    if cleaned.empty:
        empty_events = pd.DataFrame(columns=EVENT_COLUMNS)
        empty_detections = df.copy()
        empty_detections["event_id"] = pd.Series(dtype="object")
        return empty_events, empty_detections

    n = len(cleaned)
    dsu = DisjointSet(n)
    time_gap_seconds = max_time_gap_hours * 3600.0

    lats = cleaned["latitude"].to_numpy()
    lons = cleaned["longitude"].to_numpy()
    times = cleaned["acq_datetime"].to_numpy()

    # Pre-convert timestamps to epoch seconds for fast difference calculation
    epoch_seconds = times.astype("datetime64[s]").astype(np.int64)

    # 1 degree latitude is approximately 111 km
    lat_degree_bound = spatial_distance_km / 110.0

    for i in range(n):
        t_i = epoch_seconds[i]
        lat_i = lats[i]
        lon_i = lons[i]

        for j in range(i + 1, n):
            # Since array is chronologically sorted, if time delta exceeds threshold, stop inner loop
            dt = epoch_seconds[j] - t_i
            if dt > time_gap_seconds:
                break

            # Fast bounding prune on latitude
            if abs(lats[j] - lat_i) > lat_degree_bound:
                continue

            dist = haversine_distance_km(lat_i, lon_i, lats[j], lons[j])
            if dist <= spatial_distance_km:
                dsu.union(i, j)

    # Assemble clusters from disjoint set roots
    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        root = dsu.find(i)
        clusters.setdefault(root, []).append(i)

    # Sort clusters deterministically for stable identifiers
    # Ordering tuple: (earliest_time, mean_lat, mean_lon)
    sorted_cluster_roots = sorted(
        clusters.keys(),
        key=lambda r: (
            cleaned.loc[clusters[r], "acq_datetime"].min(),
            cleaned.loc[clusters[r], "latitude"].mean(),
            cleaned.loc[clusters[r], "longitude"].mean(),
        ),
    )

    events_list: List[Dict[str, Any]] = []
    cleaned["event_id"] = ""

    for idx, root in enumerate(sorted_cluster_roots, start=1):
        event_id = f"TE-EVT-{idx:05d}"
        member_indices = clusters[root]
        cleaned.loc[member_indices, "event_id"] = event_id
        cluster_df = cleaned.loc[member_indices]

        # Calculate event features
        det_count = len(cluster_df)
        active_days = int(cluster_df["acq_date"].nunique())
        first_dt = cluster_df["acq_datetime"].min()
        last_dt = cluster_df["acq_datetime"].max()
        duration_sec = (last_dt - first_dt).total_seconds()
        duration_days = round(duration_sec / 86400.0, 2)

        # Centroid coordinates
        c_lat = round(float(cluster_df["latitude"].mean()), 5)
        c_lon = round(float(cluster_df["longitude"].mean()), 5)

        # FRP features
        mean_frp: Optional[float] = None
        max_frp: Optional[float] = None
        if "frp" in cluster_df.columns:
            frp_series = pd.to_numeric(cluster_df["frp"], errors="coerce").dropna()
            if not frp_series.empty:
                mean_frp = round(float(frp_series.mean()), 2)
                max_frp = round(float(frp_series.max()), 2)

        # Confidence features
        mean_conf: Optional[float] = None
        if "confidence" in cluster_df.columns:
            parsed_confs = [
                parse_confidence_score(c)
                for c in cluster_df["confidence"]
                if parse_confidence_score(c) is not None
            ]
            if parsed_confs:
                mean_conf = round(float(np.mean(parsed_confs)), 1)

        persistence = calculate_persistence_score(
            active_days=active_days,
            detection_count=det_count,
        )

        feat = EventFeatures(
            event_id=event_id,
            centroid_lat=c_lat,
            centroid_lon=c_lon,
            detection_count=det_count,
            active_days=active_days,
            first_detected=str(first_dt),
            last_detected=str(last_dt),
            duration_days=duration_days,
            mean_frp=mean_frp,
            max_frp=max_frp,
            mean_firms_confidence=mean_conf,
            persistence_score=persistence,
        )
        events_list.append(feat.to_dict())

    events_df = pd.DataFrame(events_list, columns=EVENT_COLUMNS)
    return events_df, cleaned


def run_temporal_analysis(
    input_csv_path: Path,
    output_events_path: Path = DEFAULT_EVENTS_OUTPUT_PATH,
    output_grouped_path: Path = DEFAULT_GROUPED_OUTPUT_PATH,
    spatial_distance_km: float = DEFAULT_SPATIAL_THRESHOLD_KM,
    max_time_gap_hours: float = DEFAULT_MAX_TIME_GAP_HOURS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute temporal analysis on a FIRMS CSV input and save results to data/processed/."""
    if not input_csv_path.exists():
        raise FileNotFoundError(f"Input detections CSV does not exist: {input_csv_path}")

    raw_df = pd.read_csv(input_csv_path)
    events_df, grouped_df = group_thermal_detections(
        raw_df,
        spatial_distance_km=spatial_distance_km,
        max_time_gap_hours=max_time_gap_hours,
    )

    # Ensure output directory exists
    output_events_path.parent.mkdir(parents=True, exist_ok=True)
    output_grouped_path.parent.mkdir(parents=True, exist_ok=True)

    events_df.to_csv(output_events_path, index=False)
    grouped_df.to_csv(output_grouped_path, index=False)

    return events_df, grouped_df
