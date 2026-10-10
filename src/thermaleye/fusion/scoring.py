"""Evidence fusion and priority scoring for thermal events."""
import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Configurable Weights
WEIGHTS = {
    "firms": 0.20,
    "temporal": 0.30,
    "osm": 0.25,
    "cv": 0.25
}

def calculate_firms_score(row: pd.Series) -> Optional[float]:
    """Calculate the FIRMS evidence score."""
    conf = row.get("mean_firms_confidence")
    frp = row.get("mean_frp")
    
    if pd.isna(conf) and pd.isna(frp):
        return None
        
    conf_score = (conf / 100.0) if pd.notna(conf) else 0.5
    # Cap FRP to prevent massive forest fires from dominating, industrial are often steady but smaller
    frp_score = min(frp / 20.0, 1.0) if pd.notna(frp) else 0.5
    
    return (conf_score + frp_score) / 2.0

def calculate_temporal_score(row: pd.Series) -> Optional[float]:
    """Calculate temporal persistence score."""
    persistence = row.get("persistence_score")
    if pd.isna(persistence):
        return None
    # Cap at 1.0
    return min(float(persistence), 1.0)

def calculate_osm_score(row: pd.Series) -> Optional[float]:
    """Calculate OSM contextual score."""
    if "osm_data_available" not in row or pd.isna(row["osm_data_available"]):
        return None
        
    # Explicitly distinguish missing/failed queries from empty results
    if not bool(row["osm_data_available"]):
        return None
        
    ind_1km = row.get("industrial_features_1km", 0)
    ind_landuse = row.get("industrial_landuse_present", False)
    power_plant = row.get("nearest_power_plant_km", np.nan)
    ind_dist = row.get("nearest_industrial_feature_km", np.nan)
    
    # High contextual evidence
    if ind_1km > 0 or bool(ind_landuse) or (pd.notna(power_plant) and power_plant <= 1.0):
        return 1.0
    # Medium contextual evidence
    if pd.notna(ind_dist) and ind_dist <= 3.0:
        return 0.5
        
    # Successful query, but no industrial infrastructure nearby
    return 0.0

def calculate_cv_score(row: pd.Series, cv_predictions: Dict[str, Dict[str, float]]) -> Optional[float]:
    """Calculate Satellite CV score based on predicted land cover."""
    event_id = row.get("event_id")
    if event_id not in cv_predictions:
        return None
        
    preds = cv_predictions[event_id]
    industrial_prob = preds.get("Industrial", 0.0)
    
    # Note: These are uncalibrated land-cover similarities, not probability of an active fire.
    if industrial_prob > 0.5:
        return 1.0
    elif industrial_prob > 0.2:
        return 0.5
    return 0.0

def determine_priority(final_score: float, coverage: list) -> str:
    """Map the numeric priority score to a human review category."""
    # Prevent misleading rankings: require at least one contextual source (OSM or CV).
    # FIRMS + Temporal alone cannot prioritize an industrial fire event reliably.
    if "osm" not in coverage and "cv" not in coverage:
        return "Insufficient evidence"
        
    if final_score >= 0.7:
        return "High"
    elif final_score >= 0.4:
        return "Medium"
    return "Low"

def score_event(row: pd.Series, cv_predictions: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """Process a single event row and return its evidence scores and final priority."""
    s_firms = calculate_firms_score(row)
    s_temp = calculate_temporal_score(row)
    s_osm = calculate_osm_score(row)
    s_cv = calculate_cv_score(row, cv_predictions)
    
    components = {
        "score_firms": s_firms,
        "score_temporal": s_temp,
        "score_osm": s_osm,
        "score_cv": s_cv
    }
    
    # Compute weighted average using only available evidence
    total_weight = 0.0
    weighted_sum = 0.0
    coverage = []
    
    if s_firms is not None:
        weighted_sum += s_firms * WEIGHTS["firms"]
        total_weight += WEIGHTS["firms"]
        coverage.append("firms")
    if s_temp is not None:
        weighted_sum += s_temp * WEIGHTS["temporal"]
        total_weight += WEIGHTS["temporal"]
        coverage.append("temporal")
    if s_osm is not None:
        weighted_sum += s_osm * WEIGHTS["osm"]
        total_weight += WEIGHTS["osm"]
        coverage.append("osm")
    if s_cv is not None:
        weighted_sum += s_cv * WEIGHTS["cv"]
        total_weight += WEIGHTS["cv"]
        coverage.append("cv")
        
    final_score = weighted_sum / total_weight if total_weight > 0 else 0.0
    priority = determine_priority(final_score, coverage)
    
    result = {
        "event_id": row.get("event_id"),
        "priority": priority,
        "priority_score": round(final_score, 3),
        "available_evidence_weight": round(total_weight, 2),
        "evidence_coverage": "|".join(coverage) if coverage else "none",
    }
    
    # Format components for output
    for k, v in components.items():
        result[k] = round(v, 3) if v is not None else None
        
    return result

def fuse_evidence(
    events_df: pd.DataFrame, 
    osm_df: Optional[pd.DataFrame] = None,
    cv_predictions: Optional[Dict[str, Dict[str, float]]] = None
) -> pd.DataFrame:
    """Fuse all available evidence and return prioritized events."""
    if events_df.empty:
        return pd.DataFrame()
        
    # Enforce boundary check before scoring
    from thermaleye.geo.boundary import filter_points_by_country
    try:
        # Fail closed if geopandas is missing or boundary logic crashes
        validated_df = filter_points_by_country(events_df.copy(), lat_col="centroid_lat", lon_col="centroid_lon", country_name="India")
    except Exception as e:
        logger.error(f"Geographic boundary validation failed: {e}")
        raise RuntimeError("Boundary validation is mandatory before fusion.") from e
        
    if validated_df.empty:
        return pd.DataFrame()
        
    # Left join OSM data if provided
    if osm_df is not None and not osm_df.empty:
        # Only take OSM specific columns to avoid conflicts
        osm_cols = ['event_id', 'nearest_industrial_feature_km', 'nearest_power_plant_km', 
                   'industrial_features_500m', 'industrial_features_1km', 
                   'industrial_landuse_present', 'osm_data_available']
        osm_subset = osm_df[[c for c in osm_cols if c in osm_df.columns]]
        validated_df = validated_df.merge(osm_subset, on="event_id", how="left")
        
    if cv_predictions is None:
        cv_predictions = {}
        
    scored_records = []
    for _, row in validated_df.iterrows():
        score_dict = score_event(row, cv_predictions)
        # Merge the score_dict back with original row
        merged = {**row.to_dict(), **score_dict}
        scored_records.append(merged)
        
    # Reorder columns to put priority first
    result_df = pd.DataFrame(scored_records)
    priority_cols = ["event_id", "priority", "priority_score", "evidence_coverage", "available_evidence_weight", 
                     "score_firms", "score_temporal", "score_osm", "score_cv"]
    other_cols = [c for c in result_df.columns if c not in priority_cols]
    return result_df[priority_cols + other_cols]
