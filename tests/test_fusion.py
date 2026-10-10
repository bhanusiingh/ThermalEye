import pytest
import pandas as pd
from thermaleye.fusion.scoring import (
    calculate_firms_score,
    calculate_temporal_score,
    calculate_osm_score,
    calculate_cv_score,
    score_event,
    fuse_evidence
)

def test_calculate_firms_score():
    row = pd.Series({"mean_firms_confidence": 80.0, "mean_frp": 10.0})
    score = calculate_firms_score(row)
    # (0.8 + 0.5) / 2 = 0.65
    assert score == pytest.approx(0.65)
    
    # Missing values
    row_missing = pd.Series({"mean_firms_confidence": None, "mean_frp": None})
    assert calculate_firms_score(row_missing) is None

def test_calculate_temporal_score():
    row = pd.Series({"persistence_score": 0.4})
    assert calculate_temporal_score(row) == pytest.approx(0.4)
    
    row_high = pd.Series({"persistence_score": 1.5}) # test capping
    assert calculate_temporal_score(row_high) == 1.0

def test_calculate_osm_score():
    # Strong evidence
    row_ind = pd.Series({"osm_data_available": True, "industrial_features_1km": 1})
    assert calculate_osm_score(row_ind) == 1.0
    
    # Medium evidence
    row_med = pd.Series({"osm_data_available": True, "industrial_features_1km": 0, "nearest_industrial_feature_km": 2.5})
    assert calculate_osm_score(row_med) == 0.5
    
    # No industrial
    row_none = pd.Series({"osm_data_available": True, "industrial_features_1km": 0, "nearest_industrial_feature_km": 10.0})
    assert calculate_osm_score(row_none) == 0.0
    
    # Missing query
    row_missing = pd.Series({"osm_data_available": False})
    assert calculate_osm_score(row_missing) is None

def test_calculate_cv_score():
    row = pd.Series({"event_id": "EVT-1"})
    cv_preds = {"EVT-1": {"Industrial": 0.8, "Forest": 0.2}}
    assert calculate_cv_score(row, cv_preds) == 1.0
    
    cv_preds_med = {"EVT-1": {"Industrial": 0.3}}
    assert calculate_cv_score(row, cv_preds_med) == 0.5
    
    cv_preds_low = {"EVT-1": {"Industrial": 0.1}}
    assert calculate_cv_score(row, cv_preds_low) == 0.0
    
    assert calculate_cv_score(row, {}) is None

def test_score_event():
    row = pd.Series({
        "event_id": "EVT-1",
        "mean_firms_confidence": 100.0,
        "mean_frp": 20.0,
        "persistence_score": 1.0,
        "osm_data_available": True,
        "industrial_features_1km": 1
    })
    cv_preds = {"EVT-1": {"Industrial": 0.9}}
    
    result = score_event(row, cv_preds)
    assert result["priority"] == "High"
    assert result["score_cv"] == 1.0
    assert result["score_firms"] == 1.0
    assert result["score_temporal"] == 1.0
    assert result["score_osm"] == 1.0
    assert result["priority_score"] == 1.0
    assert result["evidence_coverage"] == "firms|temporal|osm|cv"

def test_insufficient_evidence():
    row = pd.Series({
        "event_id": "EVT-INSUF",
        "mean_firms_confidence": 100.0,
        "mean_frp": 100.0,
        "persistence_score": 1.0
    })
    # High FIRMS and temporal, but NO contextual evidence (no OSM, no CV)
    result = score_event(row, {})
    assert result["priority"] == "Insufficient evidence"
    assert result["priority_score"] == 1.0 # The math still calculates 1.0 based on available
    assert result["evidence_coverage"] == "firms|temporal"

def test_osm_zero_does_not_veto_high_priority():
    # If FIRMS is perfect, Temporal is perfect, CV is perfect, but OSM is empty (0.0),
    # the total score should be (0.2*1 + 0.3*1 + 0.25*0 + 0.25*1) = 0.75 -> "High"
    row = pd.Series({
        "event_id": "EVT-OSM-ZERO",
        "mean_firms_confidence": 100.0,
        "mean_frp": 20.0,
        "persistence_score": 1.0,
        "osm_data_available": True,
        "industrial_features_1km": 0,
        "nearest_industrial_feature_km": 10.0 # Will score 0.0 for OSM
    })
    cv_preds = {"EVT-OSM-ZERO": {"Industrial": 0.9}} # Will score 1.0 for CV
    
    result = score_event(row, cv_preds)
    assert result["score_osm"] == 0.0
    assert result["priority"] == "High"
    assert result["priority_score"] == pytest.approx(0.75)
    
def test_all_coverage_combinations():
    # Only OSM context
    row_osm = pd.Series({
        "event_id": "EVT-OSM-ONLY",
        "mean_firms_confidence": 100.0,
        "mean_frp": 20.0,
        "persistence_score": 1.0,
        "osm_data_available": True,
        "industrial_features_1km": 1
    })
    res_osm = score_event(row_osm, {})
    assert res_osm["priority"] in ["Low", "Medium", "High"]
    
    # Only CV context
    row_cv = pd.Series({
        "event_id": "EVT-CV-ONLY",
        "mean_firms_confidence": 100.0,
        "mean_frp": 20.0,
        "persistence_score": 1.0,
    })
    res_cv = score_event(row_cv, {"EVT-CV-ONLY": {"Industrial": 0.9}})
    assert res_cv["priority"] in ["Low", "Medium", "High"]

def test_fuse_evidence_empty():
    res = fuse_evidence(pd.DataFrame())
    assert res.empty

def test_fuse_evidence_boundary_filter_mock():
    from unittest.mock import patch
    events_df = pd.DataFrame({
        "event_id": ["EVT-1"],
        "centroid_lat": [20.0],
        "centroid_lon": [80.0],
        "mean_firms_confidence": [50.0],
        "mean_frp": [10.0]
    })
    
    # Using the mock boundary filter
    with patch("thermaleye.geo.boundary.filter_points_by_country", return_value=events_df):
        res = fuse_evidence(events_df)
        assert len(res) == 1
        assert "priority" in res.columns
        assert res.iloc[0]["priority"] in ["Low", "Medium", "High", "Insufficient evidence"]
