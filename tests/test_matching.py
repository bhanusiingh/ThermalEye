"""Tests for the spatial-temporal evaluation matching module."""
import pytest
import pandas as pd
from datetime import datetime, timezone
from thermaleye.evaluation.matching import haversine_distance_km, match_incidents

def test_haversine_distance():
    # Known distance: 1 degree latitude is ~111 km
    dist = haversine_distance_km(0.0, 0.0, 1.0, 0.0)
    assert 110.0 <= dist <= 112.0

def test_match_incidents():
    # Golden dataset (Curated incidents)
    golden_df = pd.DataFrame({
        'sample_id': ['G1', 'G2', 'G3'],
        'latitude': [10.0, 20.0, 30.0],
        'longitude': [10.0, 20.0, 30.0],
        'event_date': ['2024-01-01T12:00:00Z', '2024-01-01T12:00:00Z', '2024-01-01T12:00:00Z'],
        'ground_truth_class': ['confirmed_industrial_fire', 'vegetation_fire', 'confirmed_industrial_fire']
    })
    
    # Pipeline events
    events_df = pd.DataFrame({
        'event_id': ['E1', 'E2', 'E3_NO_MATCH'],
        'centroid_lat': [10.001, 20.0, 40.0],  # E1 is very close to G1. E2 is exactly G2. E3 is far away.
        'centroid_lon': [10.001, 20.0, 40.0],
        'first_detected': ['2024-01-01T13:00:00Z', '2024-01-01T12:00:00Z', '2024-01-01T12:00:00Z'],
        'last_detected': ['2024-01-01T14:00:00Z', '2024-01-01T12:00:00Z', '2024-01-01T12:00:00Z']
    })
    
    matched = match_incidents(golden_df, events_df, spatial_tol_km=5.0, time_tol_hours=24.0)
    
    # G1 should match E1 (distance > 0 but < 5km)
    m1 = matched[matched['sample_id'] == 'G1'].iloc[0]
    assert m1['match_status'] == 'Matched'
    assert m1['matched_event_id'] == 'E1'
    assert m1['match_distance_km'] > 0
    assert m1['match_time_diff_hours'] == 1.0 # 13:00 - 12:00
    
    # G2 should match E2 (distance == 0)
    m2 = matched[matched['sample_id'] == 'G2'].iloc[0]
    assert m2['match_status'] == 'Matched'
    assert m2['matched_event_id'] == 'E2'
    assert m2['match_distance_km'] == 0.0
    
    # G3 has no matches. Now correctly labeled Unmatched.
    m3 = matched[matched['sample_id'] == 'G3'].iloc[0]
    assert m3['match_status'] == 'Unmatched'
    assert pd.isna(m3['matched_event_id'])

def test_multiple_candidates_ambiguous():
    golden_df = pd.DataFrame({
        'sample_id': ['G1'],
        'latitude': [10.0],
        'longitude': [10.0],
        'event_date': ['2024-01-01T12:00:00Z']
    })
    
    events_df = pd.DataFrame({
        'event_id': ['E1', 'E2'],
        'centroid_lat': [10.001, 10.002],
        'centroid_lon': [10.001, 10.002],
        'first_detected': ['2024-01-01T12:00:00Z', '2024-01-01T12:00:00Z'],
        'last_detected': ['2024-01-01T12:00:00Z', '2024-01-01T12:00:00Z']
    })
    
    matched = match_incidents(golden_df, events_df, spatial_tol_km=5.0)
    
    m1 = matched.iloc[0]
    assert m1['match_status'] == 'Ambiguous'
    assert pd.isna(m1['matched_event_id'])

def test_competing_incidents_conflict():
    golden_df = pd.DataFrame({
        'sample_id': ['G1', 'G2'],
        'latitude': [10.0, 10.0], # Same location
        'longitude': [10.0, 10.0],
        'event_date': ['2024-01-01T12:00:00Z', '2024-01-01T12:00:00Z']
    })
    
    events_df = pd.DataFrame({
        'event_id': ['E1'],
        'centroid_lat': [10.0],
        'centroid_lon': [10.0],
        'first_detected': ['2024-01-01T12:00:00Z'],
        'last_detected': ['2024-01-01T12:00:00Z']
    })
    
    matched = match_incidents(golden_df, events_df, spatial_tol_km=5.0)
    
    # Both try to claim E1, they should both be marked Conflict
    for _, row in matched.iterrows():
        assert row['match_status'] == 'Conflict'
        assert pd.isna(row['matched_event_id'])
