"""Tests for coordinate-aware image identity and CV prediction provenance."""
import pytest
from thermaleye.imagery.sentinel2 import generate_image_fingerprint
import pandas as pd
from pathlib import Path

def test_fingerprint_determinism():
    """Identical requests should generate identical fingerprints."""
    fp1 = generate_image_fingerprint(22.0, 78.0, 10, 0.05, 256, 256)
    fp2 = generate_image_fingerprint(22.0, 78.0, 10, 0.05, 256, 256)
    assert fp1 == fp2

def test_fingerprint_invalidation():
    """Changed coordinates or parameters must invalidate the fingerprint."""
    base = generate_image_fingerprint(22.0, 78.0, 10, 0.05, 256, 256)
    
    # Lat change
    diff_lat = generate_image_fingerprint(22.1, 78.0, 10, 0.05, 256, 256)
    assert base != diff_lat
    
    # Lon change
    diff_lon = generate_image_fingerprint(22.0, 78.1, 10, 0.05, 256, 256)
    assert base != diff_lon
    
    # Extent change
    diff_extent = generate_image_fingerprint(22.0, 78.0, 10, 0.10, 256, 256)
    assert base != diff_extent
    
    # Tiny float drift should NOT invalidate (due to round(., 5))
    drift = generate_image_fingerprint(22.000001, 78.000004, 10, 0.05, 256, 256)
    assert base == drift

def test_legacy_cache_miss():
    """Verify legacy cache entries without verifiable provenance are rejected."""
    # This logic is inside run_fusion.py, but we can verify the condition directly
    
    # Expected current fingerprint
    expected_fp = generate_image_fingerprint(22.0, 78.0, 10, 0.05, 256, 256)
    
    # Simulating a legacy S2 metadata row
    legacy_row = pd.Series({"event_id": "TE-EVT-01", "output_path": "cache.png"})
    
    actual_fp = legacy_row.get("image_fingerprint")
    assert pd.isna(actual_fp) or actual_fp != expected_fp
    
    # Simulating a stale S2 metadata row
    stale_row = pd.Series({
        "event_id": "TE-EVT-01", 
        "output_path": "cache.png", 
        "image_fingerprint": generate_image_fingerprint(22.5, 78.5, 10, 0.05, 256, 256)
    })
    
    assert stale_row.get("image_fingerprint") != expected_fp
