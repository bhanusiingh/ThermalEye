"""Unit tests for dashboard data loading and filtering logic."""
import pandas as pd
from pathlib import Path
from thermaleye.dashboard.data_loader import load_events_data, filter_events

def test_load_events_data_empty():
    """Test loading with a non-existent file gracefully fails to an empty DataFrame."""
    df = load_events_data(Path("non_existent_file.csv"))
    assert df.empty

def test_filter_events():
    """Test dashboard filtering logic."""
    df = pd.DataFrame({
        "event_id": ["TE-EVT-01", "TE-EVT-02", "TE-EVT-03"],
        "priority": ["High", "Medium", "Insufficient evidence"],
        "evidence_coverage": ["firms|temporal|cv", "firms|temporal|osm", "firms|temporal"]
    })
    
    # Empty search
    res = filter_events(df)
    assert len(res) == 3
    
    # Search ID
    res = filter_events(df, search_id="02")
    assert len(res) == 1
    assert res.iloc[0]["event_id"] == "TE-EVT-02"
    
    # Priority multiselect
    res = filter_events(df, priorities=["High", "Medium"])
    assert len(res) == 2
    
    # Evidence coverage requirement (must contain 'osm')
    res = filter_events(df, coverage_required=["osm"])
    assert len(res) == 1
    assert res.iloc[0]["priority"] == "Medium"
    
    # Evidence coverage requirement (must contain 'firms')
    res = filter_events(df, coverage_required=["firms"])
    assert len(res) == 3

def test_map_color_generation():
    """Test color generation maps to correct lengths."""
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test.csv"
        df = pd.DataFrame({
            "event_id": ["1", "2", "3", "4"],
            "priority": ["High", "Medium", "Low", "Insufficient evidence"],
            "evidence_coverage": ["a", "b", "c", None]
        })
        df.to_csv(csv_path, index=False)
        
        loaded = load_events_data(csv_path)
        assert len(loaded) == 4
        assert "map_color" in loaded.columns
        assert len(loaded.iloc[0]["map_color"]) == 4  # RGBA list
        assert loaded.iloc[3]["evidence_coverage"] == "none" # NaN replaced with "none"

def test_format_val():
    """Test UI formatting of missing or numeric values."""
    from src.thermaleye.dashboard.app import format_val
    import numpy as np
    
    assert format_val(np.nan) == "Unavailable"
    assert format_val(None) == "Unavailable"
    assert format_val(1.23456) == 1.235
    assert format_val(0.0) == 0.0
    assert format_val("Valid") == "Valid"
