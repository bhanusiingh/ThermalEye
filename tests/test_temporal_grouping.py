"""Unit tests for Phase 2: Temporal Event Grouping and Persistence Scoring."""

from datetime import datetime
import pandas as pd
import pytest

from thermaleye.temporal.grouping import (
    haversine_distance_km,
    parse_confidence_score,
    clean_detections,
    group_thermal_detections,
    EVENT_COLUMNS,
)
from thermaleye.temporal.persistence import calculate_persistence_score


from unittest.mock import patch

# Mock boundary filtering so temporal tests don't require geopandas
patcher = patch("thermaleye.geo.boundary.filter_points_by_country", side_effect=lambda df, **kwargs: df)
patcher.start()

class TestGeospatialDistance:
    """Test Haversine distance calculations."""

    def test_identical_points(self) -> None:
        assert haversine_distance_km(28.6139, 77.2090, 28.6139, 77.2090) == 0.0

    def test_known_distance(self) -> None:
        # Distance between New Delhi (28.6139, 77.2090) and Agra (27.1767, 78.0081) is ~180 km
        dist = haversine_distance_km(28.6139, 77.2090, 27.1767, 78.0081)
        assert 175.0 < dist < 190.0

    def test_small_subkilometer_distance(self) -> None:
        # ~0.005 deg latitude difference is ~0.55 km
        dist = haversine_distance_km(20.0, 80.0, 20.005, 80.0)
        assert 0.5 < dist < 0.6


class TestConfidenceParsing:
    """Test FIRMS confidence values handling."""

    def test_categorical_strings(self) -> None:
        assert parse_confidence_score("l") == 30.0
        assert parse_confidence_score("low") == 30.0
        assert parse_confidence_score("n") == 60.0
        assert parse_confidence_score("nominal") == 60.0
        assert parse_confidence_score("h") == 90.0
        assert parse_confidence_score("high") == 90.0

    def test_numeric_confidence(self) -> None:
        assert parse_confidence_score(85) == 85.0
        assert parse_confidence_score("75.5") == 75.5

    def test_invalid_and_nan(self) -> None:
        assert parse_confidence_score(None) is None
        assert parse_confidence_score(float("nan")) is None
        assert parse_confidence_score("unknown") is None


class TestPersistenceScore:
    """Test transparent persistence score calculation and bounds."""

    def test_zero_or_negative_inputs(self) -> None:
        assert calculate_persistence_score(0, 0) == 0.0
        assert calculate_persistence_score(-1, 5) == 0.0

    def test_single_day_single_detection(self) -> None:
        # No multi-day recurrence and single pass -> 0.0
        assert calculate_persistence_score(active_days=1, detection_count=1) == 0.0

    def test_single_day_multiple_detections(self) -> None:
        # Intra-day repeat detection (e.g. 4 passes) -> intra-day frequency only
        score = calculate_persistence_score(active_days=1, detection_count=4)
        assert 0.0 < score <= 0.15

    def test_multi_day_high_persistence(self) -> None:
        # High multi-day and high detection count -> close to 1.0
        score = calculate_persistence_score(active_days=5, detection_count=10)
        assert score == 1.0

    def test_intermediate_persistence(self) -> None:
        score = calculate_persistence_score(active_days=3, detection_count=5)
        # Expected: 0.7 * (2/4) + 0.3 * (4/9) = 0.35 + 0.133 = 0.483
        assert 0.45 < score < 0.52


class TestInputCleaningAndValidation:
    """Test input validation, deduplication, and invalid coordinate handling."""

    def test_missing_required_column_raises(self) -> None:
        bad_df = pd.DataFrame({"latitude": [20.0], "longitude": [80.0]})
        with pytest.raises(ValueError, match="missing required column: 'acq_date'"):
            clean_detections(bad_df)

    def test_filter_invalid_coordinates(self) -> None:
        df = pd.DataFrame({
            "latitude": [20.0, 999.0, -95.0, float("nan"), 25.0],
            "longitude": [80.0, 80.0, 80.0, 80.0, float("nan")],
            "acq_date": ["2026-10-09"] * 5,
            "acq_time": [730] * 5,
        })
        cleaned = clean_detections(df)
        assert len(cleaned) == 1
        assert cleaned.iloc[0]["latitude"] == 20.0

    def test_deduplicates_exact_records(self) -> None:
        df = pd.DataFrame({
            "latitude": [20.0, 20.0],
            "longitude": [80.0, 80.0],
            "acq_date": ["2026-10-09", "2026-10-09"],
            "acq_time": [730, 730],
            "satellite": ["20", "20"],
        })
        cleaned = clean_detections(df)
        assert len(cleaned) == 1


class TestSyntheticGroupingScenarios:
    """Test synthetic temporal grouping fixtures."""

    def test_single_day_event(self) -> None:
        """Single-day event with two consecutive passes (NOAA-20 and NOAA-21)."""
        df = pd.DataFrame({
            "latitude": [22.5000, 22.5010],
            "longitude": [85.2000, 85.2010],
            "acq_date": ["2026-10-09", "2026-10-09"],
            "acq_time": [730, 815],
            "satellite": ["20", "21"],
            "frp": [10.5, 12.0],
            "confidence": ["nominal", "high"],
        })
        events, grouped = group_thermal_detections(df, spatial_distance_km=2.5, max_time_gap_hours=36.0)

        assert len(events) == 1
        assert events.iloc[0]["detection_count"] == 2
        assert events.iloc[0]["active_days"] == 1
        assert events.iloc[0]["mean_frp"] == 11.25
        assert events.iloc[0]["max_frp"] == 12.0
        assert events.iloc[0]["mean_firms_confidence"] == 75.0  # mean of 60 and 90
        assert events.iloc[0]["persistence_score"] < 0.1
        assert len(grouped["event_id"].unique()) == 1

    def test_multi_day_persistent_event(self) -> None:
        """Event detected over 3 consecutive days at an industrial site."""
        df = pd.DataFrame({
            "latitude": [21.1500, 21.1510, 21.1505],
            "longitude": [79.0800, 79.0810, 79.0805],
            "acq_date": ["2026-10-01", "2026-10-02", "2026-10-03"],
            "acq_time": [730, 800, 745],
            "satellite": ["20", "21", "20"],
            "frp": [25.0, 30.0, 22.0],
            "confidence": ["high", "high", "nominal"],
        })
        events, grouped = group_thermal_detections(df, spatial_distance_km=2.5, max_time_gap_hours=36.0)

        assert len(events) == 1
        assert events.iloc[0]["detection_count"] == 3
        assert events.iloc[0]["active_days"] == 3
        assert events.iloc[0]["duration_days"] >= 1.9
        assert events.iloc[0]["persistence_score"] > 0.35

    def test_nearby_spatially_but_separated_in_time(self) -> None:
        """Detections at exact same coordinates but separated by 5 days (>36h gap)."""
        df = pd.DataFrame({
            "latitude": [23.0000, 23.0000],
            "longitude": [80.0000, 80.0000],
            "acq_date": ["2026-10-01", "2026-10-06"],
            "acq_time": [730, 730],
        })
        events, grouped = group_thermal_detections(df, spatial_distance_km=2.5, max_time_gap_hours=36.0)

        # Must NOT be merged into a single event
        assert len(events) == 2
        assert len(grouped["event_id"].unique()) == 2

    def test_simultaneous_but_spatially_distant_detections(self) -> None:
        """Two detections at the exact same timestamp but 50 km apart."""
        df = pd.DataFrame({
            "latitude": [20.0000, 20.4500],  # ~50 km apart in latitude
            "longitude": [80.0000, 80.0000],
            "acq_date": ["2026-10-09", "2026-10-09"],
            "acq_time": [730, 730],
        })
        events, grouped = group_thermal_detections(df, spatial_distance_km=2.5, max_time_gap_hours=36.0)

        assert len(events) == 2
        assert len(grouped["event_id"].unique()) == 2

    def test_empty_dataframe_handling(self) -> None:
        """Empty input DataFrame returns empty tables with correct column schema."""
        empty_df = pd.DataFrame()
        events, grouped = group_thermal_detections(empty_df)

        assert events.empty
        assert list(events.columns) == EVENT_COLUMNS
        assert grouped.empty

    def test_all_invalid_rows(self) -> None:
        """DataFrame with only invalid coordinates returns empty tables safely."""
        bad_df = pd.DataFrame({
            "latitude": [-999.0, 999.0],
            "longitude": [0.0, 0.0],
            "acq_date": ["2026-10-09", "2026-10-09"],
        })
        events, grouped = group_thermal_detections(bad_df)
        assert events.empty
        assert list(events.columns) == EVENT_COLUMNS

    def test_stable_event_identifiers(self) -> None:
        """Shuffled input rows yield identical event IDs and assignments."""
        df = pd.DataFrame({
            "latitude": [20.0000, 20.0010, 25.0000, 25.0010],
            "longitude": [80.0000, 80.0010, 85.0000, 85.0010],
            "acq_date": ["2026-10-09"] * 4,
            "acq_time": [730, 735, 730, 735],
        })
        events1, grouped1 = group_thermal_detections(df)
        shuffled_df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        events2, grouped2 = group_thermal_detections(shuffled_df)

        assert len(events1) == 2
        assert len(events2) == 2
        assert list(events1["event_id"]) == list(events2["event_id"])
        assert set(events1["centroid_lat"]) == set(events2["centroid_lat"])
