"""Unit tests for Phase 3: OpenStreetMap Industrial Context Module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
import requests

from thermaleye.osm.overpass import (
    OverpassClient,
    DiskCache,
    build_overpass_query,
    extract_features_from_elements,
    enrich_events_with_osm,
    OSM_FEATURE_COLUMNS,
)


class TestQueryBuilder:
    """Test Overpass QL query formatting."""

    def test_query_contains_required_tags_and_coordinates(self) -> None:
        query = build_overpass_query(lat=28.6139, lon=77.2090, radius_m=3000, timeout=25)
        assert "[out:json][timeout:25];" in query
        assert "around:3000,28.6139,77.209" in query
        assert '"landuse"="industrial"' in query
        assert '"power"="plant"' in query
        assert '"power"="generator"' in query
        assert '"industrial"' in query
        assert '"man_made"="works"' in query
        assert '"man_made"="chimney"' in query
        assert "out center tags;" in query


class TestDiskCache:
    """Test cache operations."""

    def test_cache_set_get_and_persistence(self, tmp_path: Path) -> None:
        cache_file = tmp_path / "test_cache.json"
        cache = DiskCache(cache_file)
        assert cache.get("nonexistent") is None

        cache.set("key1", [{"id": 1, "type": "node"}])
        assert cache.get("key1") == [{"id": 1, "type": "node"}]
        cache.save()

        assert cache_file.exists()
        # Verify persistence across instances
        cache_reloaded = DiskCache(cache_file)
        assert cache_reloaded.get("key1") == [{"id": 1, "type": "node"}]


class TestFeatureExtraction:
    """Test geospatial extraction and tag classification from OSM elements."""

    def test_tagged_industrial_and_power_features(self) -> None:
        centroid_lat = 20.0000
        centroid_lon = 80.0000

        # Element 1: Factory node at ~300m (~0.0027 deg lat)
        elem_factory = {
            "type": "node",
            "id": 101,
            "lat": 20.0027,
            "lon": 80.0000,
            "tags": {"man_made": "works", "name": "Steel Rolling Mill"},
        }

        # Element 2: Power plant way with center at ~800m (~0.0072 deg lat)
        elem_power = {
            "type": "way",
            "id": 202,
            "center": {"lat": 20.0072, "lon": 80.0000},
            "tags": {"power": "plant", "plant:source": "coal"},
        }

        # Element 3: Industrial landuse polygon at ~1500m (~0.0135 deg lat)
        elem_landuse = {
            "type": "way",
            "id": 303,
            "center": {"lat": 20.0135, "lon": 80.0000},
            "tags": {"landuse": "industrial"},
        }

        elements = [elem_factory, elem_power, elem_landuse]
        features = extract_features_from_elements(centroid_lat, centroid_lon, elements)

        assert features["osm_data_available"] is True
        assert features["nearest_industrial_feature_km"] is not None
        assert 0.25 < features["nearest_industrial_feature_km"] < 0.35  # ~0.3 km
        assert features["nearest_power_plant_km"] is not None
        assert 0.75 < features["nearest_power_plant_km"] < 0.85  # ~0.8 km
        assert features["industrial_features_500m"] == 1  # only factory within 500m
        assert features["industrial_features_1km"] == 2   # factory and power plant within 1km
        assert features["industrial_landuse_present"] is True

    def test_duplicate_features_not_double_counted(self) -> None:
        centroid_lat = 20.0000
        centroid_lon = 80.0000

        # Identical feature appearing twice in results
        duplicate_elem = {
            "type": "node",
            "id": 555,
            "lat": 20.0020,
            "lon": 80.0000,
            "tags": {"industrial": "factory"},
        }
        elements = [duplicate_elem, duplicate_elem]
        features = extract_features_from_elements(centroid_lat, centroid_lon, elements)

        assert features["industrial_features_500m"] == 1
        assert features["industrial_features_1km"] == 1

    def test_confirmed_absence_empty_elements(self) -> None:
        centroid_lat = 20.0000
        centroid_lon = 80.0000

        # Query succeeded but returned no elements
        features = extract_features_from_elements(centroid_lat, centroid_lon, [])

        assert features["osm_data_available"] is True
        assert features["nearest_industrial_feature_km"] is None
        assert features["nearest_power_plant_km"] is None
        assert features["industrial_features_500m"] == 0
        assert features["industrial_features_1km"] == 0
        assert features["industrial_landuse_present"] is False


class TestOverpassClient:
    """Test Overpass client API handling, caching, and errors."""

    @patch("requests.post")
    def test_successful_query_caches_result(self, mock_post: MagicMock, tmp_path: Path) -> None:
        cache_file = tmp_path / "client_cache.json"
        client = OverpassClient(cache_path=cache_file, request_delay=0.0)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "elements": [{"type": "node", "id": 1, "lat": 20.0, "lon": 80.0, "tags": {"industrial": "factory"}}]
        }
        mock_post.return_value = mock_resp

        success, elements = client.query_nearby_industrial(lat=20.0, lon=80.0, radius_m=3000)
        assert success is True
        assert len(elements) == 1
        assert mock_post.call_count == 1

        # Second query for same location should use cache without HTTP request
        success2, elements2 = client.query_nearby_industrial(lat=20.0, lon=80.0, radius_m=3000)
        assert success2 is True
        assert len(elements2) == 1
        assert mock_post.call_count == 1

    @patch("requests.post")
    def test_rate_limit_429_returns_failure(self, mock_post: MagicMock, tmp_path: Path) -> None:
        cache_file = tmp_path / "rate_limit_cache.json"
        client = OverpassClient(cache_path=cache_file, request_delay=0.0)

        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_post.return_value = mock_resp

        success, elements = client.query_nearby_industrial(lat=20.0, lon=80.0, radius_m=3000)
        assert success is False
        assert elements == []

    @patch("requests.post")
    def test_timeout_504_returns_failure(self, mock_post: MagicMock, tmp_path: Path) -> None:
        cache_file = tmp_path / "timeout_cache.json"
        client = OverpassClient(cache_path=cache_file, request_delay=0.0)

        mock_resp = MagicMock()
        mock_resp.status_code = 504
        mock_post.return_value = mock_resp

        success, elements = client.query_nearby_industrial(lat=20.0, lon=80.0, radius_m=3000)
        assert success is False
        assert elements == []


class TestEnrichmentWorkflow:
    """Test complete DataFrame enrichment workflow."""

    def test_enrich_events_respects_max_events_limit(self, tmp_path: Path) -> None:
        events_df = pd.DataFrame({
            "event_id": ["TE-EVT-00001", "TE-EVT-00002", "TE-EVT-00003"],
            "centroid_lat": [20.0, 21.0, 22.0],
            "centroid_lon": [80.0, 81.0, 82.0],
            "detection_count": [2, 3, 1],
        })

        client = OverpassClient(cache_path=tmp_path / "workflow_cache.json", request_delay=0.0)
        # Mock client to return success for all queries
        client.query_nearby_industrial = MagicMock(return_value=(True, []))

        # Set max_events to 2 (only first 2 events processed, 3rd skipped)
        enriched = enrich_events_with_osm(events_df, client=client, max_events=2)

        for col in OSM_FEATURE_COLUMNS:
            assert col in enriched.columns

        # First 2 events should have osm_data_available = True
        assert bool(enriched.iloc[0]["osm_data_available"]) is True
        assert bool(enriched.iloc[1]["osm_data_available"]) is True

        # Third event should be skipped: osm_data_available = False, others None
        assert bool(enriched.iloc[2]["osm_data_available"]) is False
        assert enriched.iloc[2]["nearest_industrial_feature_km"] is None
        assert enriched.iloc[2]["industrial_features_500m"] is None

    def test_enrich_empty_dataframe(self) -> None:
        empty_df = pd.DataFrame()
        enriched = enrich_events_with_osm(empty_df)
        assert enriched.empty
        for col in OSM_FEATURE_COLUMNS:
            assert col in enriched.columns
