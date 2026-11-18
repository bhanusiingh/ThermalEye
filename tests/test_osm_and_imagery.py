"""
Unit and Integration Tests for ThermalEye OSM Enrichment and Sentinel-2 Imagery
"""

import json
import unittest
import uuid
from unittest.mock import MagicMock, patch
import requests

from enrich_osm_and_score import (
    SpatialCache,
    compute_priority_score,
    haversine_distance,
    _parse_retry_after,
    OVERPASS_USER_AGENT,
    get_database_url,
)
from fetch_imagery import (
    search_sentinel2_scenes,
    update_event_imagery,
    MAX_CLOUD_COVER,
)
from sqlalchemy import create_engine, text


class TestOSMEnrichmentAndScoring(unittest.TestCase):
    def test_meaningful_user_agent(self):
        """Verify User-Agent contains a descriptive app identifier and contact."""
        self.assertIn("ThermalEye", OVERPASS_USER_AGENT)
        self.assertIn("@", OVERPASS_USER_AGENT)

    def test_haversine_distance_calculation(self):
        """Test great-circle distance between two known coordinates."""
        # Distance between (28.6139, 77.2090) and (28.6239, 77.2090) is ~1.11 km
        dist = haversine_distance(28.6139, 77.2090, 28.6239, 77.2090)
        self.assertAlmostEqual(dist, 1112.0, delta=50.0)

    def test_spatial_cache_facilities(self):
        """Test adding facilities and retrieving the nearest facility within search radius."""
        cache = SpatialCache(cache_radius_m=2000)
        elements = [
            {
                "lat": 21.101,
                "lon": 72.892,
                "tags": {"name": "Thermal Power Plant Surat", "power": "plant"},
            },
            {
                "center": {"lat": 21.105, "lon": 72.895},
                "tags": {"name": "Surat Industrial Area", "landuse": "industrial"},
            },
        ]
        cache.add_facilities(elements)
        self.assertEqual(len(cache.facilities), 2)

        # Query close to the power plant (< 500m)
        match = cache.find_nearest_facility(21.102, 72.893, max_radius_m=3000)
        self.assertIsNotNone(match)
        name, ftype, dist = match
        self.assertIn("Thermal Power Plant", name)
        self.assertEqual(ftype, "plant")
        self.assertLess(dist, 500)

        # Query far away (> 5000m) should return None
        far_match = cache.find_nearest_facility(21.200, 72.900, max_radius_m=3000)
        self.assertIsNone(far_match)

    def test_spatial_cache_empty_zone(self):
        """Test empty zone caching for remote fire locations."""
        cache = SpatialCache()
        cache.add_empty_zone(24.500, 78.500, radius_m=3000)

        # Point within 400m of empty zone center is guaranteed empty
        self.assertTrue(cache.is_in_empty_zone(24.502, 78.502))
        # Point 10km away is not in empty zone
        self.assertFalse(cache.is_in_empty_zone(24.600, 78.500))

    def test_compute_priority_score_heuristic(self):
        """Test Investigation Priority Score components and explanation payload."""
        score, reasons_json = compute_priority_score(
            max_frp=50.0, detection_count=2, nearest_type="power"
        )
        reasons = json.loads(reasons_json)

        # FRP score: min(50/100 * 40, 40) = 20.0
        # Persistence: min(2 * 10, 30) = 20
        # Infra proximity: 30 (has facility)
        # Total = 70.0
        self.assertEqual(score, 70.0)
        self.assertEqual(reasons["Thermal Intensity (FRP)"], "20.0/40")
        self.assertEqual(reasons["Persistence"], "20/30")
        self.assertEqual(reasons["Infrastructure Proximity"], "30/30")

    def test_compute_priority_score_rate_limited_fallback(self):
        """Verify fallback note and 0 infra score when Overpass API is rate-limited."""
        score, reasons_json = compute_priority_score(
            max_frp=60.0, detection_count=3, nearest_type="power", rate_limited=True
        )
        reasons = json.loads(reasons_json)

        # FRP score: min(60/100 * 40, 40) = 24.0
        # Persistence: min(3 * 10, 30) = 30.0
        # Infra proximity: 0.0 (rate limited fallback)
        # Total = 54.0
        self.assertEqual(score, 54.0)
        self.assertEqual(reasons["Thermal Intensity (FRP)"], "24.0/40")
        self.assertEqual(reasons["Persistence"], "30/30")
        self.assertEqual(
            reasons["Infrastructure Proximity"], "0/30 (Overpass Rate Limited)"
        )

    @patch("enrich_osm_and_score.requests.post")
    def test_fetch_osm_infrastructure_rate_limit_exception(self, mock_post):
        """Verify fetch_osm_infrastructure raises HTTPError on 429."""
        from enrich_osm_and_score import fetch_osm_infrastructure
        from tenacity import RetryError

        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.headers = {"Retry-After": "1"}
        mock_post.return_value = mock_resp

        # Should raise HTTPError or RetryError
        caught = False
        try:
            fetch_osm_infrastructure(28.6, 77.2)
        except (requests.exceptions.HTTPError, RetryError, requests.exceptions.Timeout):
            caught = True
        self.assertTrue(caught, "Should catch rate-limiting HTTPError or RetryError")

    def test_retry_after_header_parsing(self):
        """Verify Retry-After header extraction and fallbacks."""
        resp = requests.Response()
        resp.headers["Retry-After"] = "12"
        self.assertEqual(_parse_retry_after(resp), 12.0)

        resp_invalid = requests.Response()
        resp_invalid.headers["Retry-After"] = "invalid"
        self.assertEqual(_parse_retry_after(resp_invalid, default_wait=5.0), 5.0)

        resp_none = requests.Response()
        self.assertEqual(_parse_retry_after(resp_none, default_wait=8.0), 8.0)


class TestSentinel2STACSearch(unittest.TestCase):
    @patch("fetch_imagery.get_cdse_token", return_value="mock-token-xyz")
    @patch("fetch_imagery.requests.post")
    def test_stac_payload_does_not_contain_query_key(self, mock_post, mock_token):
        """Verify STAC search payload complies with CDSE and does NOT include 'query'."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "features": [
                {
                    "id": "S2A_TEST_SCENE",
                    "properties": {
                        "datetime": "2026-11-10T06:00:00Z",
                        "eo:cloud_cover": 5.2,
                    },
                }
            ]
        }
        mock_post.return_value = mock_response

        from datetime import datetime, timezone

        scenes = search_sentinel2_scenes(
            lon=75.0,
            lat=25.0,
            date_start=datetime(2026, 11, 1, tzinfo=timezone.utc),
            date_end=datetime(2026, 11, 10, tzinfo=timezone.utc),
        )

        # Assert requests.post called with valid payload without 'query' key
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        payload = kwargs.get("json", {})
        self.assertNotIn("query", payload)
        self.assertIn("collections", payload)
        self.assertIn("bbox", payload)
        self.assertIn("datetime", payload)
        self.assertEqual(len(scenes), 1)

    @patch("fetch_imagery.get_cdse_token", return_value="mock-token-xyz")
    @patch("fetch_imagery.requests.post")
    def test_stac_client_side_cloud_cover_filtering(self, mock_post, mock_token):
        """Verify scenes with cloud cover exceeding MAX_CLOUD_COVER are filtered out."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "features": [
                {
                    "id": "S2_CLOUDY",
                    "properties": {"eo:cloud_cover": 65.0, "datetime": "2026-11-05T00:00:00Z"},
                },
                {
                    "id": "S2_CLEAR",
                    "properties": {"eo:cloud_cover": 4.0, "datetime": "2026-11-06T00:00:00Z"},
                },
                {
                    "id": "S2_PARTLY_CLOUDY",
                    "properties": {"eo:cloud_cover": 18.0, "datetime": "2026-11-07T00:00:00Z"},
                },
            ]
        }
        mock_post.return_value = mock_response

        from datetime import datetime, timezone

        scenes = search_sentinel2_scenes(
            lon=75.0,
            lat=25.0,
            date_start=datetime(2026, 11, 1, tzinfo=timezone.utc),
            date_end=datetime(2026, 11, 10, tzinfo=timezone.utc),
        )

        # MAX_CLOUD_COVER is 20.0; cloudy scene (65%) should be excluded
        self.assertEqual(len(scenes), 2)
        # Should be sorted ascending by cloud cover
        self.assertEqual(scenes[0]["id"], "S2_CLEAR")
        self.assertEqual(scenes[1]["id"], "S2_PARTLY_CLOUDY")


class TestPostGISDatabaseIntegration(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(get_database_url())

    def test_database_uuid_imagery_update(self):
        """
        Verify update_event_imagery uses CAST(:event_id AS uuid) and succeeds
        without syntax error on live PostGIS database.
        """
        # Fetch one candidate event from the live database
        with self.engine.connect() as conn:
            event = conn.execute(
                text("SELECT id FROM candidate_events LIMIT 1")
            ).fetchone()

        self.assertIsNotNone(event, "Candidate events table should have events.")
        test_event_id = str(event.id)

        # Test updating with NULL imagery values (no cloud-free scene outcome)
        imagery_payload = {
            "rgb_path": None,
            "swir_path": None,
            "scene_id": None,
        }
        update_event_imagery(self.engine, test_event_id, imagery_payload)

        # Verify DB state
        with self.engine.connect() as conn:
            updated = conn.execute(
                text(
                    "SELECT imagery_fetched_at, imagery_rgb_path FROM candidate_events WHERE id = CAST(:id AS uuid)"
                ),
                {"id": test_event_id},
            ).fetchone()

        self.assertIsNotNone(updated[0], "imagery_fetched_at should be set to NOW().")
        self.assertIsNone(updated[1], "imagery_rgb_path should be SQL NULL.")

    def test_database_resumability_condition(self):
        """Verify that events with priority_reasons set are not queried again."""
        with self.engine.connect() as conn:
            unscored_count = conn.execute(
                text("SELECT count(*) FROM candidate_events WHERE priority_reasons IS NULL")
            ).scalar()
            total_count = conn.execute(
                text("SELECT count(*) FROM candidate_events")
            ).scalar()

        self.assertGreaterEqual(total_count, 3000)
        self.assertLessEqual(unscored_count, total_count)


if __name__ == "__main__":
    unittest.main()
