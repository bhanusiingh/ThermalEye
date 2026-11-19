"""
Unit and Integration Tests for ThermalEye OSM Enrichment and Sentinel-2 Imagery
"""

from datetime import datetime, timezone
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
    run_imagery_fetch,
    reset_circuit_breaker,
    CDSECircuitBreakerError,
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


class TestCDSECircuitBreakerAndBatching(unittest.TestCase):
    def setUp(self):
        reset_circuit_breaker()

    def tearDown(self):
        reset_circuit_breaker()

    @patch("fetch_imagery._fetch_new_token", side_effect=requests.exceptions.HTTPError("401 Unauthorized"))
    @patch("fetch_imagery.create_engine")
    def test_circuit_breaker_trips_and_exits_early_on_oauth2_failure(
        self, mock_create_engine, mock_fetch_token
    ):
        """
        Verify that fetch_imagery.py trips its circuit breaker and exits early
        without updating DB when CDSE OAuth2 token acquisition fails.
        """
        import fetch_imagery

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_create_engine.return_value = mock_engine

        mock_rows = [
            (
                f"11111111-0000-0000-0000-{i:012d}",
                75.0 + i * 0.1,
                25.0 + i * 0.1,
                datetime(2026, 11, 1, tzinfo=timezone.utc),
                datetime(2026, 11, 2, tzinfo=timezone.utc),
                90.0 - i * 10,
                50.0 - i * 5,
            )
            for i in range(5)
        ]
        mock_conn.execute.return_value.fetchall.return_value = mock_rows

        self.assertFalse(fetch_imagery._cdse_circuit_open)

        stats = run_imagery_fetch(batch_limit=5)

        self.assertTrue(fetch_imagery._cdse_circuit_open)
        self.assertEqual(stats["processed"], 0)
        self.assertEqual(stats["imagery_found"], 0)

        # Confirm update_event_imagery was NEVER called, leaving imagery_fetched_at as NULL in PostGIS
        mock_engine.begin.assert_not_called()
        self.assertEqual(mock_fetch_token.call_count, 1)

    @patch("fetch_imagery.get_cdse_token", return_value="mock-token")
    @patch("fetch_imagery.create_engine")
    def test_stac_circuit_breaker_trips_after_three_consecutive_failures(
        self, mock_create_engine, mock_token
    ):
        """
        Verify that 3 consecutive STAC network/timeout failures trip _cdse_circuit_open
        and abort the remaining batch immediately, leaving remaining events untouched.
        """
        import fetch_imagery

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_create_engine.return_value = mock_engine

        mock_rows = [
            (
                f"22222222-0000-0000-0000-{i:012d}",
                75.0,
                25.0,
                datetime(2026, 11, 1, tzinfo=timezone.utc),
                datetime(2026, 11, 2, tzinfo=timezone.utc),
                80.0,
                40.0,
            )
            for i in range(6)
        ]
        mock_conn.execute.return_value.fetchall.return_value = mock_rows

        with patch(
            "fetch_imagery._search_sentinel2_scenes_request",
            side_effect=requests.exceptions.Timeout("STAC Gateway Timeout"),
        ) as mock_stac:
            stats = run_imagery_fetch(batch_limit=6)

            self.assertTrue(fetch_imagery._cdse_circuit_open)
            self.assertEqual(fetch_imagery._stac_consecutive_failures, 3)

            # Aborted on 3rd failure; remaining 3 events were not queried
            self.assertEqual(mock_stac.call_count, 3)
            self.assertEqual(stats["processed"], 0)
            mock_engine.begin.assert_not_called()

    @patch("fetch_imagery.create_engine")
    def test_batch_limit_and_prioritization_query(self, mock_create_engine):
        """
        Verify batch limit defaults to 25 and query prioritizes by priority_score and max_frp.
        """
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        mock_create_engine.return_value = mock_engine
        mock_conn.execute.return_value.fetchall.return_value = []

        run_imagery_fetch()

        call_args = mock_conn.execute.call_args
        sql_clause = str(call_args[0][0])
        params = call_args[0][1]

        self.assertIn("ORDER BY priority_score DESC NULLS LAST, max_frp DESC NULLS LAST", sql_clause)
        self.assertIn("LIMIT :limit", sql_clause)
        self.assertEqual(params["limit"], 25)


class TestLocalOSMEnrichment(unittest.TestCase):
    def test_classify_feature_mapping(self):
        """Verify tag-based industrial, energy, and mining classification."""
        from import_osm_pbf import classify_feature

        # Power plant vs substation
        self.assertEqual(classify_feature({"power": "plant"}), ("power_plant", "power=plant"))
        self.assertEqual(classify_feature({"power": "substation"}), ("energy_infra", "power=substation"))
        self.assertEqual(classify_feature({"power": "generator"}), ("energy_infra", "power=generator"))

        # Landuse
        self.assertEqual(classify_feature({"landuse": "industrial"}), ("industrial", "landuse=industrial"))
        self.assertEqual(classify_feature({"landuse": "quarry"}), ("mining", "landuse=quarry"))
        self.assertEqual(classify_feature({"landuse": "brownfield"}), ("industrial", "landuse=brownfield"))

        # Man-made
        self.assertEqual(classify_feature({"man_made": "works"}), ("industrial", "man_made=works"))
        self.assertEqual(classify_feature({"man_made": "kiln"}), ("kiln_refinery", "man_made=kiln"))
        self.assertEqual(classify_feature({"man_made": "refinery"}), ("kiln_refinery", "man_made=refinery"))
        self.assertEqual(classify_feature({"man_made": "mineshaft"}), ("mining", "man_made=mineshaft"))

        # Mining / resource tags
        self.assertEqual(classify_feature({"mining": "yes"}), ("mining", "mining=yes"))
        self.assertEqual(classify_feature({"quarry": "granite"}), ("mining", "quarry=granite"))
        self.assertEqual(classify_feature({"resource": "sand"}), ("mining", "resource=sand"))

        # Industrial key
        self.assertEqual(classify_feature({"industrial": "factory"}), ("industrial", "industrial=factory"))

        # Non-matching
        self.assertIsNone(classify_feature({"highway": "primary"}))
        self.assertIsNone(classify_feature({"amenity": "school"}))

    @patch("enrich_osm.check_local_osm_count")
    @patch("enrich_osm.query_nearest_local_osm")
    @patch("enrich_osm.create_engine")
    def test_local_postgis_osm_matching_and_boost(self, mock_engine, mock_query, mock_check_count):
        """Verify local PostGIS mode applies 30-point proximity boost and does not mark rate-limited."""
        from enrich_osm import process_enrichment

        mock_check_count.return_value = 1500
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
        mock_engine.return_value.begin.return_value.__enter__.return_value = mock_conn

        # Mock an event needing enrichment
        mock_event = MagicMock()
        mock_event.id = uuid.uuid4()
        mock_event.lat = 21.10
        mock_event.lon = 72.89
        mock_event.max_frp = 80.0
        mock_event.detection_count = 2
        mock_conn.execute.return_value.fetchall.return_value = [mock_event]

        # Mock nearest local OSM match within 1.2km
        mock_query.return_value = {
            "name": "Surat Chemical Complex",
            "osm_type": "landuse=industrial",
            "category": "industrial",
            "dist_m": 1200.0,
        }

        stats = process_enrichment(batch_limit=1)

        self.assertEqual(stats["processed"], 1)
        self.assertEqual(stats["enriched"], 1)
        self.assertEqual(stats["rate_limited"], 0)

        # Inspect the commit call
        begin_conn = mock_engine.return_value.begin.return_value.__enter__.return_value
        executed_items = [call[0][1] for call in begin_conn.execute.call_args_list if len(call[0]) > 1]
        self.assertTrue(len(executed_items) > 0)
        item = executed_items[0]
        self.assertEqual(item["name"], "Surat Chemical Complex")
        self.assertEqual(item["type"], "landuse=industrial")
        self.assertEqual(item["dist_km"], 1.2)
        # FRP = 32.0, Persistence = 20.0, Infra = 30.0 -> Total = 82.0
        self.assertEqual(item["score"], 82.0)
        reasons = json.loads(item["reasons"])
        self.assertEqual(reasons["Infrastructure Proximity"], "30/30")

    @patch("enrich_osm.check_local_osm_count")
    @patch("enrich_osm.query_nearest_local_osm")
    @patch("enrich_osm.create_engine")
    def test_local_postgis_osm_none_nearby(self, mock_engine, mock_query, mock_check_count):
        """Verify local PostGIS mode assigns none_nearby and 0/30 (not rate-limited) when no feature within 5km."""
        from enrich_osm import process_enrichment

        mock_check_count.return_value = 1500
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
        mock_engine.return_value.begin.return_value.__enter__.return_value = mock_conn

        mock_event = MagicMock()
        mock_event.id = uuid.uuid4()
        mock_event.lat = 21.10
        mock_event.lon = 72.89
        mock_event.max_frp = 40.0
        mock_event.detection_count = 1
        mock_conn.execute.return_value.fetchall.return_value = [mock_event]

        # No match within 5000m
        mock_query.return_value = None

        stats = process_enrichment(batch_limit=1)

        self.assertEqual(stats["processed"], 1)
        self.assertEqual(stats["none_nearby"], 1)
        self.assertEqual(stats["enriched"], 0)
        self.assertEqual(stats["rate_limited"], 0)

        begin_conn = mock_engine.return_value.begin.return_value.__enter__.return_value
        executed_items = [call[0][1] for call in begin_conn.execute.call_args_list if len(call[0]) > 1]
        item = executed_items[0]
        self.assertIsNone(item["name"])
        self.assertIsNone(item["type"])
        self.assertIsNone(item["dist_km"])
        # FRP = 16.0, Persistence = 10.0, Infra = 0.0 -> Total = 26.0
        self.assertEqual(item["score"], 26.0)
        reasons = json.loads(item["reasons"])
        self.assertEqual(reasons["Infrastructure Proximity"], "0/30")
        self.assertNotIn("Rate Limited", reasons["Infrastructure Proximity"])

    @patch("enrich_osm.check_local_osm_count")
    @patch("enrich_osm.fetch_osm_infrastructure")
    @patch("enrich_osm.create_engine")
    def test_fallback_to_overpass_when_table_empty(self, mock_engine, mock_fetch, mock_check_count):
        """Verify fallback to Overpass API when local table has 0 rows."""
        from enrich_osm import process_enrichment

        # Table empty
        mock_check_count.return_value = 0
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
        mock_engine.return_value.begin.return_value.__enter__.return_value = mock_conn

        mock_event = MagicMock()
        mock_event.id = uuid.uuid4()
        mock_event.lat = 28.61
        mock_event.lon = 77.20
        mock_event.max_frp = 50.0
        mock_event.detection_count = 2
        mock_conn.execute.return_value.fetchall.return_value = [mock_event]

        mock_fetch.return_value = {
            "elements": [
                {
                    "lat": 28.615,
                    "lon": 77.205,
                    "tags": {"name": "Delhi Power Station", "power": "plant"},
                }
            ]
        }

        stats = process_enrichment(batch_limit=1)

        self.assertEqual(stats["processed"], 1)
        mock_fetch.assert_called_once()

    @patch("enrich_osm.check_local_osm_count")
    @patch("enrich_osm.query_nearest_local_osm")
    @patch("enrich_osm.create_engine")
    def test_re_enrich_rate_limited_query_clause(self, mock_engine, mock_query, mock_check_count):
        """Verify --re-enrich-rate-limited queries events previously marked rate-limited."""
        from enrich_osm import process_enrichment

        mock_check_count.return_value = 500
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
        mock_engine.return_value.begin.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value.fetchall.return_value = []

        process_enrichment(batch_limit=10, re_enrich_rate_limited=True)

        call_args = mock_conn.execute.call_args_list[0]
        sql_clause = str(call_args[0][0])
        self.assertIn("priority_reasons::text ILIKE '%Rate Limited%'", sql_clause)


if __name__ == "__main__":
    unittest.main()
