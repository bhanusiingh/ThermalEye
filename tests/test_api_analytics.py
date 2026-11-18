"""
Unit Tests for ThermalEye Analytics & Telemetry API Endpoints
=============================================================

Verifies:
  - GET /api/summary
  - GET /api/analytics/states
  - GET /api/analytics/timeline
  - GET /api/providers/health
  - GET /api/events/recent
"""

import unittest
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

import main


class _FakeConn:
    """Minimal asyncpg connection mock dispatching on query order or SQL substring."""

    def __init__(self, *, fetchrow_results=None, fetch_results=None, raise_error=None):
        self._fetchrow_iter = iter(fetchrow_results or [])
        self._fetch_iter = iter(fetch_results or [])
        self._raise_error = raise_error

    async def fetchrow(self, query, *args):
        if self._raise_error is not None:
            raise self._raise_error
        return next(self._fetchrow_iter)

    async def fetch(self, query, *args):
        if self._raise_error is not None:
            raise self._raise_error
        return next(self._fetch_iter)


class _FakePool:
    def __init__(self, conn: _FakeConn):
        self._conn = conn

    @asynccontextmanager
    async def acquire(self):
        yield self._conn


class TestAnalyticsEndpoints(unittest.TestCase):
    def setUp(self):
        # Create TestClient without running lifespan (avoids real DB connect during unit tests)
        self.client = TestClient(main.app, raise_server_exceptions=True)

    # ------------------------------------------------------------------
    # 1. GET /api/summary
    # ------------------------------------------------------------------
    def test_summary_populated(self):
        updated_ts = datetime(2026, 11, 17, 12, 30, 0, tzinfo=timezone.utc)
        conn = _FakeConn(
            fetchrow_results=[
                {
                    "raw_detections": 7669,
                    "min_date": date(2026, 11, 13),
                    "max_date": date(2026, 11, 16),
                    "latest_ingested_at": updated_ts,
                },
                {
                    "candidate_events": 3407,
                    "peak_frp": 114.47,
                    "high_priority": 12,
                    "evidence_available": 5,
                    "latest_event_ts": updated_ts,
                },
            ],
            fetch_results=[
                [
                    {"lon": 75.85, "lat": 30.90},  # Punjab
                    {"lon": 75.50, "lat": 31.10},  # Punjab (duplicate state)
                    {"lon": 85.82, "lat": 20.29},  # Odisha
                    {"lon": 76.78, "lat": 18.82},  # Maharashtra
                ]
            ],
        )

        with patch.object(main, "pool", _FakePool(conn)):
            resp = self.client.get("/api/summary")

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["window"], "2026-11-13 to 2026-11-16")
        self.assertEqual(data["updated_at"], "2026-11-17T12:30:00Z")
        self.assertEqual(data["candidate_events"], 3407)
        self.assertEqual(data["raw_detections"], 7669)
        self.assertEqual(data["peak_frp"], 114.5)
        self.assertEqual(data["states_with_events"], 3)
        self.assertEqual(data["high_priority"], 12)
        self.assertEqual(data["evidence_available"], 5)

    def test_summary_empty_database(self):
        conn = _FakeConn(
            fetchrow_results=[
                {
                    "raw_detections": 0,
                    "min_date": None,
                    "max_date": None,
                    "latest_ingested_at": None,
                },
                {
                    "candidate_events": 0,
                    "peak_frp": None,
                    "high_priority": 0,
                    "evidence_available": 0,
                    "latest_event_ts": None,
                },
            ],
            fetch_results=[[]],
        )

        with patch.object(main, "pool", _FakePool(conn)):
            resp = self.client.get("/api/summary")

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIsNone(data["window"])
        self.assertIsNone(data["updated_at"])
        self.assertIsNone(data["peak_frp"])
        self.assertEqual(data["candidate_events"], 0)
        self.assertEqual(data["raw_detections"], 0)
        self.assertEqual(data["states_with_events"], 0)
        self.assertEqual(data["high_priority"], 0)
        self.assertEqual(data["evidence_available"], 0)

    # ------------------------------------------------------------------
    # 2. GET /api/analytics/states
    # ------------------------------------------------------------------
    def test_analytics_states_spatial_join_and_sorting(self):
        conn = _FakeConn(
            fetch_results=[
                [
                    # 6 events in Punjab (tests top-5 cap, priority/FRP ordering, and evidence count = 2)
                    {
                        "id": "11111111-0000-0000-0000-000000000001",
                        "lon": 75.85,
                        "lat": 30.90,
                        "max_frp": 88.44,
                        "detection_count": 4,
                        "priority_score": 82.0,
                        "imagery_rgb_path": "static/imagery/1_rgb.png",
                        "imagery_swir_path": None,
                        "nearest_osm_name": None,
                    },
                    {
                        "id": "22222222-0000-0000-0000-000000000002",
                        "lon": 75.50,
                        "lat": 31.10,
                        "max_frp": 12.10,
                        "detection_count": 2,
                        "priority_score": 45.0,
                        "imagery_rgb_path": None,
                        "imagery_swir_path": None,
                        "nearest_osm_name": "Jalandhar Industrial Area",
                    },
                    {
                        "id": "33333333-0000-0000-0000-000000000003",
                        "lon": 74.87,
                        "lat": 31.63,
                        "max_frp": 25.00,
                        "detection_count": 1,
                        "priority_score": 60.0,
                        "imagery_rgb_path": None,
                        "imagery_swir_path": None,
                        "nearest_osm_name": None,
                    },
                    {
                        "id": "44444444-0000-0000-0000-000000000004",
                        "lon": 75.40,
                        "lat": 31.03,
                        "max_frp": 19.86,
                        "detection_count": 1,
                        "priority_score": 30.0,
                        "imagery_rgb_path": None,
                        "imagery_swir_path": None,
                        "nearest_osm_name": None,
                    },
                    {
                        "id": "55555555-0000-0000-0000-000000000005",
                        "lon": 75.60,
                        "lat": 31.20,
                        "max_frp": 15.04,
                        "detection_count": 1,
                        "priority_score": 20.0,
                        "imagery_rgb_path": None,
                        "imagery_swir_path": None,
                        "nearest_osm_name": None,
                    },
                    {
                        "id": "66666666-0000-0000-0000-000000000006",
                        "lon": 75.70,
                        "lat": 30.80,
                        "max_frp": 5.00,
                        "detection_count": 1,
                        "priority_score": None,
                        "imagery_rgb_path": None,
                        "imagery_swir_path": None,
                        "nearest_osm_name": None,
                    },
                    # 2 events in Odisha (peak 32.6 -> Moderate, clusters 5, evidence 1)
                    {
                        "id": "aaaaaaaa-0000-0000-0000-000000000001",
                        "lon": 85.82,
                        "lat": 20.29,
                        "max_frp": 32.64,
                        "detection_count": 3,
                        "priority_score": 51.0,
                        "imagery_rgb_path": None,
                        "imagery_swir_path": "static/imagery/a_swir.png",
                        "nearest_osm_name": None,
                    },
                    {
                        "id": "bbbbbbbb-0000-0000-0000-000000000002",
                        "lon": 84.44,
                        "lat": 20.19,
                        "max_frp": 10.20,
                        "detection_count": 2,
                        "priority_score": 18.0,
                        "imagery_rgb_path": None,
                        "imagery_swir_path": None,
                        "nearest_osm_name": None,
                    },
                    # 1 event in Maharashtra (peak 8.4 -> Low, clusters 1, evidence 0)
                    {
                        "id": "cccccccc-0000-0000-0000-000000000001",
                        "lon": 76.78,
                        "lat": 18.82,
                        "max_frp": 8.41,
                        "detection_count": 1,
                        "priority_score": 12.0,
                        "imagery_rgb_path": None,
                        "imagery_swir_path": None,
                        "nearest_osm_name": None,
                    },
                    # 1 point in ocean (unmatched)
                    {
                        "id": "dddddddd-0000-0000-0000-000000000001",
                        "lon": 60.0,
                        "lat": 10.0,
                        "max_frp": 50.0,
                        "detection_count": 1,
                        "priority_score": 90.0,
                        "imagery_rgb_path": None,
                        "imagery_swir_path": None,
                        "nearest_osm_name": None,
                    },
                ]
            ]
        )

        with patch.object(main, "pool", _FakePool(conn)):
            resp = self.client.get("/api/analytics/states")

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data), 3)
        self.assertEqual(
            data[0],
            {
                "state": "Punjab",
                "count": 6,
                "peak_frp": 88.4,
                "clusters": 10,
                "intensity": "High",
                "evidence": 2,
                "events": [
                    {"id": "11111111", "peak_frp": 88.4},
                    {"id": "33333333", "peak_frp": 25.0},
                    {"id": "22222222", "peak_frp": 12.1},
                    {"id": "44444444", "peak_frp": 19.9},
                    {"id": "55555555", "peak_frp": 15.0},
                ],
            },
        )
        self.assertEqual(
            data[1],
            {
                "state": "Odisha",
                "count": 2,
                "peak_frp": 32.6,
                "clusters": 5,
                "intensity": "Moderate",
                "evidence": 1,
                "events": [
                    {"id": "aaaaaaaa", "peak_frp": 32.6},
                    {"id": "bbbbbbbb", "peak_frp": 10.2},
                ],
            },
        )
        self.assertEqual(
            data[2],
            {
                "state": "Maharashtra",
                "count": 1,
                "peak_frp": 8.4,
                "clusters": 1,
                "intensity": "Low",
                "evidence": 0,
                "events": [
                    {"id": "cccccccc", "peak_frp": 8.4},
                ],
            },
        )

    def test_analytics_states_empty(self):
        conn = _FakeConn(fetch_results=[[]])
        with patch.object(main, "pool", _FakePool(conn)):
            resp = self.client.get("/api/analytics/states")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), [])

    # ------------------------------------------------------------------
    # 3. GET /api/analytics/timeline
    # ------------------------------------------------------------------
    def test_analytics_timeline(self):
        conn = _FakeConn(
            fetch_results=[
                [
                    {"day": date(2026, 11, 13), "count": 1253},
                    {"day": date(2026, 11, 14), "count": 1720},
                ],
                [
                    {
                        "day": date(2026, 11, 13),
                        "event_count": 386,
                        "max_frp": 38.56,
                        "recurring_count": 62,
                    },
                    {
                        "day": date(2026, 11, 14),
                        "event_count": 617,
                        "max_frp": 80.68,
                        "recurring_count": 154,
                    },
                ],
            ]
        )

        with patch.object(main, "pool", _FakePool(conn)):
            resp = self.client.get("/api/analytics/timeline")

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(
            data["detections"],
            [
                {"date": "2026-11-13", "value": 1253},
                {"date": "2026-11-14", "value": 1720},
            ],
        )
        self.assertEqual(
            data["candidate_events"],
            [
                {"date": "2026-11-13", "value": 386},
                {"date": "2026-11-14", "value": 617},
            ],
        )
        self.assertEqual(
            data["frp"],
            [
                {"date": "2026-11-13", "value": 38.6},
                {"date": "2026-11-14", "value": 80.7},
            ],
        )
        self.assertEqual(
            data["recurring_hotspots"],
            [
                {"date": "2026-11-13", "value": 62},
                {"date": "2026-11-14", "value": 154},
            ],
        )

    # ------------------------------------------------------------------
    # 4. GET /api/providers/health
    # ------------------------------------------------------------------
    def test_providers_health_all_ok(self):
        recent_ts = datetime.now(timezone.utc) - timedelta(minutes=15)
        conn = _FakeConn(
            fetchrow_results=[
                {"total_detections": 500, "latest_acq_date": date(2026, 11, 17)},
                {
                    "total_events": 120,
                    "osm_enriched": 100,
                    "osm_rate_limited": 20,
                    "imagery_chips": 15,
                    "imagery_inspected": 40,
                    "ml_classified": 15,
                    "latest_updated_at": recent_ts,
                },
            ]
        )

        with patch.object(main, "pool", _FakePool(conn)):
            resp = self.client.get("/api/providers/health")

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        expected_keys = {
            "NASA FIRMS",
            "OpenStreetMap",
            "Copernicus CDSE",
            "MobileNetV2",
            "PostGIS",
            "Background worker",
        }
        self.assertEqual(set(data.keys()), expected_keys)
        for key in expected_keys:
            self.assertEqual(data[key]["status"], "ok", f"Expected {key} to be ok")
            self.assertTrue(data[key]["detail"])
        self.assertEqual(data["PostGIS"]["detail"], "PostgreSQL 15 / PostGIS 3.4 connected")

    def test_providers_health_degraded_and_unknown(self):
        stale_ts = datetime.now(timezone.utc) - timedelta(hours=5)
        conn = _FakeConn(
            fetchrow_results=[
                {"total_detections": 7669, "latest_acq_date": date(2026, 11, 16)},
                {
                    "total_events": 3407,
                    "osm_enriched": 0,
                    "osm_rate_limited": 3407,
                    "imagery_chips": 0,
                    "imagery_inspected": 4,
                    "ml_classified": 0,
                    "latest_updated_at": stale_ts,
                },
            ]
        )

        with patch.object(main, "pool", _FakePool(conn)):
            resp = self.client.get("/api/providers/health")

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["NASA FIRMS"]["status"], "ok")
        self.assertEqual(data["OpenStreetMap"]["status"], "degraded")
        self.assertEqual(data["Copernicus CDSE"]["status"], "degraded")
        self.assertEqual(data["MobileNetV2"]["status"], "unknown")
        self.assertEqual(data["PostGIS"]["status"], "ok")
        self.assertEqual(data["Background worker"]["status"], "degraded")

    def test_providers_health_db_failure(self):
        conn = _FakeConn(raise_error=RuntimeError("connection refused"))
        with patch.object(main, "pool", _FakePool(conn)):
            resp = self.client.get("/api/providers/health")

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["PostGIS"]["status"], "failing")
        self.assertNotIn("connection refused", data["PostGIS"]["detail"])
        self.assertEqual(data["NASA FIRMS"]["status"], "unknown")

    # ------------------------------------------------------------------
    # 5. GET /api/events/recent
    # ------------------------------------------------------------------
    def test_recent_events_evidence_classification_and_fallback(self):
        id1, id2, id3 = uuid4(), uuid4(), uuid4()
        conn = _FakeConn(
            fetch_results=[
                [
                    {
                        "id": id1,
                        "lon": 75.85,
                        "lat": 30.90,  # Punjab
                        "max_frp": 92.34,
                        "priority_score": 78.5,
                        "nearest_osm_name": "Ludhiana Substation",
                        "imagery_rgb_path": "static/imagery/1_rgb.png",
                        "imagery_swir_path": None,
                        "land_cover_class": "Cropland",
                    },
                    {
                        "id": id2,
                        "lon": 85.82,
                        "lat": 20.29,  # Odisha
                        "max_frp": 41.0,
                        "priority_score": 44.2,
                        "nearest_osm_name": None,
                        "imagery_rgb_path": None,
                        "imagery_swir_path": None,
                        "land_cover_class": None,
                    },
                    {
                        "id": id3,
                        "lon": 60.0,
                        "lat": 10.0,  # Unmatched -> fallback 'India'
                        "max_frp": None,
                        "priority_score": None,
                        "nearest_osm_name": None,
                        "imagery_rgb_path": None,
                        "imagery_swir_path": None,
                        "land_cover_class": None,
                    },
                ]
            ]
        )

        with patch.object(main, "pool", _FakePool(conn)):
            resp = self.client.get("/api/events/recent")

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data), 3)
        self.assertEqual(
            data[0],
            {
                "id": str(id1),
                "state": "Punjab",
                "peak_frp": 92.3,
                "evidence": "available",
            },
        )
        self.assertEqual(
            data[1],
            {
                "id": str(id2),
                "state": "Odisha",
                "peak_frp": 41.0,
                "evidence": "partial",
            },
        )
        self.assertEqual(
            data[2],
            {
                "id": str(id3),
                "state": "India",
                "peak_frp": None,
                "evidence": "unavailable",
            },
        )


if __name__ == "__main__":
    unittest.main()
