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

import io
import json
import os
import tempfile
import unittest
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
from PIL import Image

import main


class _FakeConn:
    """Minimal asyncpg connection mock dispatching on query order or SQL substring."""

    def __init__(self, *, fetchrow_results=None, fetch_results=None, fetchval_results=None, raise_error=None):
        self._fetchrow_iter = iter(fetchrow_results or [])
        self._fetch_iter = iter(fetch_results or [])
        self._fetchval_iter = iter(fetchval_results or [])
        self._raise_error = raise_error

    async def fetchrow(self, query, *args):
        if self._raise_error is not None:
            raise self._raise_error
        try:
            return next(self._fetchrow_iter)
        except StopIteration:
            return None

    async def fetch(self, query, *args):
        if self._raise_error is not None:
            raise self._raise_error
        try:
            return next(self._fetch_iter)
        except StopIteration:
            return []

    async def fetchval(self, query, *args):
        if self._raise_error is not None:
            raise self._raise_error
        try:
            return next(self._fetchval_iter)
        except StopIteration:
            return None


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
                "lat": 30.9,
                "lon": 75.85,
                "peak_frp": 92.3,
                "evidence": "available",
                "imagery_rgb_path": "static/imagery/1_rgb.png",
                "imagery_swir_path": None,
            },
        )
        self.assertEqual(
            data[1],
            {
                "id": str(id2),
                "state": "Odisha",
                "lat": 20.29,
                "lon": 85.82,
                "peak_frp": 41.0,
                "evidence": "partial",
                "imagery_rgb_path": None,
                "imagery_swir_path": None,
            },
        )
        self.assertEqual(
            data[2],
            {
                "id": str(id3),
                "state": "India",
                "lat": 10.0,
                "lon": 60.0,
                "peak_frp": None,
                "evidence": "unavailable",
                "imagery_rgb_path": None,
                "imagery_swir_path": None,
            },
        )

    # ------------------------------------------------------------------
    # 6. GET /api/ml/metrics
    # ------------------------------------------------------------------
    def test_get_ml_metrics_populated(self):
        """Verify /api/ml/metrics reads real artifact files when present."""
        resp = self.client.get("/api/ml/metrics")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertIn("dataset_stats", data)
        self.assertIn("model_comparison", data)
        self.assertIn("evaluation_report", data)
        self.assertIn("experimental_validation", data)

        # In this workspace, artifacts/ has real generated benchmark artifacts
        if os.path.exists(os.path.join(main.ARTIFACTS_DIR, "dataset_stats.json")):
            self.assertIsNotNone(data["dataset_stats"])
            self.assertIn("total_samples", data["dataset_stats"])
            self.assertIn("domain_classes", data["dataset_stats"])
            self.assertIn("class_counts", data["dataset_stats"])

        if os.path.exists(os.path.join(main.ARTIFACTS_DIR, "model_comparison.json")):
            self.assertIsNotNone(data["model_comparison"])
            self.assertIn("configurations", data["model_comparison"])
            self.assertIn("benchmark_summary", data["model_comparison"])

    def test_get_ml_metrics_missing_files(self):
        """Verify /api/ml/metrics returns null for missing artifacts without fabricating or failing."""
        with tempfile.TemporaryDirectory() as empty_dir:
            with patch.object(main, "ARTIFACTS_DIR", empty_dir):
                resp = self.client.get("/api/ml/metrics")

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(
            data,
            {
                "dataset_stats": None,
                "model_comparison": None,
                "evaluation_report": None,
                "experimental_validation": None,
            },
        )

    # ------------------------------------------------------------------
    # 7. ML & Grad-CAM properties on /api/events and /api/events/{event_id}
    # ------------------------------------------------------------------
    def test_events_endpoint_includes_gradcam_and_ml_attributes(self):
        """Verify GET /api/events exposes gradcam_url, land_cover_confidence, and ml_model_arch."""
        event_id = uuid4()
        conn = _FakeConn(
            fetch_results=[
                [
                    {
                        "id": event_id,
                        "geojson": json.dumps({"type": "Point", "coordinates": [75.85, 30.90]}),
                        "footprint_geojson": None,
                        "first_seen": datetime(2026, 11, 15, 10, 0, tzinfo=timezone.utc),
                        "last_seen": datetime(2026, 11, 15, 11, 0, tzinfo=timezone.utc),
                        "max_frp": 85.5,
                        "priority_score": 75.0,
                        "priority_reasons_raw": json.dumps({"base": 40, "frp": 35}),
                        "nearest_osm_type": "industrial",
                        "nearest_osm_name": "Steel Rolling Mill",
                        "nearest_osm_distance_km": 0.45,
                        "land_cover_class": "industrial_thermal",
                        "land_cover_confidence": 0.942,
                        "ml_model_arch": "efficientnet_b0",
                        "ml_disclaimer": "Experimental model classification",
                        "imagery_rgb_path": "static/imagery/test_rgb.png",
                        "imagery_swir_path": "static/imagery/test_swir.png",
                        "gradcam_path": "static/imagery/test_gradcam.png",
                        "sentinel2_scene_id": "S2A_MSIL2A_TEST",
                        "imagery_fetched_at": datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc),
                        "detection_count": 5,
                        "created_at": datetime(2026, 11, 15, 10, 0, tzinfo=timezone.utc),
                        "updated_at": datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc),
                    }
                ]
            ]
        )

        with patch.object(main, "pool", _FakePool(conn)):
            resp = self.client.get("/api/events")

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["type"], "FeatureCollection")
        self.assertEqual(len(data["features"]), 1)
        props = data["features"][0]["properties"]

        self.assertEqual(props["id"], str(event_id))
        self.assertEqual(props["gradcam_url"], "/static/imagery/test_gradcam.png")
        self.assertEqual(props["land_cover_confidence"], 0.942)
        self.assertEqual(props["ml_model_arch"], "efficientnet_b0")
        self.assertEqual(props["land_cover_class"], "industrial_thermal")

    def test_event_dossier_endpoint_includes_gradcam_and_ml_attributes(self):
        """Verify GET /api/events/{event_id} exposes gradcam_url, land_cover_confidence, and ml_model_arch in dossier."""
        event_id = uuid4()
        conn = _FakeConn(
            fetchrow_results=[
                {
                    "id": event_id,
                    "geojson": json.dumps({"type": "Point", "coordinates": [75.85, 30.90]}),
                    "footprint_geojson": None,
                    "first_seen": datetime(2026, 11, 15, 10, 0, tzinfo=timezone.utc),
                    "last_seen": datetime(2026, 11, 15, 11, 0, tzinfo=timezone.utc),
                    "max_frp": 85.5,
                    "priority_score": 75.0,
                    "priority_reasons_raw": json.dumps({"base": 40, "frp": 35}),
                    "nearest_osm_type": "industrial",
                    "nearest_osm_name": "Steel Rolling Mill",
                    "nearest_osm_distance_km": 0.45,
                    "land_cover_class": "industrial_thermal",
                    "land_cover_confidence": 0.942,
                    "ml_model_arch": "efficientnet_b0",
                    "ml_disclaimer": "Experimental model classification",
                    "imagery_rgb_path": "static/imagery/test_rgb.png",
                    "imagery_swir_path": "static/imagery/test_swir.png",
                    "gradcam_path": "static/imagery/test_gradcam.png",
                    "sentinel2_scene_id": "S2A_MSIL2A_TEST",
                    "imagery_fetched_at": datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc),
                    "detection_count": 5,
                    "created_at": datetime(2026, 11, 15, 10, 0, tzinfo=timezone.utc),
                    "updated_at": datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc),
                }
            ],
            fetch_results=[[]],  # raw detections
        )

        with patch.object(main, "pool", _FakePool(conn)):
            resp = self.client.get(f"/api/events/{event_id}")

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["type"], "Feature")
        props = data["properties"]

        self.assertEqual(props["id"], str(event_id))
        self.assertEqual(props["gradcam_url"], "/static/imagery/test_gradcam.png")
        self.assertEqual(props["land_cover_confidence"], 0.942)
        self.assertEqual(props["ml_model_arch"], "efficientnet_b0")
        self.assertIn("raw_detections", props)

    # ------------------------------------------------------------------
    # POST /api/analyze-sample Tests
    # ------------------------------------------------------------------
    def _create_test_image_bytes(self, color=(128, 128, 128), size=(64, 64), format="PNG"):
        img = Image.new("RGB", size, color=color)
        buf = io.BytesIO()
        img.save(buf, format=format)
        buf.seek(0)
        return buf.getvalue()

    def test_analyze_sample_rgb_only_with_coordinates(self):
        """Verify POST /api/analyze-sample with RGB-only upload and valid coordinates."""
        img_bytes = self._create_test_image_bytes(color=(200, 100, 50))

        conn = _FakeConn(
            fetchval_results=[True, 2],  # inside_india = True, nearby_events_5km = 2
            fetchrow_results=[
                {
                    "nearest_osm_name": "Tata Steel Plant",
                    "nearest_osm_type": "industrial",
                    "dist_km": 1.85,
                }
            ],
        )

        with patch.object(main, "pool", _FakePool(conn)):
            resp = self.client.post(
                "/api/analyze-sample",
                files={"rgb_file": ("satellite_rgb.png", img_bytes, "image/png")},
                data={"latitude": "23.7957", "longitude": "86.4304"},
            )

        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertIn("sample_id", data)
        self.assertEqual(data["modality"], "RGB-Only (SWIR Synthesized)")
        self.assertIn(data["predicted_class"], [
            "industrial_thermal",
            "agricultural_stubble_burn",
            "forest_wildfire",
            "urban_false_positive",
        ])
        self.assertGreaterEqual(data["confidence"], 0.0)
        self.assertLessEqual(data["confidence"], 1.0)
        self.assertEqual(len(data["class_probabilities"]), 4)
        self.assertGreaterEqual(data["inference_latency_ms"], 0.0)

        # File URLs
        self.assertTrue(data["rgb_url"].startswith("/static/uploads/"))
        self.assertIsNone(data["swir_url"])
        self.assertTrue(data["gradcam_url"].startswith("/static/uploads/"))

        # Verify physical files written to static/uploads/
        rgb_disk_path = os.path.join(main.STATIC_DIR, data["rgb_url"].lstrip("/static/"))
        gradcam_disk_path = os.path.join(main.STATIC_DIR, data["gradcam_url"].lstrip("/static/"))
        self.assertTrue(os.path.isfile(rgb_disk_path))
        self.assertTrue(os.path.isfile(gradcam_disk_path))

        # Location verification
        loc = data["location"]
        self.assertAlmostEqual(loc["latitude"], 23.7957, places=3)
        self.assertAlmostEqual(loc["longitude"], 86.4304, places=3)
        self.assertTrue(loc["inside_india"])
        self.assertEqual(loc["state"], "Jharkhand")
        self.assertEqual(loc["nearby_events_5km"], 2)
        self.assertIn("Tata Steel Plant", loc["nearest_infrastructure"])

        # Clean up created upload files
        if os.path.isfile(rgb_disk_path):
            os.remove(rgb_disk_path)
        if os.path.isfile(gradcam_disk_path):
            os.remove(gradcam_disk_path)

    def test_analyze_sample_paired_rgb_and_swir(self):
        """Verify POST /api/analyze-sample with paired RGB and SWIR uploads."""
        rgb_bytes = self._create_test_image_bytes(color=(180, 120, 60))
        swir_bytes = self._create_test_image_bytes(color=(50, 150, 200))

        conn = _FakeConn(
            fetchval_results=[True, 0],
            fetchrow_results=[None],
        )

        with patch.object(main, "pool", _FakePool(conn)):
            resp = self.client.post(
                "/api/analyze-sample",
                files={
                    "rgb_file": ("chip_rgb.png", rgb_bytes, "image/png"),
                    "swir_file": ("chip_swir.png", swir_bytes, "image/png"),
                },
                data={"latitude": "23.7957", "longitude": "86.4304"},
            )

        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["modality"], "RGB + SWIR (Paired)")
        self.assertIsNotNone(data["swir_url"])
        self.assertTrue(data["swir_url"].startswith("/static/uploads/"))
        self.assertTrue(data["gradcam_url"].startswith("/static/uploads/"))

        # Clean up files
        for url_key in ("rgb_url", "swir_url", "gradcam_url"):
            url = data.get(url_key)
            if url:
                disk_path = os.path.join(main.STATIC_DIR, url.lstrip("/static/"))
                if os.path.isfile(disk_path):
                    os.remove(disk_path)

    def test_analyze_sample_invalid_file_format(self):
        """Verify 400 error when non-image or corrupted file is uploaded."""
        resp = self.client.post(
            "/api/analyze-sample",
            files={"rgb_file": ("notes.txt", b"this is not an image", "text/plain")},
            data={"latitude": "23.7957", "longitude": "86.4304"},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Unsupported", resp.json()["detail"])

    def test_analyze_sample_out_of_bounds_coordinates(self):
        """Verify 400 error when latitude or longitude exceed WGS84 bounds."""
        img_bytes = self._create_test_image_bytes()
        resp = self.client.post(
            "/api/analyze-sample",
            files={"rgb_file": ("test.png", img_bytes, "image/png")},
            data={"latitude": "95.5", "longitude": "86.4304"},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("out of bounds", resp.json()["detail"].lower())

        resp2 = self.client.post(
            "/api/analyze-sample",
            files={"rgb_file": ("test.png", img_bytes, "image/png")},
            data={"latitude": "23.5", "longitude": "-195.0"},
        )
        self.assertEqual(resp2.status_code, 400)
        self.assertIn("out of bounds", resp2.json()["detail"].lower())

    def test_analyze_sample_no_coordinates_fallback(self):
        """Verify fallback behavior when coordinates are omitted and no EXIF data exists."""
        img_bytes = self._create_test_image_bytes()

        resp = self.client.post(
            "/api/analyze-sample",
            files={"rgb_file": ("sample.png", img_bytes, "image/png")},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        loc = data["location"]
        self.assertIsNone(loc["latitude"])
        self.assertIsNone(loc["longitude"])
        self.assertEqual(loc["status"], "Coordinates not provided")

        # Clean up
        rgb_disk_path = os.path.join(main.STATIC_DIR, data["rgb_url"].lstrip("/static/"))
        gradcam_disk_path = os.path.join(main.STATIC_DIR, data["gradcam_url"].lstrip("/static/"))
        if os.path.isfile(rgb_disk_path):
            os.remove(rgb_disk_path)
        if os.path.isfile(gradcam_disk_path):
            os.remove(gradcam_disk_path)

    # ------------------------------------------------------------------
    # 13. Single-Container Cloud Frontend Serving & Database DSN Resolution
    # ------------------------------------------------------------------
    def test_serve_frontend_index_and_dashboard(self):
        """Verify GET / and GET /dashboard serve production frontend HTML bundles."""
        resp_index = self.client.get("/")
        self.assertEqual(resp_index.status_code, 200)
        self.assertIn("text/html", resp_index.headers.get("content-type", ""))
        self.assertIn("ThermalEye", resp_index.text)

        resp_dash = self.client.get("/dashboard")
        self.assertEqual(resp_dash.status_code, 200)
        self.assertIn("text/html", resp_dash.headers.get("content-type", ""))

    def test_resolve_database_url_preserves_neon_ssl_and_fallbacks(self):
        """Verify resolve_database_url preserves Neon SSL query parameters and falls back to DB_*."""
        orig_db_url = os.environ.get("DATABASE_URL")
        try:
            # 1. Neon postgres URL with query params
            os.environ["DATABASE_URL"] = (
                "postgres://neon_user:neon_pass@ep-cool-fog.ap-southeast-1.aws.neon.tech/neondb"
                "?sslmode=require&channel_binding=require"
            )
            neon_resolved = main.resolve_database_url()
            self.assertTrue(neon_resolved.startswith("postgresql://"))
            self.assertIn("sslmode=require", neon_resolved)
            self.assertIn("channel_binding=require", neon_resolved)

            # 2. Fallback to DB_* vars
            del os.environ["DATABASE_URL"]
            os.environ["DB_HOST"] = "custom-pg-host"
            os.environ["DB_PORT"] = "5433"
            os.environ["DB_NAME"] = "custom_db"
            os.environ["DB_USER"] = "custom_user"
            os.environ["DB_PASSWORD"] = "custom_pass"
            fallback_resolved = main.resolve_database_url()
            self.assertEqual(
                fallback_resolved,
                "postgresql://custom_user:custom_pass@custom-pg-host:5433/custom_db",
            )
        finally:
            if orig_db_url is not None:
                os.environ["DATABASE_URL"] = orig_db_url
            for k in ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"):
                os.environ.pop(k, None)


if __name__ == "__main__":
    unittest.main()
