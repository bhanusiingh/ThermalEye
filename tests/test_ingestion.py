"""
ThermalEye — Ingestion & PostGIS Persistence Test Suite
========================================================

Tests for:
  - Driver compatibility and engine instantiation
  - Bundled India boundary loading, CRS, and validity
  - PostGIS boundary persistence (no copy_expert errors)
  - PostGIS boundary filtering (ST_Intersects + buffer)
  - Deterministic UUID generation
  - Idempotent detection insertion and deduplication
"""

import os
import unittest
import uuid
from datetime import datetime, timezone

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from sqlalchemy import create_engine, text

import ingest_firms


class TestIngestionAndPersistence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = create_engine(ingest_firms.DATABASE_URL)
        # Ensure database is accessible
        with cls.engine.connect() as conn:
            conn.execute(text("SELECT 1;"))

    def test_01_driver_compatibility(self):
        """Verify the resolved SQLAlchemy engine uses psycopg2 without copy_expert issues."""
        url = ingest_firms.DATABASE_URL
        self.assertTrue(
            "postgresql" in url, f"Expected postgresql URL, got {url}"
        )
        engine = create_engine(url)
        driver = engine.dialect.driver
        self.assertIn(
            driver,
            ("psycopg2", "psycopg"),
            f"Expected psycopg2 or psycopg driver, got {driver}",
        )

    def test_02_bundled_boundary_file(self):
        """Verify the bundled data/india_boundary.geojson exists, loads, and has valid geometry."""
        path = os.path.join(
            ingest_firms._script_dir, "data", "india_boundary.geojson"
        )
        self.assertTrue(os.path.exists(path), f"File not found: {path}")

        gdf = gpd.read_file(path)
        self.assertFalse(gdf.empty, "Bundled boundary file is empty.")
        self.assertEqual(gdf.crs.to_epsg(), 4326, f"Expected EPSG:4326, got {gdf.crs}")

        geom = gdf.geometry.iloc[0]
        self.assertTrue(geom.is_valid, "Boundary geometry is invalid.")
        self.assertIn(geom.geom_type, ("Polygon", "MultiPolygon"))

    def test_03_boundary_persistence_in_postgis(self):
        """Verify _ensure_india_boundary persists geometry into PostGIS without errors."""
        ingest_firms._ensure_india_boundary(self.engine)

        with self.engine.connect() as conn:
            table_exists = conn.execute(
                text(
                    """
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables
                        WHERE table_name = 'india_boundary' AND table_schema = 'public'
                    );
                    """
                )
            ).scalar()
            self.assertTrue(table_exists, "Table india_boundary was not created.")

            row_count = conn.execute(
                text("SELECT count(*) FROM india_boundary;")
            ).scalar()
            self.assertGreaterEqual(
                row_count, 1, "Table india_boundary has no rows."
            )

            # Test that ST_Buffer and ST_Intersects work on the persisted geometry
            buffered_valid = conn.execute(
                text(
                    """
                    SELECT ST_IsValid(ST_Buffer(geom, :tol))
                    FROM india_boundary LIMIT 1;
                    """
                ),
                {"tol": ingest_firms.BOUNDARY_TOLERANCE_DEG},
            ).scalar()
            self.assertTrue(buffered_valid, "Buffered geometry is invalid.")

    def test_04_deterministic_id(self):
        """Verify make_detection_id produces reproducible UUIDs."""
        id1 = ingest_firms.make_detection_id(
            latitude=28.6139,
            longitude=77.2090,
            satellite="N20",
            acq_date="2026-11-14",
            acq_time="0530",
        )
        id2 = ingest_firms.make_detection_id(
            latitude=28.6139,
            longitude=77.2090,
            satellite="N20",
            acq_date="2026-11-14",
            acq_time="0530",
        )
        self.assertEqual(id1, id2, "Deterministic ID generation is not stable.")
        self.assertEqual(len(id1), 36, "Generated ID is not a valid UUID string.")

    def test_05_boundary_filtering(self):
        """Verify points inside India pass and points outside India are filtered."""
        # Delhi (inside), Mumbai (inside), London (outside), Tokyo (outside)
        test_points = [
            {"name": "Delhi", "lat": 28.6139, "lon": 77.2090, "inside": True},
            {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777, "inside": True},
            {"name": "London", "lat": 51.5074, "lon": -0.1278, "inside": False},
            {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503, "inside": False},
        ]

        df = pd.DataFrame(
            [
                {
                    "id": str(uuid.uuid4()),
                    "latitude": p["lat"],
                    "longitude": p["lon"],
                    "satellite": "TEST",
                    "acq_datetime": datetime.now(timezone.utc),
                    "frp": 15.0,
                    "confidence": "nominal",
                    "geometry": Point(p["lon"], p["lat"]),
                }
                for p in test_points
            ]
        )
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

        inside, outside = ingest_firms.filter_india_boundary(gdf, self.engine)

        self.assertEqual(len(inside), 2, f"Expected 2 inside, got {len(inside)}")
        self.assertEqual(len(outside), 2, f"Expected 2 outside, got {len(outside)}")

    def test_06_detection_insertion_and_idempotency(self):
        """Verify detection insertion into PostGIS and deduplication (ON CONFLICT DO NOTHING)."""
        test_uuid = str(uuid.uuid4())
        test_data = pd.DataFrame(
            [
                {
                    "id": test_uuid,
                    "satellite": "TEST_SAT",
                    "acq_datetime": datetime.now(timezone.utc),
                    "frp": 25.5,
                    "confidence": "h",
                    "brightness": 320.0,
                    "bright_t31": 298.0,
                    "scan": 0.4,
                    "track": 0.5,
                    "daynight": "D",
                    "version": "1.0",
                    "geometry": Point(77.2090, 28.6139),
                }
            ]
        )
        gdf = gpd.GeoDataFrame(test_data, geometry="geometry", crs="EPSG:4326")

        try:
            # First insert
            inserted_1 = ingest_firms.insert_detections(gdf, self.engine)
            self.assertEqual(inserted_1, 1, "First insert should insert exactly 1 row.")

            # Second insert (same UUID)
            inserted_2 = ingest_firms.insert_detections(gdf, self.engine)
            self.assertEqual(
                inserted_2, 0, "Duplicate insert should be skipped (idempotent)."
            )

            # Verify presence in database
            with self.engine.connect() as conn:
                count = conn.execute(
                    text("SELECT count(*) FROM raw_firms_detections WHERE id = :id;"),
                    {"id": test_uuid},
                ).scalar()
                self.assertEqual(count, 1, "Record should exist in raw_firms_detections.")
        finally:
            # Clean up test row
            with self.engine.begin() as conn:
                conn.execute(
                    text("DELETE FROM raw_firms_detections WHERE id = :id;"),
                    {"id": test_uuid},
                )


if __name__ == "__main__":
    unittest.main()
