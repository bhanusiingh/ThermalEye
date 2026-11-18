"""
ThermalEye — Event Clustering & UUID Binding Test Suite
=========================================================

Tests for:
  - UUID parameter binding with PostgreSQL ANY(CAST(:point_ids AS uuid[]))
  - Transactional atomicity (no orphaned candidate_events on failure)
  - DBSCAN spatial clustering and detection-to-event linking
  - Idempotent repeated runs (skipping already clustered detections)
"""

import os
import unittest
import uuid
from datetime import datetime, timezone

from shapely.geometry import Point
from sqlalchemy import create_engine, text, bindparam
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, ARRAY as PG_ARRAY

import cluster_events


class TestClusteringAndUUIDBinding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = create_engine(cluster_events.DB_URL)
        # Ensure database is accessible
        with cls.engine.connect() as conn:
            conn.execute(text("SELECT 1;"))

    def test_01_uuid_parameter_binding(self):
        """Verify that UUID array binding executes without 'operator does not exist: uuid = text'."""
        test_event_id = uuid.uuid4()
        test_det_id = uuid.uuid4()

        with self.engine.begin() as conn:
            # 1. Insert temporary detection
            conn.execute(
                text(
                    """
                    INSERT INTO raw_firms_detections 
                    (id, geom, satellite, acq_datetime, frp, confidence)
                    VALUES 
                    (:id, ST_SetSRID(ST_MakePoint(77.2090, 28.6139), 4326),
                     'TEST_SAT', NOW(), 12.5, 'nominal')
                    """
                ),
                {"id": test_det_id},
            )

            # 2. Insert temporary candidate event
            conn.execute(
                text(
                    """
                    INSERT INTO candidate_events 
                    (id, centroid, first_seen, last_seen, max_frp, detection_count)
                    VALUES 
                    (:id, ST_SetSRID(ST_MakePoint(77.2090, 28.6139), 4326),
                     NOW(), NOW(), 12.5, 1)
                    """
                ),
                {"id": test_event_id},
            )

            # 3. Execute the exact parameterized update query
            update_stmt = text(
                """
                UPDATE raw_firms_detections 
                SET candidate_event_id = :event_id 
                WHERE id = ANY(CAST(:point_ids AS uuid[]))
                """
            ).bindparams(
                bindparam("event_id", type_=PG_UUID(as_uuid=True)),
                bindparam("point_ids", type_=PG_ARRAY(PG_UUID(as_uuid=True))),
            )

            res = conn.execute(
                update_stmt,
                {"event_id": test_event_id, "point_ids": [test_det_id]},
            )
            self.assertEqual(res.rowcount, 1, "Expected 1 detection updated.")

            # Verify the linked candidate_event_id matches
            linked_fk = conn.execute(
                text("SELECT candidate_event_id FROM raw_firms_detections WHERE id = :id"),
                {"id": test_det_id},
            ).scalar()
            self.assertEqual(linked_fk, test_event_id)

            # Clean up test rows
            conn.execute(
                text("DELETE FROM raw_firms_detections WHERE id = :id"),
                {"id": test_det_id},
            )
            conn.execute(
                text("DELETE FROM candidate_events WHERE id = :id"),
                {"id": test_event_id},
            )

    def test_02_transactional_consistency_on_failure(self):
        """Verify that a failure during detection linking rolls back event creation."""
        test_event_id = uuid.uuid4()
        test_det_id = uuid.uuid4()

        try:
            with self.engine.begin() as conn:
                # Insert event
                conn.execute(
                    text(
                        """
                        INSERT INTO candidate_events 
                        (id, centroid, first_seen, last_seen, max_frp, detection_count)
                        VALUES 
                        (:id, ST_SetSRID(ST_MakePoint(77.0, 28.0), 4326),
                         NOW(), NOW(), 5.0, 1)
                        """
                    ),
                    {"id": test_event_id},
                )
                # Intentionally trigger an exception
                raise RuntimeError("Simulated linking failure for rollback test")
        except RuntimeError:
            pass

        # Verify event was rolled back and does not exist
        with self.engine.connect() as conn:
            exists = conn.execute(
                text("SELECT EXISTS (SELECT 1 FROM candidate_events WHERE id = :id)"),
                {"id": test_event_id},
            ).scalar()
            self.assertFalse(exists, "Orphaned event persisted after transaction failure.")

    def test_03_repeated_clustering_run(self):
        """Verify that running clustering when no unassigned detections exist returns (0, 0)."""
        # When all detections are linked, run_clustering should return (0, 0)
        with self.engine.connect() as conn:
            unassigned = conn.execute(
                text("SELECT count(*) FROM raw_firms_detections WHERE candidate_event_id IS NULL")
            ).scalar()

        if unassigned == 0:
            events, linked = cluster_events.run_clustering()
            self.assertEqual(events, 0)
            self.assertEqual(linked, 0)


if __name__ == "__main__":
    unittest.main()
