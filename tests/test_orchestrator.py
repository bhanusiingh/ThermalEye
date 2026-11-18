"""
Unit and Integration Tests for ThermalEye Pipeline Orchestrator
===============================================================
"""

import unittest
from unittest.mock import MagicMock, call, patch

from run_pipeline import (
    PipelineStage,
    build_pipeline_stages,
    format_stage_summary_table,
    format_telemetry_table,
    parse_args,
    run_pipeline,
)


class TestOrchestratorCLI(unittest.TestCase):
    def test_cli_defaults(self):
        """Verify default CLI arguments."""
        args = parse_args([])
        self.assertFalse(args.daemon)
        self.assertEqual(args.interval, 3600)
        self.assertFalse(args.skip_imagery)

    def test_cli_custom_args(self):
        """Verify CLI flag parsing for daemon mode, interval, and skip-imagery."""
        args = parse_args(["--daemon", "--interval", "1800", "--skip-imagery"])
        self.assertTrue(args.daemon)
        self.assertEqual(args.interval, 1800)
        self.assertTrue(args.skip_imagery)


class TestPipelineOrchestration(unittest.TestCase):
    def setUp(self):
        # Create mock stages with tracked call order
        self.call_order = []

        def make_stage_fn(name):
            def fn():
                self.call_order.append(name)
                return {"status": "ok"}
            return fn

        self.mock_ingest = MagicMock(side_effect=make_stage_fn("ingest"))
        self.mock_cluster = MagicMock(side_effect=make_stage_fn("cluster"))
        self.mock_enrich = MagicMock(side_effect=make_stage_fn("enrich"))
        self.mock_imagery = MagicMock(side_effect=make_stage_fn("imagery"))
        self.mock_classify = MagicMock(side_effect=make_stage_fn("classify"))

        self.test_stages = [
            PipelineStage(1, "FIRMS Ingestion", self.mock_ingest),
            PipelineStage(2, "Spatial Clustering", self.mock_cluster),
            PipelineStage(3, "OSM Enrichment & Priority Scoring", self.mock_enrich),
            PipelineStage(4, "Sentinel-2 Imagery Retrieval", self.mock_imagery, is_imagery_dependent=True),
            PipelineStage(5, "ML Land-Cover Classification", self.mock_classify, is_imagery_dependent=True),
        ]
        self.mock_engine = MagicMock()

    @patch("run_pipeline.get_postgis_telemetry")
    def test_strict_sequential_execution_order(self, mock_telemetry):
        """Verify stages execute in exact 1 -> 2 -> 3 -> 4 -> 5 order."""
        mock_telemetry.return_value = {
            "raw_firms_detections": 100,
            "candidate_events": 20,
            "scored_events": 20,
            "imagery_checked_events": 5,
            "imagery_acquired_events": 2,
            "classified_events": 2,
        }

        result = run_pipeline(
            skip_imagery=False,
            stages=self.test_stages,
            engine=self.mock_engine,
        )

        expected_order = ["ingest", "cluster", "enrich", "imagery", "classify"]
        self.assertEqual(self.call_order, expected_order)

        # Check all stages succeeded
        for stage in result["stages"]:
            self.assertEqual(stage["status"], "SUCCESS")
            self.assertGreaterEqual(stage["duration"], 0.0)

    @patch("run_pipeline.get_postgis_telemetry")
    def test_skip_imagery_bypasses_stages_4_and_5(self, mock_telemetry):
        """Verify --skip-imagery bypasses Sentinel-2 download and ML classification."""
        mock_telemetry.return_value = {
            "raw_firms_detections": 100,
            "candidate_events": 20,
            "scored_events": 20,
            "imagery_checked_events": 0,
            "imagery_acquired_events": 0,
            "classified_events": 0,
        }

        result = run_pipeline(
            skip_imagery=True,
            stages=self.test_stages,
            engine=self.mock_engine,
        )

        # Only stages 1, 2, 3 should have executed
        self.assertEqual(self.call_order, ["ingest", "cluster", "enrich"])
        self.mock_imagery.assert_not_called()
        self.mock_classify.assert_not_called()

        # Check stage status breakdown
        stages_by_num = {s["stage_num"]: s for s in result["stages"]}
        self.assertEqual(stages_by_num[1]["status"], "SUCCESS")
        self.assertEqual(stages_by_num[2]["status"], "SUCCESS")
        self.assertEqual(stages_by_num[3]["status"], "SUCCESS")
        self.assertEqual(stages_by_num[4]["status"], "SKIPPED")
        self.assertEqual(stages_by_num[5]["status"], "SKIPPED")

    @patch("run_pipeline.get_postgis_telemetry")
    def test_fault_isolation_non_critical_enrichment_failure(self, mock_telemetry):
        """
        Verify that an exception in Stage 3 (OSM Enrichment) is isolated:
        it logs traceback, marks stage as FAILED, and allows Stage 4 & 5 to execute.
        """
        mock_telemetry.return_value = {}
        self.mock_enrich.side_effect = RuntimeError("Overpass HTTP 504 Gateway Timeout")

        result = run_pipeline(
            skip_imagery=False,
            stages=self.test_stages,
            engine=self.mock_engine,
        )

        stages_by_num = {s["stage_num"]: s for s in result["stages"]}
        self.assertEqual(stages_by_num[1]["status"], "SUCCESS")
        self.assertEqual(stages_by_num[2]["status"], "SUCCESS")
        self.assertEqual(stages_by_num[3]["status"], "FAILED")
        self.assertIn("Overpass HTTP 504", stages_by_num[3]["error"])
        # Stage 4 and 5 must still execute
        self.assertEqual(stages_by_num[4]["status"], "SUCCESS")
        self.assertEqual(stages_by_num[5]["status"], "SUCCESS")

        self.mock_imagery.assert_called_once()
        self.mock_classify.assert_called_once()

    @patch("run_pipeline.get_postgis_telemetry")
    def test_fault_isolation_ingestion_failure(self, mock_telemetry):
        """
        Verify that an exception in Stage 1 (FIRMS Ingestion) is isolated:
        it logs error and allows subsequent stages to run on existing database rows.
        """
        mock_telemetry.return_value = {}
        self.mock_ingest.side_effect = ConnectionError("NASA FIRMS API connection refused")

        result = run_pipeline(
            skip_imagery=False,
            stages=self.test_stages,
            engine=self.mock_engine,
        )

        stages_by_num = {s["stage_num"]: s for s in result["stages"]}
        self.assertEqual(stages_by_num[1]["status"], "FAILED")
        self.assertIn("NASA FIRMS API connection refused", stages_by_num[1]["error"])
        # Stages 2 through 5 must continue
        self.assertEqual(stages_by_num[2]["status"], "SUCCESS")
        self.assertEqual(stages_by_num[3]["status"], "SUCCESS")
        self.assertEqual(stages_by_num[4]["status"], "SUCCESS")
        self.assertEqual(stages_by_num[5]["status"], "SUCCESS")


class TestTelemetryAndFormatting(unittest.TestCase):
    def test_format_telemetry_table(self):
        """Verify telemetry table string formatting."""
        telemetry = {
            "raw_firms_detections": 7669,
            "candidate_events": 3407,
            "scored_events": 3407,
            "imagery_checked_events": 4,
            "imagery_acquired_events": 1,
            "classified_events": 1,
        }
        table_str = format_telemetry_table(telemetry)
        self.assertIn("7,669", table_str)
        self.assertIn("3,407", table_str)
        self.assertIn("ThermalEye PostGIS Consolidated Telemetry Summary", table_str)

    def test_format_stage_summary_table(self):
        """Verify stage summary table string formatting."""
        stages = [
            {"name": "1. FIRMS Ingestion", "status": "SUCCESS", "duration": 3.456},
            {"name": "2. Spatial Clustering", "status": "SUCCESS", "duration": 0.821},
            {"name": "3. OSM Enrichment & Priority Scoring", "status": "FAILED", "duration": 1.234},
            {"name": "4. Sentinel-2 Imagery Retrieval", "status": "SKIPPED", "duration": 0.0},
        ]
        table_str = format_stage_summary_table(stages, total_duration=5.511)
        self.assertIn("1. FIRMS Ingestion", table_str)
        self.assertIn("SUCCESS", table_str)
        self.assertIn("FAILED", table_str)
        self.assertIn("SKIPPED", table_str)
        self.assertIn("Total Pipeline Execution Time: 5.51s", table_str)


if __name__ == "__main__":
    unittest.main()
