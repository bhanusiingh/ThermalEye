"""
ThermalEye — Unified Pipeline Orchestrator
==========================================

Coordinates the sequential execution of all five ThermalEye pipeline stages:
  1. NASA FIRMS Data Ingestion      (ingest_firms.py)
  2. Spatial Clustering             (cluster_events.py)
  3. OSM Enrichment & Priority Scoring (enrich_osm_and_score.py)
  4. Sentinel-2 Imagery Retrieval   (fetch_imagery.py)
  5. ML Land-Cover Classification   (classify_landcover.py)

Architectural invariants:
  • Sequential Execution: Stages run in strict dependency order (1 -> 2 -> 3 -> 4 -> 5).
  • Fault Isolation: Each stage is isolated in a try...except block. Non-critical
    stage failures log full tracebacks and do not crash the pipeline run.
  • CLI Flexibility: Supports single-run, daemon loop, and --skip-imagery modes.
  • Telemetry: Logs stage-level execution durations, timestamps, and a consolidated
    PostGIS database summary table.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Walk up to find .env if not loaded
_script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(_script_dir, ".env"))

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("thermaleye.orchestrator")


# ---------------------------------------------------------------------------
# Database Utilities
# ---------------------------------------------------------------------------
def get_database_url() -> str:
    url = os.environ.get(
        "DATABASE_URL",
        "postgresql://thermaleye:thermaleye_secret@localhost:5432/thermaleye",
    )
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    if "+" not in url.split("://")[0]:
        try:
            import psycopg2
            url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
        except ImportError:
            try:
                import psycopg
                url = url.replace("postgresql://", "postgresql+psycopg://", 1)
            except ImportError:
                pass
    return url


def get_postgis_telemetry(engine) -> Dict[str, int]:
    """Query PostGIS for consolidated table counts."""
    query = text(
        """
        SELECT
            (SELECT count(*) FROM raw_firms_detections) AS total_detections,
            (SELECT count(*) FROM candidate_events) AS total_events,
            (SELECT count(*) FROM candidate_events WHERE priority_score IS NOT NULL) AS scored_events,
            (SELECT count(*) FROM candidate_events WHERE imagery_fetched_at IS NOT NULL) AS imagery_checked_events,
            (SELECT count(*) FROM candidate_events WHERE imagery_rgb_path IS NOT NULL) AS imagery_acquired_events,
            (SELECT count(*) FROM candidate_events WHERE land_cover_class IS NOT NULL) AS classified_events
        """
    )
    try:
        with engine.connect() as conn:
            row = conn.execute(query).fetchone()
            if row:
                m = row._mapping
                return {
                    "raw_firms_detections": int(m["total_detections"]),
                    "candidate_events": int(m["total_events"]),
                    "scored_events": int(m["scored_events"]),
                    "imagery_checked_events": int(m["imagery_checked_events"]),
                    "imagery_acquired_events": int(m["imagery_acquired_events"]),
                    "classified_events": int(m["classified_events"]),
                }
    except Exception as exc:
        logger.error("Failed to query PostGIS telemetry: %s", exc)
    return {}


def format_telemetry_table(telemetry: Dict[str, int]) -> str:
    """Format PostGIS row counts into a clean ASCII summary table."""
    lines = [
        "=" * 66,
        "  ThermalEye PostGIS Consolidated Telemetry Summary",
        "=" * 66,
        f"  Raw FIRMS Detections             : {telemetry.get('raw_firms_detections', 0):,}",
        f"  Candidate Fire Events            : {telemetry.get('candidate_events', 0):,}",
        f"  Priority Scored Events (0-100)   : {telemetry.get('scored_events', 0):,}",
        f"  Imagery Inspected Events         : {telemetry.get('imagery_checked_events', 0):,}",
        f"  Imagery Acquired Chips (RGB)     : {telemetry.get('imagery_acquired_events', 0):,}",
        f"  Land-Cover Classified Events     : {telemetry.get('classified_events', 0):,}",
        "=" * 66,
    ]
    return "\n".join(lines)


def format_stage_summary_table(stages: List[Dict[str, Any]], total_duration: float) -> str:
    """Format per-stage execution durations and statuses."""
    lines = [
        "-" * 72,
        f" {'Pipeline Stage':<42} | {'Status':<10} | {'Duration':<12}",
        "-" * 72,
    ]
    for s in stages:
        dur = f"{s['duration']:.2f}s"
        lines.append(f" {s['name']:<42} | {s['status']:<10} | {dur:<12}")
    lines.append("-" * 72)
    lines.append(f" Total Pipeline Execution Time: {total_duration:.2f}s")
    lines.append("-" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage Execution Definition
# ---------------------------------------------------------------------------
class PipelineStage:
    def __init__(
        self,
        stage_num: int,
        name: str,
        entrypoint: Callable[[], Any],
        is_imagery_dependent: bool = False,
    ):
        self.stage_num = stage_num
        self.name = f"{stage_num}. {name}"
        self.entrypoint = entrypoint
        self.is_imagery_dependent = is_imagery_dependent


def build_pipeline_stages() -> List[PipelineStage]:
    """
    Import and construct the 5 sequential pipeline stages.
    Dynamic import allows mocking and prevents module import failures
    from breaking CLI inspection.
    """
    from ingest_firms import run_ingestion
    from cluster_events import run_clustering
    from enrich_osm_and_score import process_enrichment
    from fetch_imagery import run_imagery_fetch
    from classify_landcover import run_classification

    return [
        PipelineStage(1, "FIRMS Ingestion", run_ingestion),
        PipelineStage(2, "Spatial Clustering", run_clustering),
        PipelineStage(3, "OSM Enrichment & Priority Scoring", process_enrichment),
        PipelineStage(4, "Sentinel-2 Imagery Retrieval", run_imagery_fetch, is_imagery_dependent=True),
        PipelineStage(5, "ML Land-Cover Classification", run_classification, is_imagery_dependent=True),
    ]


# ---------------------------------------------------------------------------
# Pipeline Runner
# ---------------------------------------------------------------------------
def run_pipeline(
    skip_imagery: bool = False,
    stages: Optional[List[PipelineStage]] = None,
    engine: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Execute a single pass through the ThermalEye data pipeline.

    Args:
        skip_imagery: When True, bypasses Sentinel-2 download and ML classification.
        stages: Optional list of PipelineStage instances (useful for testing).
        engine: Optional SQLAlchemy engine.

    Returns:
        Dict containing run telemetry, stage results, and final PostGIS counts.
    """
    start_time = time.time()
    start_iso = datetime.now(timezone.utc).isoformat()
    logger.info("=" * 72)
    logger.info("ThermalEye Pipeline Orchestrator — Run Started at %s", start_iso)
    if skip_imagery:
        logger.info("Mode: Fast update (--skip-imagery enabled, stages 4 & 5 will be bypassed)")
    logger.info("=" * 72)

    if stages is None:
        stages = build_pipeline_stages()

    if engine is None:
        engine = create_engine(get_database_url(), echo=False)

    stage_records: List[Dict[str, Any]] = []

    for stage in stages:
        stage_record: Dict[str, Any] = {
            "stage_num": stage.stage_num,
            "name": stage.name,
            "status": "PENDING",
            "duration": 0.0,
            "error": None,
            "result": None,
        }

        if skip_imagery and stage.is_imagery_dependent:
            logger.info("Skipping %s (--skip-imagery active).", stage.name)
            stage_record["status"] = "SKIPPED"
            stage_records.append(stage_record)
            continue

        stage_start = time.time()
        logger.info(">>> Executing %s...", stage.name)

        try:
            result = stage.entrypoint()
            stage_duration = time.time() - stage_start
            stage_record["status"] = "SUCCESS"
            stage_record["duration"] = stage_duration
            stage_record["result"] = result
            logger.info(
                "<<< Finished %s in %.2fs. Result: %s",
                stage.name,
                stage_duration,
                result,
            )
        except Exception as exc:
            stage_duration = time.time() - stage_start
            stage_record["status"] = "FAILED"
            stage_record["duration"] = stage_duration
            stage_record["error"] = str(exc)
            logger.error("!!! Failure in %s after %.2fs:", stage.name, stage_duration)
            logger.error(traceback.format_exc())
            logger.warning("Continuing pipeline execution despite failure in %s.", stage.name)

        stage_records.append(stage_record)

    total_duration = time.time() - start_time
    end_iso = datetime.now(timezone.utc).isoformat()

    # Query PostGIS row counts
    telemetry = get_postgis_telemetry(engine)

    # Print summary reports
    logger.info("\n" + format_stage_summary_table(stage_records, total_duration))
    logger.info("\n" + format_telemetry_table(telemetry))

    logger.info("ThermalEye Pipeline Run Completed at %s (Total: %.2fs)", end_iso, total_duration)
    logger.info("=" * 72)

    return {
        "start_time": start_iso,
        "end_time": end_iso,
        "total_duration": total_duration,
        "stages": stage_records,
        "telemetry": telemetry,
    }


# ---------------------------------------------------------------------------
# CLI Argument Parsing & Daemon Loop
# ---------------------------------------------------------------------------
def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ThermalEye Unified Pipeline Orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        default=False,
        help="Run continuously in a loop with a configurable sleep interval.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=3600,
        help="Polling interval in seconds for daemon mode.",
    )
    parser.add_argument(
        "--skip-imagery",
        action="store_true",
        default=False,
        help="Bypass Sentinel-2 download and ML classification for rapid updates.",
    )
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()

    if not args.daemon:
        # Run once
        run_pipeline(skip_imagery=args.skip_imagery)
        return

    # Daemon mode
    logger.info("ThermalEye Orchestrator starting in DAEMON mode (Interval: %ds)", args.interval)
    logger.info("Press Ctrl+C to terminate.")

    cycle_count = 0
    try:
        while True:
            cycle_count += 1
            logger.info("Starting daemon pipeline iteration #%d...", cycle_count)
            run_pipeline(skip_imagery=args.skip_imagery)
            logger.info(
                "Iteration #%d complete. Sleeping for %ds until next scheduled run...",
                cycle_count,
                args.interval,
            )
            time.sleep(args.interval)
    except KeyboardInterrupt:
        logger.info("ThermalEye Orchestrator daemon received termination signal (Ctrl+C). Exiting.")
        sys.exit(0)


if __name__ == "__main__":
    main()
