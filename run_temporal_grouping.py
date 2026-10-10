"""Root entry point for executing Phase 2 Temporal Event Grouping."""

import argparse
from pathlib import Path
import sys

# Ensure src/ is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from thermaleye.temporal.grouping import (
    DEFAULT_SPATIAL_THRESHOLD_KM,
    DEFAULT_MAX_TIME_GAP_HOURS,
    DEFAULT_EVENTS_OUTPUT_PATH,
    DEFAULT_GROUPED_OUTPUT_PATH,
    run_temporal_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="ThermalEye Phase 2: Temporal Event Grouping")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/samples/india_viirs_recent.csv"),
        help="Path to input FIRMS CSV detections (default: data/samples/india_viirs_recent.csv)",
    )
    parser.add_argument(
        "--spatial-km",
        type=float,
        default=DEFAULT_SPATIAL_THRESHOLD_KM,
        help=f"Spatial distance threshold in km (default: {DEFAULT_SPATIAL_THRESHOLD_KM})",
    )
    parser.add_argument(
        "--time-gap-hours",
        type=float,
        default=DEFAULT_MAX_TIME_GAP_HOURS,
        help=f"Maximum time gap in hours between linked detections (default: {DEFAULT_MAX_TIME_GAP_HOURS})",
    )
    args = parser.parse_args()

    print("Initiating ThermalEye Temporal Event Grouping (Phase 2)...")
    print(f"Input file           : {args.input}")
    print(f"Spatial threshold    : {args.spatial_km} km")
    print(f"Max time gap         : {args.time_gap_hours} hours")

    try:
        events_df, grouped_df = run_temporal_analysis(
            input_csv_path=args.input,
            output_events_path=DEFAULT_EVENTS_OUTPUT_PATH,
            output_grouped_path=DEFAULT_GROUPED_OUTPUT_PATH,
            spatial_distance_km=args.spatial_km,
            max_time_gap_hours=args.time_gap_hours,
        )

        multi_det_events = int((events_df["detection_count"] > 1).sum()) if not events_df.empty else 0
        single_det_events = int((events_df["detection_count"] == 1).sum()) if not events_df.empty else 0
        max_detections = int(events_df["detection_count"].max()) if not events_df.empty else 0

        print("\n================ TEMPORAL GROUPING SUMMARY ================")
        print(f"Raw Input Records    : {len(grouped_df)}")
        print(f"Candidate Events     : {len(events_df)}")
        print(f"Multi-detection Evts : {multi_det_events}")
        print(f"Single-detection Evts: {single_det_events}")
        print(f"Max Detections/Event : {max_detections}")
        print(f"Events Output        : {DEFAULT_EVENTS_OUTPUT_PATH}")
        print(f"Grouped Detections   : {DEFAULT_GROUPED_OUTPUT_PATH}")
        print("===========================================================")
        print(
            "\nNote: A single-day sample captures intra-day overpasses (NOAA-20 and NOAA-21) "
            "but cannot establish multi-day persistence. A multi-day archive is required to observe "
            "temporal recurrence across consecutive days."
        )

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during temporal analysis: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
