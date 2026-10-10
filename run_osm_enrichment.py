"""Root entry point for executing Phase 3 OpenStreetMap Industrial Context Enrichment."""

import argparse
from pathlib import Path
import sys
import pandas as pd

# Ensure src/ is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from thermaleye.config import (
    DEFAULT_OSM_SEARCH_RADIUS_M,
    DEFAULT_OSM_OUTPUT_PATH,
    DEFAULT_MAX_OSM_EVENTS,
)
from thermaleye.osm.overpass import run_osm_enrichment


def main() -> None:
    parser = argparse.ArgumentParser(description="ThermalEye Phase 3: OSM Industrial Context Enrichment")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/thermal_events.csv"),
        help="Path to candidate thermal events CSV (default: data/processed/thermal_events.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OSM_OUTPUT_PATH,
        help=f"Path to enriched output CSV (default: {DEFAULT_OSM_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=DEFAULT_MAX_OSM_EVENTS,
        help=f"Maximum number of events to query via Overpass (default: {DEFAULT_MAX_OSM_EVENTS})",
    )
    parser.add_argument(
        "--radius-m",
        type=int,
        default=DEFAULT_OSM_SEARCH_RADIUS_M,
        help=f"Search radius in meters around event centroid (default: {DEFAULT_OSM_SEARCH_RADIUS_M}m)",
    )
    args = parser.parse_args()

    print("Initiating ThermalEye OSM Industrial Context Enrichment (Phase 3)...")
    print(f"Input file           : {args.input}")
    print(f"Output destination   : {args.output}")
    print(f"Search radius        : {args.radius_m} m")
    print(f"Max events to query  : {args.max_events}")

    try:
        enriched_df = run_osm_enrichment(
            input_events_path=args.input,
            output_events_path=args.output,
            max_events=args.max_events,
            radius_m=args.radius_m,
        )

        total_events = len(enriched_df)
        queried_events = min(total_events, args.max_events)
        skipped_events = max(0, total_events - queried_events)
        available_osm = int((enriched_df["osm_data_available"] == True).sum())
        ind_series = pd.to_numeric(enriched_df["industrial_features_1km"], errors="coerce").fillna(0)
        with_ind_features = int((ind_series > 0).sum())
        with_power_plant = int((enriched_df["nearest_power_plant_km"].notna()).sum())

        print("\n================ OSM ENRICHMENT SUMMARY ================")
        print(f"Total Events         : {total_events}")
        print(f"Events Processed     : {queried_events}")
        print(f"Events Skipped       : {skipped_events}")
        print(f"Successful OSM Data  : {available_osm}")
        print(f"Events w/ Ind Features : {with_ind_features} (within 1 km)")
        print(f"Events w/ Power Plants : {with_power_plant}")
        print(f"Output File          : {args.output}")
        print("========================================================")
        print(
            "\nNote: Proximity to industrial infrastructure provides spatial context, "
            "but does NOT prove that a thermal detection is an industrial fire. "
            "Downstream verification with Sentinel-2 optical imagery is required."
        )

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during OSM enrichment: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
