"""Root CLI for Phase 4: Sentinel-2 image patch retrieval."""

import argparse
import csv
from pathlib import Path
import sys
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from thermaleye.config import (
    DEFAULT_S2_DATE_RANGE_DAYS,
    DEFAULT_S2_MAX_CLOUD_PCT,
    DEFAULT_S2_OUTPUT_DIR,
    DEFAULT_S2_META_PATH,
    DEFAULT_S2_PATCH_WIDTH_PX,
    DEFAULT_S2_PATCH_HEIGHT_PX,
)
from thermaleye.imagery.sentinel2 import (
    CDSEClient,
    SentinelHubConfigError,
    retrieve_sentinel2_patch,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ThermalEye Phase 4: Sentinel-2 L2A Image Patch Retrieval"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/thermal_events.csv"),
        help="Path to thermal events CSV (default: data/processed/thermal_events.csv)",
    )
    parser.add_argument(
        "--event-id",
        type=str,
        default=None,
        help="Process a specific event by ID (e.g., TE-EVT-00001)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Maximum events to process when --event-id is not given (default: 1)",
    )
    parser.add_argument(
        "--date-range-days",
        type=int,
        default=DEFAULT_S2_DATE_RANGE_DAYS,
        help=f"Days before event to search for Sentinel-2 scenes (default: {DEFAULT_S2_DATE_RANGE_DAYS})",
    )
    parser.add_argument(
        "--max-cloud-pct",
        type=float,
        default=DEFAULT_S2_MAX_CLOUD_PCT,
        help=f"Maximum cloud coverage %% to accept a scene (default: {DEFAULT_S2_MAX_CLOUD_PCT})",
    )
    parser.add_argument(
        "--strict-eval-mode",
        action="store_true",
        help="If set, strictly restrict Sentinel-2 search window to end ON OR BEFORE the event detection time.",
    )
    args = parser.parse_args()

    print("Initiating ThermalEye Sentinel-2 Retrieval (Phase 4)...")
    if args.strict_eval_mode:
        print("STRICT EVALUATION MODE ENABLED: Post-incident imagery will be excluded.")

    if not args.input.exists():
        print(f"Error: Input events file not found: {args.input}")
        sys.exit(1)

    try:
        client = CDSEClient()
    except SentinelHubConfigError as e:
        print(f"\nConfiguration Error: {e}")
        print("Add CDSE_CLIENT_ID and CDSE_CLIENT_SECRET to your .env file.")
        sys.exit(1)

    events_df = pd.read_csv(args.input)

    if args.event_id:
        mask = events_df["event_id"] == args.event_id
        target_df = events_df[mask]
        if target_df.empty:
            print(f"Error: Event ID '{args.event_id}' not found in {args.input}")
            sys.exit(1)
    else:
        target_df = events_df.head(args.limit)

    results = []
    for _, row in target_df.iterrows():
        event_id = row["event_id"]
        lat = float(row["centroid_lat"])
        lon = float(row["centroid_lon"])
        event_date = str(row.get("last_detected", row.get("first_detected", "")))[:10]

        print(f"  Querying: {event_id} | lat={lat}, lon={lon} | date={event_date}")
        result = retrieve_sentinel2_patch(
            event_id=event_id,
            centroid_lat=lat,
            centroid_lon=lon,
            event_date=event_date,
            client=client,
            output_dir=DEFAULT_S2_OUTPUT_DIR,
            date_range_days=args.date_range_days,
            max_cloud_pct=args.max_cloud_pct,
            width_px=DEFAULT_S2_PATCH_WIDTH_PX,
            height_px=DEFAULT_S2_PATCH_HEIGHT_PX,
            strict_eval_mode=args.strict_eval_mode,
        )
        results.append(result)

        if result.success:
            print(f"    [OK] Saved: {result.output_path}")
            print(f"      Scene: {result.scene_id}")
            print(f"      Acquired: {result.acquisition_time}")
            if result.cloud_cover_pct is not None:
                print(f"      Cloud cover: {result.cloud_cover_pct:.1f}%")
        else:
            print(f"    [FAIL] Failed: {result.error_message}")

    # Save metadata
    if results:
        DEFAULT_S2_META_PATH.parent.mkdir(parents=True, exist_ok=True)
        new_df = pd.DataFrame([r.to_metadata_dict() for r in results])
        if DEFAULT_S2_META_PATH.exists():
            old_df = pd.read_csv(DEFAULT_S2_META_PATH)
            # Remove old entries for the same event IDs to avoid duplicates
            old_df = old_df[~old_df['event_id'].isin(new_df['event_id'])]
            final_df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            final_df = new_df
        final_df.to_csv(DEFAULT_S2_META_PATH, index=False)

    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful

    print(f"\n================ S2 RETRIEVAL SUMMARY ================")
    print(f"Events Attempted     : {len(results)}")
    print(f"Successful Patches   : {successful}")
    print(f"Failed / No Data     : {failed}")
    print(f"Output Directory     : {DEFAULT_S2_OUTPUT_DIR}")
    print(f"Metadata Log         : {DEFAULT_S2_META_PATH}")
    print(f"======================================================")
    if failed > 0:
        print(
            "\nNote: Missing or cloudy imagery is treated as unavailable evidence, "
            "not as a negative classification of the thermal event."
        )


if __name__ == "__main__":
    main()
