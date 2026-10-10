"""CLI script to run evidence fusion (Phase 6)."""
import argparse
from pathlib import Path
import pandas as pd
import json

from thermaleye.fusion.scoring import fuse_evidence
from thermaleye.cv.classifier import load_model, predict_image
DEFAULT_EVENTS_OUTPUT_PATH = Path("data/processed/thermal_events.csv")
DEFAULT_OSM_OUTPUT_PATH = Path("data/processed/thermal_events_osm.csv")
DEFAULT_S2_METADATA_PATH = Path("data/processed/sentinel2_metadata.csv")
DEFAULT_MODEL_DIR = Path("models/eurosat_cv")

def main():
    parser = argparse.ArgumentParser(description="Run evidence fusion baseline")
    parser.add_argument("--events-csv", type=Path, default=DEFAULT_EVENTS_OUTPUT_PATH)
    parser.add_argument("--osm-csv", type=Path, default=DEFAULT_OSM_OUTPUT_PATH)
    parser.add_argument("--s2-metadata", type=Path, default=DEFAULT_S2_METADATA_PATH)
    parser.add_argument("--output", type=Path, default=Path("data/processed/prioritized_events.csv"))
    args = parser.parse_args()
    
    print("Initiating ThermalEye Evidence Fusion (Phase 6)...")
    
    if not args.events_csv.exists():
        print(f"Error: Events file {args.events_csv} missing.")
        return
        
    events_df = pd.read_csv(args.events_csv)
    print(f"Loaded {len(events_df)} candidate thermal events.")
    
    osm_df = pd.read_csv(args.osm_csv) if args.osm_csv.exists() else None
    if osm_df is not None:
        print(f"Loaded OSM context for {len(osm_df)} events.")
        
    # Process Sentinel-2 CV metadata
    cv_predictions = {}
    if args.s2_metadata.exists():
        s2_meta = pd.read_csv(args.s2_metadata)
        
        # Load CV model if metadata exists
        print("Loading EfficientNet-B0 CV model for available patches...")
        try:
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            weights_path = DEFAULT_MODEL_DIR / "efficientnet_b0_eurosat.pth"
            config_path = DEFAULT_MODEL_DIR / "model_config.json"
            model, classes = load_model(weights_path, config_path, device)
            
            from thermaleye.config import DEFAULT_S2_DATE_RANGE_DAYS, DEFAULT_S2_PATCH_HALF_DEG, DEFAULT_S2_PATCH_WIDTH_PX, DEFAULT_S2_PATCH_HEIGHT_PX
            from thermaleye.imagery.sentinel2 import generate_image_fingerprint
            
            event_coords = events_df.set_index('event_id')[['centroid_lat', 'centroid_lon']].to_dict('index')

            for _, row in s2_meta.iterrows():
                event_id = row.get("event_id")
                
                # Fingerprint verification
                if event_id not in event_coords:
                    continue
                
                expected_fp = generate_image_fingerprint(
                    event_coords[event_id]['centroid_lat'],
                    event_coords[event_id]['centroid_lon'],
                    DEFAULT_S2_DATE_RANGE_DAYS,
                    DEFAULT_S2_PATCH_HALF_DEG,
                    DEFAULT_S2_PATCH_WIDTH_PX,
                    DEFAULT_S2_PATCH_HEIGHT_PX
                )
                actual_fp = row.get("image_fingerprint")
                expected_strict_fp = expected_fp + "_STRICT"
                if pd.isna(actual_fp) or actual_fp not in (expected_fp, expected_strict_fp):
                    # Stale cache or legacy entry without verifiable provenance
                    continue
                    
                local_path_str = str(row.get("output_path", ""))
                if not local_path_str or local_path_str == "nan":
                    continue
                patch_path = Path(local_path_str)
                if patch_path.is_file():
                    preds = predict_image(patch_path, model, classes, device)
                    cv_predictions[event_id] = preds
                    
            print(f"Generated CV scores for {len(cv_predictions)} available images.")
        except Exception as e:
            print(f"Warning: CV model could not be initialized or failed. Skipping CV scores. {e}")
            
    # Run fusion
    print("\nExecuting priority scoring...")
    try:
        fused_df = fuse_evidence(events_df, osm_df, cv_predictions)
    except Exception as e:
        print(f"Fusion failed: {e}")
        return
        
    if fused_df.empty:
        print("No events remaining after fusion (all dropped by boundary filter or empty input).")
        return
        
    fused_df.to_csv(args.output, index=False)
    
    high = len(fused_df[fused_df['priority'] == 'High'])
    med = len(fused_df[fused_df['priority'] == 'Medium'])
    low = len(fused_df[fused_df['priority'] == 'Low'])
    insuf = len(fused_df[fused_df['priority'] == 'Insufficient evidence'])
    missing_cv = sum(pd.isna(fused_df['score_cv']))
    missing_osm = sum(pd.isna(fused_df['score_osm']))
    
    print("\n================ FUSION SUMMARY ================")
    print(f"Events Scored        : {len(fused_df)}")
    print(f"High Priority        : {high}")
    print(f"Medium Priority      : {med}")
    print(f"Low Priority         : {low}")
    print(f"Insufficient Evid.   : {insuf}")
    print(f"Missing OSM Data     : {missing_osm}")
    print(f"Missing S2 Imagery   : {missing_cv}")
    print(f"Output File          : {args.output}")
    print("================================================")
    
    print("\nDisclaimer: This system prioritizes events for human review; it does not yet prove industrial-fire detection accuracy.")

if __name__ == "__main__":
    main()
