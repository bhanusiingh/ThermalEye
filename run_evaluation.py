"""Isolated evaluation runner for ThermalEye."""
import argparse
import pandas as pd
from pathlib import Path
from thermaleye.evaluation.dataset import validate_golden_dataset
from thermaleye.evaluation.metrics import evaluate_baselines

def main():
    parser = argparse.ArgumentParser(description="Evaluate ThermalEye against a golden dataset.")
    parser.add_argument("--golden-csv", type=Path, default=Path("data/samples/golden_fires_india_template.csv"))
    parser.add_argument("--fusion-csv", type=Path, default=Path("data/processed/prioritized_events_v2.csv"))
    parser.add_argument("--k", type=int, default=10, help="K for Precision@K")
    parser.add_argument("--spatial-tol-km", type=float, default=2.0, help="Spatial tolerance for matching in km")
    parser.add_argument("--time-tol-hours", type=float, default=24.0, help="Time tolerance for matching in hours")
    args = parser.parse_args()

    print("=== ThermalEye Phase 15: Isolated Baseline Evaluation ===")
    
    if not args.golden_csv.exists():
        print(f"Error: Golden dataset not found at {args.golden_csv}")
        return
        
    golden_df = pd.read_csv(args.golden_csv)
    
    # Empty template check
    if golden_df.empty:
        print("\nNotice: The provided golden dataset is empty.")
        print("This is expected if manual curation has not yet occurred.")
        print("Run aborted safely. No production files were modified.")
        return
        
    validate_golden_dataset(golden_df)
    
    if not args.fusion_csv.exists():
        print(f"Error: Fusion output not found at {args.fusion_csv}")
        return
        
    fusion_df = pd.read_csv(args.fusion_csv)
    
    from thermaleye.evaluation.matching import match_incidents
    matched_golden_df = match_incidents(golden_df, fusion_df, args.spatial_tol_km, args.time_tol_hours)
    
    # Audit log of matches
    print(f"\n--- Matching Audit (Tol: {args.spatial_tol_km}km, {args.time_tol_hours}h) ---")
    status_counts = matched_golden_df['match_status'].value_counts()
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
        
    # Explicit exclusions and eligibility
    target_class = "confirmed_industrial_fire"
    total_golden = len(golden_df)
    total_target_count = (golden_df['ground_truth_class'] == target_class).sum()
    
    unmatched = matched_golden_df[matched_golden_df['match_status'] == 'Unmatched']
    ambiguous = matched_golden_df[matched_golden_df['match_status'] == 'Ambiguous']
    conflicts = matched_golden_df[matched_golden_df['match_status'] == 'Conflict']
    inconclusive = matched_golden_df[matched_golden_df['match_status'] == 'Inconclusive']
    
    # Exclude all unverified or conflicting matches from the target denominator
    # Unmatched is excluded because observation adequacy cannot be independently established (might be cloud cover)
    eligible_golden = matched_golden_df[matched_golden_df['match_status'] == 'Matched']
    eligible_target_count = (eligible_golden['ground_truth_class'] == target_class).sum()
    
    # We do not have independent proof of observation adequacy for unmatched incidents (e.g. no cloud cover logs)
    # Therefore, we cannot confidently assert end-to-end recall.
    observation_adequacy_known = False
    
    print(f"\n--- Metrics Denominators ---")
    print(f"  Total Curated Incidents: {total_golden}")
    print(f"  Total Target Class ('{target_class}'): {total_target_count}")
    print(f"  Excluded (Unverified/Unmatched): {len(unmatched)}")
    print(f"  Excluded (Inconclusive/No Detections): {len(inconclusive)}")
    print(f"  Excluded (Ambiguous Candidates): {len(ambiguous)}")
    print(f"  Excluded (Incident Conflicts): {len(conflicts)}")
    print(f"  End-to-End Target Class Denominator: {total_target_count}")
    print(f"  Match-Conditional Target Class Denominator: {eligible_target_count}")
    
    valid_matches = matched_golden_df[matched_golden_df['matched_event_id'].notna()]
    merged = pd.merge(valid_matches, fusion_df, left_on='matched_event_id', right_on='event_id', how='left')
    
    print(f"\nEligible Matched Samples: {len(merged)}")
    if len(merged) < args.k:
        print(f"Warning: Merged sample size ({len(merged)}) is less than K={args.k}.")
        print("Reporting results as a preliminary smoke test, not a statistically significant performance claim.")
    class_dist = golden_df['ground_truth_class'].value_counts().to_dict()
    print(f"\nClass Distribution: {class_dist}")
    
    results = evaluate_baselines(
        merged, 
        conditional_denominator=eligible_target_count, 
        end_to_end_denominator=total_target_count,
        observation_adequacy_known=observation_adequacy_known,
        target_class=target_class, 
        k=args.k
    )
    
    print("\n--- Baseline Results ---")
    for baseline, metrics in results.items():
        print(f"\n[{baseline}]")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: {value}")
            
    print("\nNote: Missing evidence (e.g. S2 CV imagery) was preserved as missing in the fusion output.")
    
if __name__ == "__main__":
    main()
