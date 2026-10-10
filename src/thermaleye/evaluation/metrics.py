from typing import Union, Any, Dict
"""Metrics calculation for ThermalEye baseline evaluation."""
import pandas as pd
import numpy as np

def calculate_precision_at_k(df: pd.DataFrame, target_class: str, k: int, ground_truth_completeness_known: bool = False) -> Union[float, str]:
    """
    Calculate Precision@K for a ranked dataframe.

    Denominator convention: The denominator is strictly `k`, even if the candidate pool
    size `len(df)` is smaller than `k`. This ensures precision correctly reflects the
    fixed-size output budget of K.
    """
    if k <= 0 or df.empty:
        return "N/A"
    if not ground_truth_completeness_known:
        return "Unverified (Unknown Ground-Truth Completeness)"

    top_k = df.head(k)
    true_positives = (top_k['ground_truth_class'] == target_class).sum()
    return float(true_positives) / k

def calculate_recall_at_threshold(df: pd.DataFrame, target_class: str, priority_thresholds: list, total_true: int) -> Union[float, str]:
    """Calculate Recall for events matching the given priority thresholds."""
    if total_true <= 0:
        return "N/A"
    if df.empty:
        return 0.0

    retrieved = df[df['priority'].isin(priority_thresholds)]
    retrieved_true = (retrieved['ground_truth_class'] == target_class).sum()
    return float(retrieved_true) / total_true

def evaluate_baselines(merged_df: pd.DataFrame,
                       conditional_denominator: int,
                       end_to_end_denominator: int,
                       observation_adequacy_known: bool,
                       ground_truth_completeness_known: bool = False,
                       target_class: str = "confirmed_industrial_fire",
                       k: int = 10) -> dict:
    """Evaluate Random, FRP-only, and Fusion baselines."""
    results = {}

    def _build_metrics(df_ranked: pd.DataFrame) -> dict:
        m = {
            'Precision@K': calculate_precision_at_k(df_ranked, target_class, k, ground_truth_completeness_known),
            'Conditional_Recall_Medium_High': calculate_recall_at_threshold(df_ranked, target_class, ['High', 'Medium'], conditional_denominator)
        }
        if observation_adequacy_known:
            m['End_to_End_Recall_Medium_High'] = calculate_recall_at_threshold(df_ranked, target_class, ['High', 'Medium'], end_to_end_denominator)
        else:
            m['End_to_End_Recall_Medium_High'] = "Unverified (Unknown Observation Adequacy)"
        return m

    # 1. Random Baseline (Seed 42)
    df_random = merged_df.sample(frac=1, random_state=42) if not merged_df.empty else merged_df.copy()
    results['Random'] = _build_metrics(df_random)

    # 2. FRP-Only Baseline (Descending max_frp, falling back to mean_frp if needed)
    if not merged_df.empty:
        frp_col = 'max_frp' if 'max_frp' in merged_df.columns else 'mean_frp'
        # Secondary sort by event_id ensures deterministic ranking for tied scores
        df_frp = merged_df.sort_values(by=[frp_col, 'event_id'], ascending=[False, True], na_position='last')
    else:
        df_frp = merged_df.copy()

    results['FRP_Only'] = _build_metrics(df_frp)

    # 3. Fusion Baseline
    if not merged_df.empty:
        # Secondary sort by event_id ensures deterministic ranking for tied scores
        df_fusion = merged_df.sort_values(by=['priority_score', 'event_id'], ascending=[False, True], na_position='last')
    else:
        df_fusion = merged_df.copy()

    results['ThermalEye_Fusion'] = _build_metrics(df_fusion)

    return results
