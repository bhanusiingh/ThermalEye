"""Tests for dataset validation and metrics calculation."""
import pytest
import pandas as pd
from thermaleye.evaluation.dataset import validate_golden_dataset
from thermaleye.evaluation.metrics import evaluate_baselines

def test_validate_golden_dataset_empty():
    df = pd.DataFrame()
    assert validate_golden_dataset(df) is True

def test_validate_golden_dataset_invalid_columns():
    df = pd.DataFrame({"wrong_col": [1]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_golden_dataset(df)

def test_validate_golden_dataset_invalid_class():
    df = pd.DataFrame({
        "sample_id": ["1"],
        "latitude": [10.0],
        "longitude": [20.0],
        "event_date": ["2026-10-10"],
        "ground_truth_class": ["fake_class"],
        "source_url": [""],
        "source_date": [""],
        "verification_notes": [""]
    })
    with pytest.raises(ValueError, match="Invalid ground_truth_class"):
        validate_golden_dataset(df)

def test_evaluate_baselines():
    # Synthetic dataset
    df = pd.DataFrame({
        "event_id": ["A", "B", "C", "D"],
        "ground_truth_class": [
            "confirmed_industrial_fire",
            "vegetation_fire",
            "confirmed_industrial_fire",
            "routine_flaring"
        ],
        "priority_score": [0.9, 0.2, 0.8, 0.6],  # Fusion score
        "mean_frp": [10.0, 50.0, 5.0, 100.0],     # FRP score
        "priority": ["High", "Low", "Medium", "Medium"]
    })

    # Total confirmed industrial fires is 2
    total_target = 2
    res = evaluate_baselines(
        df,
        conditional_denominator=total_target,
        end_to_end_denominator=total_target,
        observation_adequacy_known=True,
        ground_truth_completeness_known=True,
        target_class="confirmed_industrial_fire",
        k=2
    )

    # FRP-only should rank D(100), B(50) -> Top 2 are routine_flaring and vegetation_fire
    # Precision@2 = 0.0
    assert res["FRP_Only"]["Precision@K"] == 0.0

    # Fusion should rank A(0.9), C(0.8) -> Top 2 are both confirmed_industrial_fire
    # Precision@2 = 1.0
    assert res["ThermalEye_Fusion"]["Precision@K"] == 1.0

    # Recall for Fusion: True positives (A, C) are High and Medium. Total true = 2. Retrieved true = 2.
    assert res["ThermalEye_Fusion"]["Conditional_Recall_Medium_High"] == 1.0


def test_conditional_vs_end_to_end_recall():
    """Test how unmatched incidents affect end-to-end vs conditional recall."""
    from thermaleye.evaluation.metrics import evaluate_baselines
    import pandas as pd

    # 4 Total confirmed fires
    # 2 are successfully matched (eligible) and retrieved in top priorities.
    # 2 are unmatched (unknown observation adequacy, so not in the dataframe).
    df = pd.DataFrame({
        "event_id": ["E1", "E2"],
        "ground_truth_class": ["confirmed_industrial_fire", "confirmed_industrial_fire"],
        "priority_score": [0.9, 0.8],
        "priority": ["High", "Medium"],
        "mean_frp": [10.0, 5.0]
    })

    # If observation adequacy is known (e.g. we know there were no clouds, so unmatched are true misses)
    res_known = evaluate_baselines(
        df,
        conditional_denominator=2, # Only the 2 eligible/matched
        end_to_end_denominator=4, # All 4 curated fires
        observation_adequacy_known=True,
        ground_truth_completeness_known=True,
        target_class="confirmed_industrial_fire",
        k=2
    )

    # Conditional recall: 2 retrieved / 2 eligible = 1.0
    assert res_known["ThermalEye_Fusion"]["Conditional_Recall_Medium_High"] == 1.0

    # End-to-end recall: 2 retrieved / 4 total = 0.5
    assert res_known["ThermalEye_Fusion"]["End_to_End_Recall_Medium_High"] == 0.5

    # If observation adequacy is UNKNOWN (e.g. we don't know if clouds caused the misses)
    res_unknown = evaluate_baselines(
        df,
        conditional_denominator=2,
        end_to_end_denominator=4,
        observation_adequacy_known=False,
        ground_truth_completeness_known=True,
        target_class="confirmed_industrial_fire",
        k=2
    )

    # Conditional recall remains 1.0
    assert res_unknown["ThermalEye_Fusion"]["Conditional_Recall_Medium_High"] == 1.0

    # End-to-end recall is unverified
    assert res_unknown["ThermalEye_Fusion"]["End_to_End_Recall_Medium_High"] == "Unverified (Unknown Observation Adequacy)"

def test_recall_zero_eligible_targets():
    from thermaleye.evaluation.metrics import evaluate_baselines
    import pandas as pd

    # 0 eligible targets (e.g. all unmatched and known observation adequacy)
    df = pd.DataFrame(columns=["event_id", "ground_truth_class", "priority_score", "priority", "mean_frp"])

    res = evaluate_baselines(
        df,
        conditional_denominator=0,
        end_to_end_denominator=2, # 2 curated but 0 matched
        observation_adequacy_known=True,
        ground_truth_completeness_known=True,
        target_class="confirmed_industrial_fire",
        k=2
    )
    # When denominator is 0, recall should gracefully return 0.0 or be handled without ZeroDivisionError
    assert res["ThermalEye_Fusion"]["Conditional_Recall_Medium_High"] == "N/A"
    assert res["ThermalEye_Fusion"]["End_to_End_Recall_Medium_High"] == 0.0

def test_recall_unknown_adequacy_unverified():
    from thermaleye.evaluation.metrics import evaluate_baselines
    import pandas as pd

    df = pd.DataFrame({
        "event_id": ["E1"],
        "ground_truth_class": ["confirmed_industrial_fire"],
        "priority_score": [0.9],
        "priority": ["High"],
        "mean_frp": [10.0]
    })

    res = evaluate_baselines(
        df,
        conditional_denominator=1,
        end_to_end_denominator=2,
        observation_adequacy_known=False,
        ground_truth_completeness_known=True,
        target_class="confirmed_industrial_fire",
        k=2
    )
    assert res["ThermalEye_Fusion"]["Conditional_Recall_Medium_High"] == 1.0
    assert res["ThermalEye_Fusion"]["End_to_End_Recall_Medium_High"] == "Unverified (Unknown Observation Adequacy)"

def test_recall_genuine_zero():
    from thermaleye.evaluation.metrics import evaluate_baselines
    import pandas as pd

    # 2 eligible targets, but NONE were retrieved in High or Medium (they are Low)
    df = pd.DataFrame({
        "event_id": ["E1", "E2"],
        "ground_truth_class": ["confirmed_industrial_fire", "confirmed_industrial_fire"],
        "priority_score": [0.1, 0.2],
        "priority": ["Low", "Low"],
        "mean_frp": [1.0, 2.0]
    })

    res = evaluate_baselines(
        df,
        conditional_denominator=2,
        end_to_end_denominator=2,
        observation_adequacy_known=True,
        ground_truth_completeness_known=True,
        target_class="confirmed_industrial_fire",
        k=2
    )
    # Genuine zero because denominator is > 0 but retrieved target count is 0
    assert res["ThermalEye_Fusion"]["Conditional_Recall_Medium_High"] == 0.0
    assert res["ThermalEye_Fusion"]["End_to_End_Recall_Medium_High"] == 0.0

def test_metrics_empty_target_set():
    """When target set is empty (denominator=0), recall is undefined ('N/A')."""
    from thermaleye.evaluation.metrics import evaluate_baselines
    import pandas as pd

    df = pd.DataFrame({
        "event_id": ["E1", "E2"],
        "ground_truth_class": ["vegetation_fire", "routine_flaring"],
        "priority_score": [0.9, 0.8],
        "priority": ["High", "Medium"],
        "mean_frp": [10.0, 5.0]
    })

    res = evaluate_baselines(
        df,
        conditional_denominator=0,
        end_to_end_denominator=0,
        observation_adequacy_known=True,
        ground_truth_completeness_known=True,
        target_class="confirmed_industrial_fire",
        k=2
    )
    assert res["ThermalEye_Fusion"]["Conditional_Recall_Medium_High"] == "N/A"
    assert res["ThermalEye_Fusion"]["End_to_End_Recall_Medium_High"] == "N/A"

def test_metrics_empty_candidate_pool():
    """When candidate pool is empty, Precision@K is undefined ('N/A'), and Recall is 0.0 if targets exist."""
    from thermaleye.evaluation.metrics import evaluate_baselines
    import pandas as pd

    df = pd.DataFrame(columns=["event_id", "ground_truth_class", "priority_score", "priority", "mean_frp"])

    res = evaluate_baselines(
        df,
        conditional_denominator=3,
        end_to_end_denominator=3,
        observation_adequacy_known=True,
        ground_truth_completeness_known=True,
        target_class="confirmed_industrial_fire",
        k=2
    )
    assert res["ThermalEye_Fusion"]["Precision@K"] == "N/A"
    assert res["ThermalEye_Fusion"]["Conditional_Recall_Medium_High"] == 0.0
    assert res["ThermalEye_Fusion"]["End_to_End_Recall_Medium_High"] == 0.0

def test_metrics_valid_non_empty_pool_no_retrievals():
    """When candidates exist but none match the target class, precision and recall are genuine measured 0.0."""
    from thermaleye.evaluation.metrics import evaluate_baselines, calculate_precision_at_k
    import pandas as pd

    df = pd.DataFrame({
        "event_id": ["E1", "E2"],
        "ground_truth_class": ["vegetation_fire", "routine_flaring"],
        "priority_score": [0.9, 0.8],
        "priority": ["High", "Medium"],
        "mean_frp": [10.0, 5.0]
    })

    res = evaluate_baselines(
        df,
        conditional_denominator=2,
        end_to_end_denominator=2,
        observation_adequacy_known=True,
        ground_truth_completeness_known=True,
        target_class="confirmed_industrial_fire",
        k=2
    )
    assert res["ThermalEye_Fusion"]["Precision@K"] == 0.0
    assert res["ThermalEye_Fusion"]["Conditional_Recall_Medium_High"] == 0.0
    assert res["ThermalEye_Fusion"]["End_to_End_Recall_Medium_High"] == 0.0

def test_precision_at_k_zero_or_negative_k():
    """When k <= 0, Precision@K denominator is 0, so it must return 'N/A'."""
    from thermaleye.evaluation.metrics import calculate_precision_at_k
    import pandas as pd

    df = pd.DataFrame({
        "ground_truth_class": ["confirmed_industrial_fire"]
    })
    assert calculate_precision_at_k(df, "confirmed_industrial_fire", k=0, ground_truth_completeness_known=True) == "N/A"
    assert calculate_precision_at_k(df, "confirmed_industrial_fire", k=-1, ground_truth_completeness_known=True) == "N/A"

def test_precision_unknown_ground_truth_completeness():
    """Precision is reported as Unverified if ground-truth completeness is unknown."""
    from thermaleye.evaluation.metrics import calculate_precision_at_k
    import pandas as pd

    df = pd.DataFrame({
        "ground_truth_class": ["confirmed_industrial_fire"]
    })
    # Passing valid K, but completeness is False (default)
    res = calculate_precision_at_k(df, "confirmed_industrial_fire", k=1)
    assert res == "Unverified (Unknown Ground-Truth Completeness)"

def test_candidate_pool_merge():
    """Verify that the left join candidate pool encompasses all pipeline events without duplication."""
    import pandas as pd

    # Simulate fusion_df (all pipeline events)
    fusion_df = pd.DataFrame({
        'event_id': ['EVT-1', 'EVT-2', 'EVT-3', 'EVT-4'],
        'priority_score': [0.9, 0.8, 0.7, 0.6]
    })

    # Simulate valid_matches (golden incidents successfully matched)
    # EVT-1 matched a single target.
    # EVT-2 matched a non-target.
    # EVT-3 and EVT-4 are unmatched pipeline events.
    # We also simulate an Ambiguous or Conflict golden incident which would NOT appear in valid_matches.
    valid_matches = pd.DataFrame({
        'matched_event_id': ['EVT-1', 'EVT-2'],
        'ground_truth_class': ['confirmed_industrial_fire', 'vegetation_fire']
    })

    # The evaluation candidate pool must encompass all events in fusion_df, not just matches.
    merged = pd.merge(
        fusion_df,
        valid_matches[['matched_event_id', 'ground_truth_class']],
        left_on='event_id',
        right_on='matched_event_id',
        how='left'
    )

    merged['ground_truth_class'] = merged['ground_truth_class'].fillna('uncurated_detection')

    # Assert exact event counts
    assert len(merged) == 4, "No events should be dropped or duplicated"

    # Assert candidate IDs
    assert set(merged['event_id']) == {'EVT-1', 'EVT-2', 'EVT-3', 'EVT-4'}

    # Assert labels
    evt1_label = merged[merged['event_id'] == 'EVT-1'].iloc[0]['ground_truth_class']
    assert evt1_label == 'confirmed_industrial_fire'

    evt3_label = merged[merged['event_id'] == 'EVT-3'].iloc[0]['ground_truth_class']
    assert evt3_label == 'uncurated_detection', "Unmatched events should be 'uncurated_detection'"

def test_random_baseline_reproducibility():
    from thermaleye.evaluation.metrics import evaluate_baselines
    import pandas as pd

    df = pd.DataFrame({
        "event_id": [f"E{i}" for i in range(100)],
        "ground_truth_class": ["confirmed_industrial_fire"] * 50 + ["uncurated_detection"] * 50,
        "priority": ["Low"] * 100,
        "priority_score": [0.1] * 100,
        "mean_frp": [1.0] * 100
    })

    res1 = evaluate_baselines(
        df,
        conditional_denominator=50,
        end_to_end_denominator=50,
        observation_adequacy_known=True,
        ground_truth_completeness_known=True,
        target_class="confirmed_industrial_fire",
        k=10
    )

    res2 = evaluate_baselines(
        df,
        conditional_denominator=50,
        end_to_end_denominator=50,
        observation_adequacy_known=True,
        ground_truth_completeness_known=True,
        target_class="confirmed_industrial_fire",
        k=10
    )

    assert res1["Random"]["Precision@K"] == res2["Random"]["Precision@K"], "Random baseline must be reproducible via a fixed seed."

def test_top_k_exceeds_pool():
    from thermaleye.evaluation.metrics import calculate_precision_at_k
    import pandas as pd

    # 2 items in pool, K=10
    df = pd.DataFrame({
        "ground_truth_class": ["confirmed_industrial_fire", "uncurated_detection"]
    })
    # Top K will return 2 items. 1 is target. Precision@10 = 1/10 = 0.1
    # This correctly bounds precision if pool size is smaller than K.
    precision = calculate_precision_at_k(df, "confirmed_industrial_fire", k=10, ground_truth_completeness_known=True)
    assert precision == 0.1

def test_recall_denominator_with_ambiguous_and_conflicts():
    """Verify that ambiguous and conflict incidents correctly affect denominators."""
    import pandas as pd
    from thermaleye.evaluation.matching import match_incidents

    # 3 golden incidents
    golden_df = pd.DataFrame({
        "event_id": ["G1", "G2", "G3"],
        "latitude": [10.0, 20.0, 20.0],
        "longitude": [10.0, 20.0, 20.0], # G2 and G3 are identical locations
        "event_date": ["2026-10-10", "2026-10-10", "2026-10-10"],
        "ground_truth_class": ["confirmed_industrial_fire", "confirmed_industrial_fire", "confirmed_industrial_fire"]
    })

    # 2 pipeline events
    events_df = pd.DataFrame({
        "event_id": ["E1", "E2"],
        "centroid_lat": [10.0, 20.0],
        "centroid_lon": [10.0, 20.0],
        "first_detected": ["2026-10-10", "2026-10-10"],
        "last_detected": ["2026-10-10", "2026-10-10"]
    })

    # Match incidents
    matched_df = match_incidents(golden_df, events_df, spatial_tol_km=2.0, time_tol_hours=24.0)

    # G1 matches E1 (1-to-1)
    assert matched_df[matched_df['event_id'] == 'G1'].iloc[0]['match_status'] == 'Matched'

    # G2 and G3 both match E2. Thus, they conflict.
    assert matched_df[matched_df['event_id'] == 'G2'].iloc[0]['match_status'] == 'Conflict'
    assert matched_df[matched_df['event_id'] == 'G3'].iloc[0]['match_status'] == 'Conflict'

    # Denominators
    target_class = "confirmed_industrial_fire"
    total_target_count = (matched_df['ground_truth_class'] == target_class).sum()
    assert total_target_count == 3

    eligible = matched_df[matched_df['match_status'] == 'Matched']
    eligible_target_count = (eligible['ground_truth_class'] == target_class).sum()
    assert eligible_target_count == 1

def test_metric_serialization_types():
    """Verify that unavailable metrics return string reasons, not numbers/NaN, and available return floats."""
    from thermaleye.evaluation.metrics import evaluate_baselines
    import pandas as pd

    df = pd.DataFrame({
        "event_id": ["E1"],
        "ground_truth_class": ["confirmed_industrial_fire"],
        "priority_score": [0.9],
        "priority": ["High"],
        "mean_frp": [10.0]
    })

    # Unknown completeness and known adequacy
    res_unknown_comp = evaluate_baselines(
        df,
        conditional_denominator=1,
        end_to_end_denominator=1,
        observation_adequacy_known=True,
        ground_truth_completeness_known=False,
        target_class="confirmed_industrial_fire",
        k=1
    )
    assert isinstance(res_unknown_comp["ThermalEye_Fusion"]["Precision@K"], str)
    assert isinstance(res_unknown_comp["ThermalEye_Fusion"]["Conditional_Recall_Medium_High"], float)
    assert isinstance(res_unknown_comp["ThermalEye_Fusion"]["End_to_End_Recall_Medium_High"], float)

    # Known completeness and unknown adequacy
    res_unknown_adeq = evaluate_baselines(
        df,
        conditional_denominator=1,
        end_to_end_denominator=1,
        observation_adequacy_known=False,
        ground_truth_completeness_known=True,
        target_class="confirmed_industrial_fire",
        k=1
    )
    assert isinstance(res_unknown_adeq["ThermalEye_Fusion"]["Precision@K"], float)
    assert isinstance(res_unknown_adeq["ThermalEye_Fusion"]["Conditional_Recall_Medium_High"], float)
    assert isinstance(res_unknown_adeq["ThermalEye_Fusion"]["End_to_End_Recall_Medium_High"], str)

def test_ranking_ties_deterministic():
    """Verify that identical scores are broken deterministically by event_id."""
    from thermaleye.evaluation.metrics import evaluate_baselines
    import pandas as pd

    df = pd.DataFrame({
        "event_id": ["E2", "E1", "E3"],
        "ground_truth_class": ["uncurated_detection", "confirmed_industrial_fire", "uncurated_detection"],
        "priority_score": [0.5, 0.5, 0.5], # All tied
        "priority": ["Medium"] * 3,
        "mean_frp": [10.0, 10.0, 10.0]
    })

    res = evaluate_baselines(
        df,
        conditional_denominator=1,
        end_to_end_denominator=1,
        observation_adequacy_known=True,
        ground_truth_completeness_known=True,
        target_class="confirmed_industrial_fire",
        k=1
    )
    # The tie should be broken by event_id ascending: E1, E2, E3.
    # So E1 is ranked first. E1 is a target. Precision@1 = 1.0.
    assert res["ThermalEye_Fusion"]["Precision@K"] == 1.0
    assert res["FRP_Only"]["Precision@K"] == 1.0
