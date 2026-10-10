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
        "sample_id": ["A", "B", "C", "D"],
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
    assert calculate_precision_at_k(df, "confirmed_industrial_fire", k=0) == "N/A"
    assert calculate_precision_at_k(df, "confirmed_industrial_fire", k=-1) == "N/A"
