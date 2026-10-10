"""Temporal persistence score calculation and transparent heuristic definition.

Formula & Methodology Documentation:
-------------------------------------
The persistence score is an uncalibrated heuristic index in [0.0, 1.0] indicating
the degree of temporal recurrence of a candidate thermal anomaly.

Rationale:
Industrial sources (e.g., steel furnaces, petrochemical flare stacks, coal-fired
power plants, brick kilns) typically exhibit thermal anomalies across multiple
satellite overpasses and consecutive observation days. In contrast, ephemeral
events (agricultural stubble burns, forest brush fires, transient bonfires)
usually appear on a single pass or burn out within a few hours.

Observable Temporal Inputs:
1. active_days (D_active): Number of distinct calendar days with detections.
2. detection_count (N): Total number of satellite detections associated with the event.
3. duration_days (T_span): Elapsed span between first and last detection.

Mathematical Formulation:
1. Multi-Day Persistence Component (S_days):
   For single-day observations (D_active <= 1), multi-day persistence is unproven:
       S_days = 0.0
   For multi-day observations, S_days scales linearly up to a reference window
   of D_ref = 5 days:
       S_days = min(1.0, (D_active - 1) / (D_ref - 1))   (where D_ref = 5)

2. Repeat Observation Frequency Component (S_freq):
   Measures intra-day or inter-pass recurrence up to N_ref = 10 detections:
       S_freq = 0.0 if N <= 1 else min(1.0, (N - 1) / (N_ref - 1))

3. Combined Persistence Score:
       persistence_score = (0.7 * S_days) + (0.3 * S_freq)
   The result is strictly clamped to [0.0, 1.0] and rounded to 3 decimal places.

Known Limitations:
------------------
1. Observation Window Dependency: Single-day inputs (such as a 24-hour sample)
   cannot establish multi-day persistence by definition; the score will reflect
   only intra-day recurrence (max 0.3 * S_freq).
2. Uncalibrated Heuristic: The weights (0.7 for active days, 0.3 for detection
   count) are engineering priors, NOT calibrated physical probabilities.
3. Cloud Occlusion / Orbital Gaps: A truly persistent industrial source may miss
   detections due to thick cloud cover or orbital swathing gaps.
4. Non-Industrial Multi-Day Wildfires: Large wildfires or prolonged peat fires
   can burn across multiple days and achieve high scores; persistence alone
   does not confirm an industrial facility without multi-modal verification
   (e.g., OpenStreetMap land-use and Sentinel-2 optical verification).
"""

from typing import Union


def calculate_persistence_score(
    active_days: int,
    detection_count: int,
    ref_active_days: int = 5,
    ref_detection_count: int = 10,
    days_weight: float = 0.7,
    freq_weight: float = 0.3,
) -> float:
    """Calculate transparent heuristic persistence score in [0.0, 1.0].

    Args:
        active_days: Number of distinct acquisition dates for the event.
        detection_count: Total detection count in the event.
        ref_active_days: Reference benchmark for multi-day persistence (default: 5).
        ref_detection_count: Reference benchmark for high recurrence count (default: 10).
        days_weight: Relative weight for multi-day recurrence (default: 0.7).
        freq_weight: Relative weight for detection frequency (default: 0.3).

    Returns:
        float score in [0.0, 1.0] rounded to 3 decimals.
    """
    if active_days <= 0 or detection_count <= 0:
        return 0.0

    # Multi-day component: 0 if active_days <= 1
    if active_days <= 1 or ref_active_days <= 1:
        s_days = 0.0
    else:
        s_days = min(1.0, (active_days - 1) / (ref_active_days - 1))

    # Frequency component: 0 if detection_count <= 1
    if detection_count <= 1 or ref_detection_count <= 1:
        s_freq = 0.0
    else:
        s_freq = min(1.0, (detection_count - 1) / (ref_detection_count - 1))

    raw_score = (days_weight * s_days) + (freq_weight * s_freq)
    score = max(0.0, min(1.0, raw_score))
    return round(score, 3)
