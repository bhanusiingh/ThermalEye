"""Schema validation and labeling rules for the ThermalEye Golden Dataset.

LABELING RULES:
1. confirmed_industrial_fire: Must have verified ground-truth (e.g., news report, official incident DB) 
   confirming an emergency fire or explosion at an industrial facility, factory, or power plant.
   Merely being located near an industrial zone is NOT sufficient.
2. vegetation_fire: Verified crop burning, forest fire, or wildland fire.
3. routine_flaring: Verified industrial gas flaring. This is a thermal anomaly but NOT an emergency fire.
4. uncertain: Insufficient evidence to confidently assign one of the above classes. Do not guess.

EVIDENCE REQUIREMENT:
Provide a `source_url` (e.g., news article) and `source_date` for any confirmed event.
"""
import pandas as pd

VALID_CLASSES = {
    "confirmed_industrial_fire",
    "vegetation_fire",
    "routine_flaring",
    "uncertain"
}

EXPECTED_COLUMNS = [
    "sample_id",
    "latitude",
    "longitude",
    "event_date",
    "ground_truth_class",
    "source_url",
    "source_date",
    "verification_notes"
]

def validate_golden_dataset(df: pd.DataFrame) -> bool:
    """Validate the schema and classes of a provided golden dataset."""
    if df.empty:
        return True
        
    # Check columns
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
        
    # Check null IDs or coords
    if df['sample_id'].isnull().any():
        raise ValueError("Dataset contains null sample_ids.")
    if df['sample_id'].duplicated().any():
        raise ValueError("Dataset contains duplicate sample_ids.")
    if df[['latitude', 'longitude']].isnull().any().any():
        raise ValueError("Dataset contains null coordinates.")
        
    # Check classes
    invalid_classes = set(df['ground_truth_class'].dropna()) - VALID_CLASSES
    if invalid_classes:
        raise ValueError(f"Invalid ground_truth_class found: {invalid_classes}. Must be one of {VALID_CLASSES}")
        
    return True
