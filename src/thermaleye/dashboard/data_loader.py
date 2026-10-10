"""Data loading and filtering logic for the ThermalEye dashboard."""
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

DEFAULT_CSV_PATH = Path("data/processed/prioritized_events_v2.csv")

def load_events_data(csv_path: Path = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """Load and optionally clean prioritized events for the dashboard."""
    if not csv_path.exists():
        logger.warning(f"Dashboard data file not found: {csv_path}")
        return pd.DataFrame()
        
    try:
        df = pd.read_csv(csv_path)
        # Convert coverage to string in case of NaN
        if 'evidence_coverage' in df.columns:
            df['evidence_coverage'] = df['evidence_coverage'].fillna('none').astype(str)
            
        # Create map color codes based on priority
        def get_color(p):
            if p == 'High': return [255, 0, 0, 200]
            if p == 'Medium': return [255, 165, 0, 200]
            if p == 'Low': return [255, 255, 0, 160]
            return [128, 128, 128, 100]  # Insufficient Evidence
            
        if 'priority' in df.columns:
            df['map_color'] = df['priority'].apply(get_color)
            
        return df
    except Exception as e:
        logger.error(f"Failed to load dashboard data: {e}")
        return pd.DataFrame()

def filter_events(df: pd.DataFrame, 
                 search_id: str = "", 
                 priorities: list = None, 
                 coverage_required: list = None) -> pd.DataFrame:
    """Filter events based on user selections."""
    if df.empty:
        return df
        
    filtered = df.copy()
    
    if search_id:
        filtered = filtered[filtered['event_id'].str.contains(search_id, case=False, na=False)]
        
    if priorities:
        filtered = filtered[filtered['priority'].isin(priorities)]
        
    if coverage_required:
        for cov in coverage_required:
            filtered = filtered[filtered['evidence_coverage'].str.contains(cov, case=False, na=False)]
            
    return filtered
