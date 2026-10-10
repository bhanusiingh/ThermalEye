"""Spatial-temporal matching for ThermalEye evaluation."""
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import math

def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    if lat1 == lat2 and lon1 == lon2: return 0.0
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlambda = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def match_incidents(
    golden_df: pd.DataFrame, 
    events_df: pd.DataFrame, 
    spatial_tol_km: float = 2.0,
    time_tol_hours: float = 24.0
) -> pd.DataFrame:
    """
    Match curated incidents to pipeline events.
    Returns the golden_df with new columns:
    - matched_event_id
    - match_status (Matched, Unmatched, Ambiguous, Inconclusive)
    - match_distance_km
    - match_time_diff_hours
    - candidate_count
    """
    results = []
    
    if events_df.empty:
        # Everything is Inconclusive if pipeline yielded exactly 0 events (e.g., satellite blindness)
        for _, row in golden_df.iterrows():
            r = row.to_dict()
            r.update({
                'matched_event_id': None,
                'match_status': 'Inconclusive',
                'match_distance_km': None,
                'match_time_diff_hours': None,
                'candidate_count': 0
            })
            results.append(r)
        return pd.DataFrame(results)

    # Pre-parse event dates
    events = events_df.copy()
    try:
        events['first_dt'] = pd.to_datetime(events['first_detected'], utc=True)
        events['last_dt'] = pd.to_datetime(events['last_detected'], utc=True)
    except Exception:
        events['first_dt'] = pd.to_datetime(events['first_detected'])
        events['last_dt'] = pd.to_datetime(events['last_detected'])

    for _, row in golden_df.iterrows():
        out = row.to_dict()
        try:
            inc_dt = pd.to_datetime(row['event_date'], utc=True)
        except Exception:
            try:
                inc_dt = pd.to_datetime(row['event_date'])
            except Exception:
                out.update({'matched_event_id': None, 'match_status': 'Inconclusive', 'match_distance_km': None, 'match_time_diff_hours': None, 'candidate_count': 0})
                results.append(out)
                continue

        lat, lon = row['latitude'], row['longitude']
        
        candidates = []
        # Check all pipeline events (in a real system use KDTree, but N=300 is tiny)
        for _, evt in events.iterrows():
            dist = haversine_distance_km(lat, lon, evt['centroid_lat'], evt['centroid_lon'])
            if dist <= spatial_tol_km:
                # Time check
                t1 = (evt['first_dt'] - inc_dt).total_seconds() / 3600.0
                t2 = (evt['last_dt'] - inc_dt).total_seconds() / 3600.0
                
                # Minimum absolute time difference to any point in the event duration
                if t1 <= time_tol_hours and t2 >= -time_tol_hours:
                    # Valid overlap or within bounds
                    # Calculate closest time diff
                    if t1 > 0: diff_h = t1
                    elif t2 < 0: diff_h = t2
                    else: diff_h = 0.0
                    
                    candidates.append({
                        'event_id': evt['event_id'],
                        'distance_km': dist,
                        'time_diff_hours': diff_h
                    })
        
        out['candidate_count'] = len(candidates)
        
        if len(candidates) == 0:
            out['match_status'] = 'Unmatched'
            out['matched_event_id'] = None
            out['match_distance_km'] = None
            out['match_time_diff_hours'] = None
        elif len(candidates) == 1:
            out['match_status'] = 'Matched'
            out['matched_event_id'] = candidates[0]['event_id']
            out['match_distance_km'] = candidates[0]['distance_km']
            out['match_time_diff_hours'] = candidates[0]['time_diff_hours']
        else:
            out['match_status'] = 'Ambiguous'
            # Do NOT force a match for ambiguous cases. Log them but do not assign matched_event_id.
            out['matched_event_id'] = None
            out['match_distance_km'] = None
            out['match_time_diff_hours'] = None
            
        results.append(out)

    df_results = pd.DataFrame(results)
    
    # Second pass: check for competing incidents (multiple incidents matched to the same event)
    if not df_results.empty:
        # Find all event IDs that are claimed by more than 1 sample
        counts = df_results[df_results['match_status'] == 'Matched']['matched_event_id'].value_counts()
        competing_events = counts[counts > 1].index.tolist()
        
        for idx, row in df_results.iterrows():
            if row['matched_event_id'] in competing_events:
                df_results.at[idx, 'match_status'] = 'Conflict'
                df_results.at[idx, 'matched_event_id'] = None
                df_results.at[idx, 'match_distance_km'] = None
                df_results.at[idx, 'match_time_diff_hours'] = None

    return df_results
