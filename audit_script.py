import pandas as pd
from pathlib import Path
import json

paths = {
    'raw': 'data/samples/india_viirs_recent.csv',
    'grouped': 'data/processed/grouped_detections.csv',
    'events': 'data/processed/thermal_events.csv',
    'osm': 'data/processed/thermal_events_osm.csv',
    's2': 'data/processed/sentinel2_metadata.csv',
    'prioritized': 'data/processed/prioritized_events_v2.csv'
}

print("=== ThermalEye E2E Audit ===")
results = {}

for name, p in paths.items():
    path = Path(p)
    if path.exists():
        df = pd.read_csv(path)
        print(f"[{name}] Rows: {len(df)}")
        if 'event_id' in df.columns:
            print(f"[{name}] Unique event_ids: {df['event_id'].nunique()}")
        if 'latitude' in df.columns and 'longitude' in df.columns:
            print(f"[{name}] Bounds - Lat: {df['latitude'].min():.2f} to {df['latitude'].max():.2f}, Lon: {df['longitude'].min():.2f} to {df['longitude'].max():.2f}")
        elif 'centroid_lat' in df.columns and 'centroid_lon' in df.columns:
            print(f"[{name}] Bounds - Lat: {df['centroid_lat'].min():.2f} to {df['centroid_lat'].max():.2f}, Lon: {df['centroid_lon'].min():.2f} to {df['centroid_lon'].max():.2f}")
            
        results[name] = df
    else:
        print(f"[{name}] NOT FOUND")
        results[name] = None

print("\n--- Cross-file consistency ---")
evt_df = results['events']
prio_df = results['prioritized']
if evt_df is not None and prio_df is not None:
    diff = set(evt_df['event_id']) - set(prio_df['event_id'])
    print(f"Events in 'events' but not in 'prioritized': {len(diff)}")

print("\n--- OSM Stats ---")
osm_df = results['osm']
if osm_df is not None:
    avail = osm_df['osm_data_available'].sum() if 'osm_data_available' in osm_df.columns else 0
    missing = len(osm_df) - avail
    print(f"OSM Queries Executed & Successful: {avail} (Failed/Timeout: {missing})")
    
    ind_1km = (osm_df['industrial_features_1km'] > 0).sum()
    ind_landuse = osm_df['industrial_landuse_present'].sum()
    print(f"OSM Successful but empty: {avail - max(ind_1km, ind_landuse)}")

print("\n--- Sentinel-2 Stats ---")
s2_df = results['s2']
if s2_df is not None:
    success = s2_df['success'].sum() if 'success' in s2_df.columns else 0
    print(f"S2 Image fetches successful: {success} / {len(s2_df)}")

print("\n--- Priority Stats ---")
if prio_df is not None:
    print(prio_df['priority'].value_counts())
    print("\nScore Distributions:")
    print(prio_df[['score_firms', 'score_temporal', 'score_osm', 'score_cv']].describe())
