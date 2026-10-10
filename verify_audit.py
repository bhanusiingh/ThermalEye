import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from PIL import Image
import json

def verify_cross_file():
    print("--- 1. Cross-File Integrity ---")
    raw = pd.read_csv("data/samples/india_viirs_recent.csv")
    dets = pd.read_csv("data/processed/grouped_detections.csv")
    events = pd.read_csv("data/processed/thermal_events.csv")
    osm = pd.read_csv("data/processed/thermal_events_osm.csv")
    prio = pd.read_csv("data/processed/prioritized_events_v2.csv")
    
    # Bidirectional set equality
    e_ids = set(events['event_id'])
    o_ids = set(osm['event_id'])
    p_ids = set(prio['event_id'])
    
    print(f"Events vs OSM exact match: {e_ids == o_ids}")
    print(f"Events vs Prio exact match: {e_ids == p_ids}")
    
    # Nulls & Duplicates
    print(f"Null IDs in Prio: {prio['event_id'].isnull().sum()}")
    print(f"Duplicate IDs in Prio: {prio['event_id'].duplicated().sum()}")
    
    # Geographic filtering via actual polygon
    try:
        india_poly = gpd.read_file("data/cache/countries.geojson")
        india_poly = india_poly[india_poly['name'] == 'India']
        
        gdf_dets = gpd.GeoDataFrame(dets, geometry=gpd.points_from_xy(dets['longitude'], dets['latitude']), crs="EPSG:4326")
        inside = gpd.sjoin(gdf_dets, india_poly, predicate="intersects")
        print(f"Detections strictly inside actual India polygon: {len(inside)} / {len(dets)}")
    except Exception as e:
        print(f"Geo validation failed: {e}")
        
    print("Raw FIRMS preservation: UNVERIFIED (No original hash baseline logged in Phase 1)")

def verify_osm():
    print("\n--- 2. OSM Coverage ---")
    osm = pd.read_csv("data/processed/thermal_events_osm.csv")
    # True status calculation
    success_avail = osm[osm['osm_data_available'] == True]
    failed = osm[osm['osm_data_available'] != True]
    
    has_features = success_avail[(success_avail['industrial_features_1km'] > 0) | (success_avail['industrial_landuse_present'] == True)]
    empty = success_avail[(success_avail['industrial_features_1km'] == 0) & (success_avail['industrial_landuse_present'] == False)]
    
    print(f"Failed/Timeouts: {len(failed)}")
    print(f"Successful & Has Features: {len(has_features)}")
    print(f"Successful & Empty: {len(empty)}")
    print("Previous audit calculation was confirmed DEFECTIVE (used aggregate max() instead of row-wise logic).")

def verify_cv():
    print("\n--- 3. Sentinel-2 CV Traceability ---")
    meta = pd.read_csv("data/processed/sentinel2_metadata.csv")
    prio = pd.read_csv("data/processed/prioritized_events_v2.csv")
    
    for _, row in meta.iterrows():
        eid = row['event_id']
        path = Path(row['output_path'])
        exists = path.exists()
        dim = "N/A"
        if exists:
            with Image.open(path) as img:
                dim = f"{img.width}x{img.height}"
                
        prio_row = prio[prio['event_id'] == eid]
        cv_score = prio_row['score_cv'].iloc[0] if not prio_row.empty else "Missing"
        print(f"Event: {eid} | Exists: {exists} | Dims: {dim} | Final CV Score: {cv_score}")
        
    missing_imgs = prio[~prio['event_id'].isin(meta['event_id'])]
    invented = missing_imgs[missing_imgs['score_cv'].notnull()]
    print(f"Events without images but with invented CV scores: {len(invented)}")

def verify_scoring():
    print("\n--- 4. Scoring Verification ---")
    prio = pd.read_csv("data/processed/prioritized_events_v2.csv")
    
    def manual_score(row):
        # FIRMS
        conf = row.get("mean_firms_confidence")
        frp = row.get("mean_frp")
        c_score = (conf/100.0) if pd.notna(conf) else 0.5
        f_score = min(frp/20.0, 1.0) if pd.notna(frp) else 0.5
        s_firms = (c_score + f_score) / 2.0
        
        # Temporal
        s_temp = min(float(row.get("persistence_score")), 1.0) if pd.notna(row.get("persistence_score")) else None
        
        # Available scores
        s_osm = row.get('score_osm')
        s_cv = row.get('score_cv')
        
        W = {'firms': 0.2, 'temporal': 0.3, 'osm': 0.25, 'cv': 0.25}
        total_w = 0.0
        w_sum = 0.0
        cov = []
        
        if pd.notna(s_firms): w_sum += s_firms*W['firms']; total_w += W['firms']; cov.append('firms')
        if pd.notna(s_temp): w_sum += s_temp*W['temporal']; total_w += W['temporal']; cov.append('temporal')
        if pd.notna(s_osm): w_sum += s_osm*W['osm']; total_w += W['osm']; cov.append('osm')
        if pd.notna(s_cv): w_sum += s_cv*W['cv']; total_w += W['cv']; cov.append('cv')
            
        calc_score = w_sum / total_w if total_w > 0 else 0.0
        
        if 'osm' not in cov and 'cv' not in cov:
            priority = "Insufficient evidence"
        else:
            if calc_score >= 0.7: priority = 'High'
            elif calc_score >= 0.4: priority = 'Medium'
            else: priority = 'Low'
            
        return calc_score, priority
        
    mismatches = 0
    for _, row in prio.iterrows():
        calc_score, calc_prio = manual_score(row)
        actual_score = row['priority_score']
        actual_prio = row['priority']
        
        if not np.isclose(calc_score, actual_score, atol=0.001) or calc_prio != actual_prio:
            mismatches += 1
            print(f"Mismatch on {row['event_id']}: Calc ({calc_score:.3f}, {calc_prio}) vs Actual ({actual_score}, {actual_prio})")
            
    print(f"Total manual recomputation mismatches: {mismatches}")

if __name__ == "__main__":
    verify_cross_file()
    verify_osm()
    verify_cv()
    verify_scoring()
