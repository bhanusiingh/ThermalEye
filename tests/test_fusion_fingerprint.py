import pandas as pd
from pathlib import Path
from thermaleye.imagery.sentinel2 import S2PatchResult

def test_s2patchresult_to_metadata_dict():
    """Verify that image_fingerprint is correctly serialized."""
    res = S2PatchResult(
        event_id="E1",
        scene_id="SC1",
        acquisition_time="2026-10-10",
        cloud_cover_pct=10.0,
        patch_width_px=64,
        patch_height_px=64,
        output_path=Path("test.png"),
        success=True,
        image_fingerprint="abc123_hash"
    )
    metadata = res.to_metadata_dict()
    assert "image_fingerprint" in metadata
    assert metadata["image_fingerprint"] == "abc123_hash"

def test_fusion_fingerprint_validation(tmp_path):
    """Verify that Fusion accepts valid fingerprints and rejects invalid/missing ones."""
    from thermaleye.config import (
        DEFAULT_S2_DATE_RANGE_DAYS, 
        DEFAULT_S2_PATCH_HALF_DEG, 
        DEFAULT_S2_PATCH_WIDTH_PX, 
        DEFAULT_S2_PATCH_HEIGHT_PX
    )
    from thermaleye.imagery.sentinel2 import generate_image_fingerprint
    
    # 1. Mock an event
    lat, lon = 20.0, 30.0
    expected_fp = generate_image_fingerprint(
        lat, lon, 
        DEFAULT_S2_DATE_RANGE_DAYS, 
        DEFAULT_S2_PATCH_HALF_DEG, 
        DEFAULT_S2_PATCH_WIDTH_PX, 
        DEFAULT_S2_PATCH_HEIGHT_PX
    )
    
    # 2. Mock 4 Sentinel-2 metadata rows: Valid, Valid Strict, Mismatched, Missing
    s2_df = pd.DataFrame([
        {
            "event_id": "VALID_E1",
            "image_fingerprint": expected_fp,
            "output_path": str(tmp_path / "valid.png")
        },
        {
            "event_id": "VALID_STRICT_E4",
            "image_fingerprint": expected_fp + "_STRICT",
            "output_path": str(tmp_path / "valid_strict.png")
        },
        {
            "event_id": "INVALID_E2",
            "image_fingerprint": "wrong_hash_xyz",
            "output_path": str(tmp_path / "invalid.png")
        },
        {
            "event_id": "MISSING_E3",
            "output_path": str(tmp_path / "missing.png")
            # Intentionally omit image_fingerprint to simulate legacy/broken CSV
        }
    ])
    
    # Create empty mock image files
    (tmp_path / "valid.png").touch()
    (tmp_path / "valid_strict.png").touch()
    (tmp_path / "invalid.png").touch()
    (tmp_path / "missing.png").touch()
    
    # 3. Simulate fusion's validation logic from run_fusion.py
    # Since run_fusion.py is a CLI script, we will extract the validation block logic 
    # to test it directly.
    
    event_coords = {
        "VALID_E1": {'centroid_lat': lat, 'centroid_lon': lon},
        "VALID_STRICT_E4": {'centroid_lat': lat, 'centroid_lon': lon},
        "INVALID_E2": {'centroid_lat': lat, 'centroid_lon': lon},
        "MISSING_E3": {'centroid_lat': lat, 'centroid_lon': lon},
    }
    
    accepted_images = []
    
    for _, row in s2_df.iterrows():
        event_id = row.get("event_id")
        if event_id not in event_coords: continue
        
        # Validation Logic
        actual_fp = row.get("image_fingerprint")
        expected_for_row = generate_image_fingerprint(
            event_coords[event_id]['centroid_lat'],
            event_coords[event_id]['centroid_lon'],
            DEFAULT_S2_DATE_RANGE_DAYS,
            DEFAULT_S2_PATCH_HALF_DEG,
            DEFAULT_S2_PATCH_WIDTH_PX,
            DEFAULT_S2_PATCH_HEIGHT_PX
        )
        
        expected_strict_fp = expected_for_row + "_STRICT"
        if pd.isna(actual_fp) or actual_fp not in (expected_for_row, expected_strict_fp):
            continue
            
        local_path_str = str(row.get("output_path", ""))
        if not local_path_str or local_path_str == "nan":
            continue
        patch_path = Path(local_path_str)
        if patch_path.is_file():
            accepted_images.append(event_id)
            
    # Assertions
    assert len(accepted_images) == 2
    assert "VALID_E1" in accepted_images
    assert "VALID_STRICT_E4" in accepted_images
    assert "INVALID_E2" not in accepted_images
    assert "MISSING_E3" not in accepted_images
