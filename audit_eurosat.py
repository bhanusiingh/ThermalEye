import os
import hashlib
from PIL import Image
from pathlib import Path
from collections import defaultdict
import torch
from torch.utils.data import random_split
from torchvision.datasets import EuroSAT
import time

def dhash(image, hash_size=8):
    image = image.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    diff = []
    for row in range(hash_size):
        for col in range(hash_size):
            pixel_left = image.getpixel((col, row))
            pixel_right = image.getpixel((col + 1, row))
            diff.append(pixel_left > pixel_right)
    decimal_value = 0
    hex_string = []
    for index, value in enumerate(diff):
        if value:
            decimal_value += 2**(index % 8)
        if (index % 8) == 7:
            hex_string.append(hex(decimal_value)[2:].rjust(2, '0'))
            decimal_value = 0
    return ''.join(hex_string)

def main():
    eurosat_dir = Path("data/raw/eurosat/eurosat/2750")
    if not eurosat_dir.exists():
        print("EuroSAT dir not found")
        return

    # Check Metadata files
    print("--- Metadata Check ---")
    metadata_files = [f for f in eurosat_dir.rglob("*") if f.is_file() and f.suffix.lower() not in ['.jpg', '.jpeg', '.png']]
    print("Non-image files found:", [str(f.relative_to(eurosat_dir)) for f in metadata_files])

    print("\n--- Dataset Splits ---")
    dataset = EuroSAT(root="data/raw/eurosat", download=False)
    total = len(dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # In EuroSAT, dataset._samples or dataset.samples is usually available if it inherits ImageFolder
    # let's safely get paths
    if hasattr(dataset, '_samples'):
        samples = dataset._samples
    elif hasattr(dataset, 'samples'):
        samples = dataset.samples
    else:
        print("Cannot find samples attribute")
        return

    split_map = {}
    for i in train_ds.indices: split_map[i] = "Train"
    for i in val_ds.indices: split_map[i] = "Val"
    for i in test_ds.indices: split_map[i] = "Test"

    print("Computing hashes for exact and near-duplicates...")
    exact_hashes = defaultdict(list)
    dhashes = defaultdict(list)
    
    start_t = time.time()
    for idx, (img_path, target) in enumerate(samples):
        with open(img_path, 'rb') as f:
            b = f.read()
            exact_hashes[hashlib.md5(b).hexdigest()].append(idx)
        try:
            img = Image.open(img_path)
            dh = dhash(img)
            dhashes[dh].append(idx)
        except Exception:
            pass
            
    print(f"Hashing took {time.time()-start_t:.2f}s")

    exact_dups = {k: v for k, v in exact_hashes.items() if len(v) > 1}
    print(f"\nExact Duplicates (MD5): {sum(len(v)-1 for v in exact_dups.values())} files involved in {len(exact_dups)} collision groups.")
    
    cross_split_exact = 0
    for k, v in exact_dups.items():
        splits = set(split_map[i] for i in v)
        if len(splits) > 1:
            cross_split_exact += 1
    print(f"Exact duplicates spanning across different splits (Train/Val/Test leakage): {cross_split_exact} groups")

    near_dups = {k: v for k, v in dhashes.items() if len(v) > 1}
    print(f"\nNear Duplicates (dHash=0): {sum(len(v)-1 for v in near_dups.values())} files involved in {len(near_dups)} collision groups.")
    
    cross_split_near = 0
    for k, v in list(near_dups.items())[:10]:
        print(f"  dHash {k}: Indices {v} -> {[split_map[i] for i in v]} | Classes: {[dataset.classes[samples[i][1]] for i in v]}")
        
    for k, v in near_dups.items():
        splits = set(split_map[i] for i in v)
        if len(splits) > 1:
            cross_split_near += 1
    print(f"Near duplicates spanning across different splits (Leakage): {cross_split_near} groups")

if __name__ == "__main__":
    main()
