import os
import hashlib
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict
import torch
from torchvision.datasets import EuroSAT

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

    dataset = EuroSAT(root="data/raw/eurosat", download=False)
    
    if hasattr(dataset, '_samples'):
        samples = dataset._samples
    elif hasattr(dataset, 'samples'):
        samples = dataset.samples
    else:
        print("Cannot find samples attribute")
        return

    print("Investigating EuroSAT duplicates...")
    dhashes = defaultdict(list)
    pixel_hashes = defaultdict(list)
    
    for idx, (img_path, target) in enumerate(samples):
        try:
            img = Image.open(img_path).convert('RGB')
            # Perceptual hash
            dh = dhash(img)
            dhashes[dh].append((idx, img_path))
            
            # Pixel hash (exact pixel values)
            arr = np.array(img)
            phash = hashlib.md5(arr.tobytes()).hexdigest()
            pixel_hashes[phash].append((idx, img_path, np.std(arr)))
        except Exception:
            pass

    # Check dHash=0 collision
    dh0 = dhashes.get("0000000000000000", [])
    print(f"\n--- Analysis of dHash '0000000000000000' ---")
    print(f"Total images with this dHash: {len(dh0)}")
    
    if len(dh0) > 0:
        print("Sample files:")
        for idx, img_path in dh0[:5]:
            arr = np.array(Image.open(img_path).convert('RGB'))
            print(f"  {Path(img_path).name} | StdDev: {np.std(arr):.4f} | Min: {arr.min()} | Max: {arr.max()}")
            
    # Exact pixel duplicates
    exact_pixel_dups = {k: v for k, v in pixel_hashes.items() if len(v) > 1}
    print(f"\n--- Genuine Duplicates (Exact Pixel Match) ---")
    print(f"Groups found: {len(exact_pixel_dups)}")
    for k, v in list(exact_pixel_dups.items())[:5]:
        print(f"  PixelHash {k}: {len(v)} images. Sample: {Path(v[0][1]).name} and {Path(v[1][1]).name}")

if __name__ == "__main__":
    main()
