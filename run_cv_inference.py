import argparse
from pathlib import Path
import sys
import torch

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from thermaleye.cv.classifier import load_model, predict_image

def main():
    parser = argparse.ArgumentParser(description="Run EfficientNet-B0 inference on a Sentinel-2 patch")
    parser.add_argument("--image", type=Path, required=True, help="Path to input image")
    parser.add_argument("--model-dir", type=Path, default=Path("models/eurosat_cv"), help="Path to saved model directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = args.model_dir / "efficientnet_b0_eurosat.pth"
    config_path = args.model_dir / "model_config.json"
    
    try:
        model, classes = load_model(weights_path, config_path, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    print(f"Running inference on {args.image.name}...")
    try:
        scores = predict_image(args.image, model, classes, device)
    except Exception as e:
        print(f"Inference error: {e}")
        sys.exit(1)
        
    print("\n--- Predictions ---")
    # Sort by score descending
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for cls, score in sorted_scores:
        print(f"  {cls:<20}: {score:.4f}")
        
if __name__ == "__main__":
    main()
