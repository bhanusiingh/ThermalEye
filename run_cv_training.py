import argparse
import logging
from pathlib import Path
import sys
import torch

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from thermaleye.cv.train import train_and_evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 on EuroSAT")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/eurosat"), help="Dataset download directory")
    parser.add_argument("--output-dir", type=Path, default=Path("models/eurosat_cv"), help="Directory to save model")
    parser.add_argument("--subset", type=int, default=-1, help="Use a subset of images for fast testing")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Phase 5 CV Training. Device: {device}")
    
    metrics = train_and_evaluate(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        subset_size=args.subset
    )
    
    print("\n--- Evaluation Results ---")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Macro F1 : {metrics['macro_f1']:.4f}")
    print("\nPer-class F1 Scores:")
    for cls, scores in metrics['per_class'].items():
        print(f"  {cls:<20}: {scores['f1']:.4f}")
        
    print(f"\nModel saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
