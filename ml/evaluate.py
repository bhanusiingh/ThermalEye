"""ThermalEye Model Evaluation and Test Set Assessment Module.

Evaluates trained model checkpoints on the held-out test split,
computes classification metrics, latency benchmarks, and confusion matrices,
and exports artifacts/evaluation_report.json and artifacts/confusion_matrix.png.
"""

from __future__ import annotations
import os
import sys
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from ml.models import build_model
from ml.train_and_benchmark import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    MultimodalSatelliteDataset,
    load_dataset_stats,
)

MANIFEST_PATH = WORKSPACE_DIR / "data" / "cv_dataset" / "manifest.csv"
MODELS_DIR = WORKSPACE_DIR / "models"
ARTIFACTS_DIR = WORKSPACE_DIR / "artifacts"


def load_model_from_checkpoint(ckpt_path: Path, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load a trained model and metadata from a .pt file."""
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Check if champion checkpoint or individual config checkpoint
    if "champion_config" in checkpoint:
        arch = checkpoint["arch"]
        in_channels = checkpoint["in_channels"]
    else:
        config = checkpoint.get("config", {})
        arch = config.get("arch", "mobilenet_v2")
        in_channels = checkpoint.get("in_channels", config.get("in_channels", 6))

    model = build_model(arch=arch, in_channels=in_channels, num_classes=len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    meta = {
        "arch": arch,
        "in_channels": in_channels,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    return model, meta


def benchmark_latency(model: nn.Module, in_channels: int, device: torch.device, repetitions: int = 50) -> float:
    """Benchmark single-sample inference latency in milliseconds."""
    model.eval()
    dummy_input = torch.randn(1, in_channels, 224, 224, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(repetitions):
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

    mean_ms = ((end - start) / repetitions) * 1000.0
    return round(mean_ms, 2)


def evaluate_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Run full test-set evaluation and return comprehensive classification metrics."""
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[List[float]] = []

    with torch.no_grad():
        for inputs, targets, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.tolist())
            all_probs.extend(probs.cpu().tolist())

    acc = float(accuracy_score(all_targets, all_preds))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(all_targets, all_preds, average="weighted", zero_division=0))
    macro_prec = float(precision_score(all_targets, all_preds, average="macro", zero_division=0))
    macro_rec = float(recall_score(all_targets, all_preds, average="macro", zero_division=0))
    weighted_prec = float(precision_score(all_targets, all_preds, average="weighted", zero_division=0))
    weighted_rec = float(recall_score(all_targets, all_preds, average="weighted", zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(CLASS_NAMES))))

    # Per-class metrics
    per_class: Dict[str, Any] = {}
    clf_report = classification_report(
        all_targets,
        all_preds,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    for cls in CLASS_NAMES:
        if cls in clf_report:
            per_class[cls] = {
                "precision": round(clf_report[cls]["precision"], 4),
                "recall": round(clf_report[cls]["recall"], 4),
                "f1_score": round(clf_report[cls]["f1-score"], 4),
                "support": int(clf_report[cls]["support"]),
            }

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "weighted_precision": round(weighted_prec, 4),
        "weighted_recall": round(weighted_rec, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "predictions": all_preds,
        "targets": all_targets,
    }


def render_confusion_matrix_plot(cm: np.ndarray, class_names: List[str], save_path: Path) -> None:
    """Render and save a high-contrast confusion matrix heatmap."""
    plt.figure(figsize=(8, 7), dpi=300)
    plt.style.use("default")

    fig, ax = plt.subplots(figsize=(8, 6.5))
    fig.patch.set_facecolor("#0d1a1f")
    ax.set_facecolor("#12232a")

    im = ax.imshow(cm, interpolation="nearest", cmap="YlGnBu")

    # Styling colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#93a6a7", labelsize=10)
    cbar.outline.set_edgecolor("#26393f")

    # Ticks & labels
    tick_marks = np.arange(len(class_names))
    short_labels = [name.replace("_", "\n") for name in class_names]
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(short_labels, color="#e8edeb", fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(short_labels, color="#e8edeb", fontsize=10)

    # Annotate numbers
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            text_color = "#091418" if val > thresh else "#e8edeb"
            ax.text(
                j,
                i,
                format(val, "d"),
                ha="center",
                va="center",
                color=text_color,
                fontsize=13,
                fontweight="bold",
            )

    ax.set_ylabel("True Ground Truth Label", color="#e8edeb", fontsize=12, labelpad=10)
    ax.set_xlabel("Predicted Model Label", color="#e8edeb", fontsize=12, labelpad=10)
    ax.set_title("ThermalEye CV Model — Test Set Confusion Matrix", color="#e8edeb", fontsize=14, pad=16, fontweight="bold")

    for spine in ax.spines.values():
        spine.set_color("#26393f")
        spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Rendered Confusion Matrix Plot -> {save_path}")


def main() -> None:
    """Execute evaluation across all models on held-out test split."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest missing at {MANIFEST_PATH}")

    manifest_df = pd.read_csv(MANIFEST_PATH)
    test_df = manifest_df[manifest_df["split"] == "test"].copy()
    if len(test_df) == 0:
        raise ValueError("No test split samples found in manifest.csv")

    mean_6ch, std_6ch = load_dataset_stats()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating {len(test_df)} held-out test samples on device: {device}")

    # Discover checkpoints
    checkpoints = {
        "mobilenet_v2_3ch": MODELS_DIR / "mobilenet_v2_3ch.pt",
        "mobilenet_v2_6ch": MODELS_DIR / "mobilenet_v2_6ch.pt",
        "resnet18_6ch": MODELS_DIR / "resnet18_6ch.pt",
        "efficientnet_b0_6ch": MODELS_DIR / "efficientnet_b0_6ch.pt",
        "thermaleye_best": MODELS_DIR / "thermaleye_best.pt",
    }

    eval_results: Dict[str, Any] = {}
    best_cm: Optional[np.ndarray] = None

    for model_name, ckpt_path in checkpoints.items():
        if not ckpt_path.exists():
            print(f"Skipping {model_name}: checkpoint not found at {ckpt_path}")
            continue

        model, meta = load_model_from_checkpoint(ckpt_path, device)
        in_ch = meta["in_channels"]

        test_ds = MultimodalSatelliteDataset(
            test_df,
            in_channels=in_ch,
            augment=False,
            mean=mean_6ch[:in_ch],
            std=std_6ch[:in_ch],
        )
        test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)

        metrics = evaluate_test_set(model, test_loader, device)
        latency_ms = benchmark_latency(model, in_ch, device)

        eval_results[model_name] = {
            "model_name": model_name,
            "architecture": meta["arch"],
            "input_channels": in_ch,
            "parameters": meta["parameters"],
            "inference_latency_ms": latency_ms,
            "metrics": metrics,
        }

        print(
            f"  {model_name:20s} | Acc: {metrics['accuracy']:.4f} | "
            f"Macro F1: {metrics['macro_f1']:.4f} | Latency: {latency_ms:.2f} ms"
        )

        if model_name == "thermaleye_best" or (best_cm is None and model_name == "efficientnet_b0_6ch"):
            best_cm = np.array(metrics["confusion_matrix"])

    # Export Evaluation Report
    report_payload = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "test_sample_count": len(test_df),
        "class_names": CLASS_NAMES,
        "models": eval_results,
    }

    report_path = ARTIFACTS_DIR / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    print(f"Saved Evaluation Report Artifact -> {report_path}")

    # Render confusion matrix plot for best model
    if best_cm is not None:
        cm_plot_path = ARTIFACTS_DIR / "confusion_matrix.png"
        render_confusion_matrix_plot(best_cm, CLASS_NAMES, cm_plot_path)


if __name__ == "__main__":
    main()
