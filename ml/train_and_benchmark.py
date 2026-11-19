"""ThermalEye Model Training and Multimodal Benchmarking Suite.

Fine-tunes convolutional architectures (MobileNetV2, ResNet18, EfficientNet-B0)
on multimodal satellite imagery (RGB + SWIR). Benchmarks 6-channel performance
against a 3-channel RGB ablation baseline and selects the champion model.
"""

from __future__ import annotations
import os
import sys
import json
import time
import random
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ml.models import build_model, SUPPORTED_ARCHITECTURES


CLASS_NAMES = [
    "industrial_thermal",
    "agricultural_stubble_burn",
    "forest_wildfire",
    "urban_false_positive",
]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = WORKSPACE_DIR / "data" / "cv_dataset" / "manifest.csv"
STATS_PATH = WORKSPACE_DIR / "artifacts" / "dataset_stats.json"
MODELS_DIR = WORKSPACE_DIR / "models"
ARTIFACTS_DIR = WORKSPACE_DIR / "artifacts"


class MultimodalSatelliteDataset(Dataset):
    """PyTorch Dataset loading paired RGB and SWIR satellite imagery chips."""

    def __init__(
        self,
        df: pd.DataFrame,
        in_channels: int = 6,
        augment: bool = False,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        target_size: Tuple[int, int] = (224, 224),
    ):
        self.df = df.reset_index(drop=True)
        self.in_channels = in_channels
        self.augment = augment
        self.target_size = target_size

        # Default normalizations if stats missing
        if mean is None:
            mean = [0.485, 0.456, 0.406, 0.35, 0.50, 0.60][:in_channels]
        if std is None:
            std = [0.229, 0.224, 0.225, 0.25, 0.25, 0.25][:in_channels]

        self.mean = torch.tensor(mean[:in_channels], dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std[:in_channels], dtype=torch.float32).view(-1, 1, 1)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        event_id = str(row["event_id"])
        label_str = str(row["label"])
        label_idx = CLASS_TO_IDX[label_str]

        # Load RGB
        rgb_full = WORKSPACE_DIR / str(row["rgb_path"])
        if not rgb_full.exists():
            raise FileNotFoundError(f"RGB image chip missing: {rgb_full}")
        rgb_img = Image.open(rgb_full).convert("RGB").resize(self.target_size, Image.Resampling.BILINEAR)

        # Load SWIR if 6-channel
        swir_img = None
        if self.in_channels == 6:
            swir_full = WORKSPACE_DIR / str(row["swir_path"])
            if not swir_full.exists():
                raise FileNotFoundError(f"SWIR image chip missing: {swir_full}")
            swir_img = Image.open(swir_full).convert("RGB").resize(self.target_size, Image.Resampling.BILINEAR)

        # Synchronized spatial augmentations
        if self.augment:
            if random.random() > 0.5:
                rgb_img = rgb_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if swir_img:
                    swir_img = swir_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

            if random.random() > 0.5:
                rgb_img = rgb_img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                if swir_img:
                    swir_img = swir_img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

            rot = random.choice([0, 90, 180, 270])
            if rot > 0:
                rgb_img = rgb_img.rotate(rot)
                if swir_img:
                    swir_img = swir_img.rotate(rot)

        # Convert to float tensors in [0, 1]
        rgb_arr = np.array(rgb_img, dtype=np.float32) / 255.0
        rgb_t = torch.from_numpy(rgb_arr).permute(2, 0, 1)  # (3, H, W)

        if self.in_channels == 6 and swir_img is not None:
            swir_arr = np.array(swir_img, dtype=np.float32) / 255.0
            swir_t = torch.from_numpy(swir_arr).permute(2, 0, 1)  # (3, H, W)
            tensor = torch.cat([rgb_t, swir_t], dim=0)  # (6, H, W)
        else:
            tensor = rgb_t  # (3, H, W)

        # Normalize
        tensor = (tensor - self.mean) / (self.std + 1e-7)

        return tensor, label_idx, event_id


def load_dataset_stats() -> Tuple[List[float], List[float]]:
    """Load per-channel mean and std from dataset_stats.json."""
    if STATS_PATH.exists():
        with open(STATS_PATH, "r", encoding="utf-8") as f:
            stats = json.load(f)
            mean = stats.get("channel_stats", {}).get("mean", [])
            std = stats.get("channel_stats", {}).get("std", [])
            if len(mean) == 6 and len(std) == 6:
                return mean, std
    # Fallback to standard remote sensing / ImageNet values
    return [0.28, 0.25, 0.17, 0.38, 0.55, 0.62], [0.19, 0.13, 0.11, 0.23, 0.27, 0.27]


def compute_class_weights(train_df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    """Compute balanced inverse-frequency class weights."""
    counts = train_df["label"].value_counts().to_dict()
    total = len(train_df)
    n_classes = len(CLASS_NAMES)
    weights = []
    for cls in CLASS_NAMES:
        c = counts.get(cls, 1)
        w = total / (n_classes * c)
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate model on a dataset split and return detailed metrics."""
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    n_samples = max(1, len(all_targets))
    avg_loss = total_loss / n_samples
    acc = float(accuracy_score(all_targets, all_preds))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(all_targets, all_preds, average="weighted", zero_division=0))
    macro_prec = float(precision_score(all_targets, all_preds, average="macro", zero_division=0))
    macro_rec = float(recall_score(all_targets, all_preds, average="macro", zero_division=0))

    per_class_f1_arr = f1_score(all_targets, all_preds, average=None, zero_division=0)
    per_class_f1 = {
        CLASS_NAMES[i]: round(float(per_class_f1_arr[i]), 4)
        for i in range(len(CLASS_NAMES))
    }

    return {
        "loss": round(avg_loss, 4),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "per_class_f1": per_class_f1,
    }


def freeze_backbone(model: nn.Module, arch: str) -> None:
    """Freeze all layers except the final classification head."""
    for param in model.parameters():
        param.requires_grad = False

    if arch in ("mobilenet_v2", "efficientnet_b0"):
        for param in model.classifier.parameters():  # type: ignore[union-attr]
            param.requires_grad = True
    elif arch == "resnet18":
        for param in model.fc.parameters():  # type: ignore[union-attr]
            param.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all model parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


def train_single_configuration(
    config_name: str,
    arch: str,
    in_channels: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    device: torch.device,
    warmup_epochs: int = 5,
    finetune_epochs: int = 20,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train a single model configuration with two-phase fine-tuning."""
    print(f"\n[{config_name.upper()}] Initializing {arch} (in_channels={in_channels})...")
    start_time = time.time()

    model = build_model(arch, in_channels=in_channels, num_classes=len(CLASS_NAMES), pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    param_count = sum(p.numel() for p in model.parameters())

    # Phase 1: Warmup (train head only)
    freeze_backbone(model, arch)
    optimizer_warmup = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=1e-4,
    )

    for epoch in range(1, warmup_epochs + 1):
        model.train()
        for inputs, targets, _ in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer_warmup.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_warmup.step()

    # Phase 2: Full Fine-tuning
    unfreeze_all(model)
    optimizer_full = torch.optim.AdamW(
        [
            {"params": [p for n, p in model.named_parameters() if "classifier" not in n and "fc" not in n], "lr": 2e-4},
            {"params": [p for n, p in model.named_parameters() if "classifier" in n or "fc" in n], "lr": 5e-4},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_full, T_max=finetune_epochs, eta_min=1e-6)

    best_val_f1 = -1.0
    best_epoch = 0
    best_metrics: Dict[str, Any] = {}
    best_state_dict: Optional[Dict[str, Any]] = None

    for epoch in range(1, finetune_epochs + 1):
        model.train()
        for inputs, targets, _ in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer_full.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_full.step()

        scheduler.step()
        val_metrics = evaluate_model(model, val_loader, criterion, device)

        if (val_metrics["macro_f1"] > best_val_f1) or (
            val_metrics["macro_f1"] == best_val_f1 and val_metrics["accuracy"] >= best_metrics.get("accuracy", 0)
        ):
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_metrics = val_metrics
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_duration = round(time.time() - start_time, 2)
    print(
        f"  Completed {config_name}: Best Epoch {best_epoch}/{finetune_epochs} | "
        f"Val Acc: {best_metrics['accuracy']:.4f} | Macro F1: {best_metrics['macro_f1']:.4f} "
        f"({train_duration}s)"
    )

    if best_state_dict is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})

    result_info = {
        "config_name": config_name,
        "arch": arch,
        "in_channels": in_channels,
        "parameters": param_count,
        "train_duration_sec": train_duration,
        "best_epoch": best_epoch,
        "val_metrics": best_metrics,
    }

    return model, result_info


def run_benchmarks() -> Dict[str, Any]:
    """Train all 4 benchmark configurations, save checkpoints, and export comparisons."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest missing at {MANIFEST_PATH}. Run ml/build_dataset.py first.")

    manifest_df = pd.read_csv(MANIFEST_PATH)
    train_df = manifest_df[manifest_df["split"] == "train"].copy()
    val_df = manifest_df[manifest_df["split"] == "val"].copy()

    print(f"Loaded dataset: {len(train_df)} train, {len(val_df)} val, {len(manifest_df[manifest_df['split'] == 'test'])} test")

    mean_6ch, std_6ch = load_dataset_stats()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    class_weights = compute_class_weights(train_df, device)

    # Define the 4 benchmark configurations
    configurations = [
        {
            "name": "mobilenet_v2_3ch",
            "arch": "mobilenet_v2",
            "in_channels": 3,
            "description": "MobileNetV2 RGB Ablation (3 Channels)",
        },
        {
            "name": "mobilenet_v2_6ch",
            "arch": "mobilenet_v2",
            "in_channels": 6,
            "description": "MobileNetV2 Multimodal (6 Channels: RGB + SWIR)",
        },
        {
            "name": "resnet18_6ch",
            "arch": "resnet18",
            "in_channels": 6,
            "description": "ResNet18 Multimodal (6 Channels: RGB + SWIR)",
        },
        {
            "name": "efficientnet_b0_6ch",
            "arch": "efficientnet_b0",
            "in_channels": 6,
            "description": "EfficientNet-B0 Multimodal (6 Channels: RGB + SWIR)",
        },
    ]

    benchmark_results: Dict[str, Any] = {}
    trained_models: Dict[str, nn.Module] = {}

    for config in configurations:
        in_ch = config["in_channels"]
        train_ds = MultimodalSatelliteDataset(
            train_df,
            in_channels=in_ch,
            augment=True,
            mean=mean_6ch[:in_ch],
            std=std_6ch[:in_ch],
        )
        val_ds = MultimodalSatelliteDataset(
            val_df,
            in_channels=in_ch,
            augment=False,
            mean=mean_6ch[:in_ch],
            std=std_6ch[:in_ch],
        )

        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

        model, info = train_single_configuration(
            config_name=config["name"],
            arch=config["arch"],
            in_channels=in_ch,
            train_loader=train_loader,
            val_loader=val_loader,
            class_weights=class_weights,
            device=device,
        )

        # Save individual checkpoint
        ckpt_path = MODELS_DIR / f"{config['name']}.pt"
        torch.save(
            {
                "config": config,
                "state_dict": model.state_dict(),
                "val_metrics": info["val_metrics"],
                "in_channels": in_ch,
                "class_names": CLASS_NAMES,
            },
            ckpt_path,
        )

        trained_models[config["name"]] = model
        benchmark_results[config["name"]] = info

    # Determine Champion Model based on Validation Macro-F1
    best_config_name = max(
        benchmark_results.keys(),
        key=lambda k: (
            benchmark_results[k]["val_metrics"]["macro_f1"],
            benchmark_results[k]["val_metrics"]["accuracy"],
        ),
    )
    champion_info = benchmark_results[best_config_name]
    champion_model = trained_models[best_config_name]

    print(f"\n==================================================================")
    print(f" CHAMPION MODEL SELECTED: {best_config_name.upper()}")
    print(f" Architecture: {champion_info['arch']} ({champion_info['in_channels']} channels)")
    print(f" Validation Macro-F1: {champion_info['val_metrics']['macro_f1']:.4f}")
    print(f" Validation Accuracy: {champion_info['val_metrics']['accuracy']:.4f}")
    print(f"==================================================================")

    # Compute Multimodal Lift (6ch MobileNetV2 vs 3ch MobileNetV2)
    m3 = benchmark_results["mobilenet_v2_3ch"]["val_metrics"]["macro_f1"]
    m6 = benchmark_results["mobilenet_v2_6ch"]["val_metrics"]["macro_f1"]
    multimodal_lift = round(m6 - m3, 4)

    champion_payload = {
        "champion_config": best_config_name,
        "arch": champion_info["arch"],
        "in_channels": champion_info["in_channels"],
        "class_names": CLASS_NAMES,
        "idx_to_class": IDX_TO_CLASS,
        "normalization": {
            "mean": mean_6ch[: champion_info["in_channels"]],
            "std": std_6ch[: champion_info["in_channels"]],
        },
        "val_metrics": champion_info["val_metrics"],
        "state_dict": champion_model.state_dict(),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }

    champion_save_path = MODELS_DIR / "thermaleye_best.pt"
    torch.save(champion_payload, champion_save_path)
    print(f"Saved Champion Checkpoint -> {champion_save_path}")

    # Export model_comparison.json
    comparison_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_summary": {
            "champion_model": best_config_name,
            "champion_macro_f1": champion_info["val_metrics"]["macro_f1"],
            "champion_accuracy": champion_info["val_metrics"]["accuracy"],
            "multimodal_macro_f1_lift_vs_rgb": multimodal_lift,
        },
        "configurations": benchmark_results,
    }

    comparison_path = ARTIFACTS_DIR / "model_comparison.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison_payload, f, indent=2)
    print(f"Exported Model Comparison Artifact -> {comparison_path}")

    return comparison_payload


if __name__ == "__main__":
    run_benchmarks()
