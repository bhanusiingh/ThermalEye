"""Training module for context classification."""
import json
import logging
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import EuroSAT
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from thermaleye.cv.classifier import ContextClassifier, get_preprocessing, save_model

logger = logging.getLogger(__name__)

def train_and_evaluate(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
    subset_size: int = -1
) -> Dict[str, Any]:
    """Train EfficientNet-B0 on EuroSAT and evaluate its performance."""
    # Ensure reproducible splits
    torch.manual_seed(seed)
    
    preprocess = get_preprocessing()
    
    # Download dataset
    logger.info(f"Loading EuroSAT dataset to {data_dir}...")
    dataset = EuroSAT(root=str(data_dir), download=True, transform=preprocess)
    
    if subset_size > 0 and subset_size < len(dataset):
        logger.info(f"Using a subset of {subset_size} images for fast testing.")
        dataset = Subset(dataset, range(subset_size))
        dataset.classes = dataset.dataset.classes
        total = len(dataset)
    else:
        total = len(dataset)
        
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Set num_workers based on OS and availability (use 0 for safer cross-platform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    logger.info("Initializing EfficientNet-B0 Context Classifier...")
    model = ContextClassifier(num_classes=len(dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training Loop
    logger.info("Starting training loop...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        train_loss = running_loss / len(train_ds)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_ds)
        
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
    # Testing
    logger.info("Evaluating on test set...")
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # Compute Metrics
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    prec_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0).tolist()
    rec_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0).tolist()
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0).tolist()
    cm = confusion_matrix(all_labels, all_preds).tolist()
    
    metrics = {
        "dataset": "EuroSAT RGB",
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": {
            dataset.classes[i]: {
                "precision": prec_per_class[i],
                "recall": rec_per_class[i],
                "f1": f1_per_class[i]
            } for i in range(len(dataset.classes))
        },
        "confusion_matrix": cm,
        "classes": dataset.classes,
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "seed": seed,
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
            "device": str(device)
        }
    }
    
    # Save model and results
    logger.info("Saving model and evaluation metrics...")
    save_model(model, dataset.classes, output_dir)
    with open(output_dir / "evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    return metrics
