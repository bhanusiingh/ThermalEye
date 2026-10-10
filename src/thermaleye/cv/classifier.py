"""EfficientNet-B0 CV classification components for ThermalEye."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image

logger = logging.getLogger(__name__)

# Default EuroSAT classes. The model can map to whatever it is trained on.
EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]

class ContextClassifier(nn.Module):
    """EfficientNet-B0 based image context classifier."""
    def __init__(self, num_classes: int = len(EUROSAT_CLASSES)):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT
        self.model = models.efficientnet_b0(weights=weights)
        # Replace classifier head for transfer learning
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def get_preprocessing() -> transforms.Compose:
    """Return standard ImageNet preprocessing used by torchvision EfficientNet-B0."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

def save_model(
    model: nn.Module, 
    class_mapping: List[str], 
    save_dir: Path
) -> None:
    """Save model weights and configuration to disk."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save weights
    torch.save(model.state_dict(), save_dir / "efficientnet_b0_eurosat.pth")
    
    # Save config
    config = {
        "model": "efficientnet-b0",
        "classes": class_mapping,
        "input_size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    }
    with open(save_dir / "model_config.json", "w") as f:
        json.dump(config, f, indent=2)

def load_model(
    weights_path: Path, 
    config_path: Path, 
    device: torch.device
) -> Tuple[nn.Module, List[str]]:
    """Load model weights and configuration for inference."""
    if not weights_path.exists() or not config_path.exists():
        raise FileNotFoundError(f"Model weights or config not found in {weights_path.parent}")
        
    with open(config_path, "r") as f:
        config = json.load(f)
        
    classes = config.get("classes", EUROSAT_CLASSES)
    model = ContextClassifier(num_classes=len(classes))
    
    # Handle incompatible checkpoints gracefully
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights (incompatible or corrupt): {e}")
        
    model.to(device)
    model.eval()
    return model, classes

def predict_image(
    image_path: Path, 
    model: nn.Module, 
    classes: List[str], 
    device: torch.device
) -> Dict[str, float]:
    """Reusable inference function to predict class probabilities for a single image."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Corrupt or invalid image: {image_path} ({e})")
        
    preprocess = get_preprocessing()
    tensor = preprocess(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        scores = torch.softmax(outputs, dim=1).squeeze(0).cpu().tolist()
        
    return {classes[i]: score for i, score in enumerate(scores)}
