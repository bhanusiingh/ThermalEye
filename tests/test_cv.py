import json
from pathlib import Path
import torch
import pytest
from PIL import Image

from thermaleye.cv.classifier import (
    ContextClassifier, 
    get_preprocessing, 
    save_model, 
    load_model, 
    predict_image,
    EUROSAT_CLASSES
)

def test_preprocessing_transforms() -> None:
    preprocess = get_preprocessing()
    # Create dummy 64x64 image
    img = Image.new('RGB', (64, 64), color='red')
    tensor = preprocess(img)
    
    # Check output shape (C, H, W) -> 3, 224, 224
    assert tensor.shape == (3, 224, 224)
    # Check tensor type
    assert isinstance(tensor, torch.Tensor)

def test_context_classifier_output_shape() -> None:
    model = ContextClassifier(num_classes=5)
    # Batch of 2, 3 channels, 224x224
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    
    assert output.shape == (2, 5)

def test_save_and_load_model(tmp_path: Path) -> None:
    model = ContextClassifier(num_classes=3)
    classes = ["A", "B", "C"]
    
    # Save
    save_model(model, classes, tmp_path)
    
    assert (tmp_path / "efficientnet_b0_eurosat.pth").exists()
    assert (tmp_path / "model_config.json").exists()
    
    with open(tmp_path / "model_config.json") as f:
        config = json.load(f)
        assert config["classes"] == classes
        
    # Load
    device = torch.device("cpu")
    loaded_model, loaded_classes = load_model(
        tmp_path / "efficientnet_b0_eurosat.pth", 
        tmp_path / "model_config.json", 
        device
    )
    
    assert loaded_classes == classes
    assert loaded_model.num_classes == 3
    
def test_load_model_missing_files(tmp_path: Path) -> None:
    device = torch.device("cpu")
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nonexistent.pth", tmp_path / "model_config.json", device)

def test_predict_image(tmp_path: Path) -> None:
    model = ContextClassifier(num_classes=2)
    classes = ["Industrial", "Forest"]
    device = torch.device("cpu")
    
    # Create dummy image
    img_path = tmp_path / "dummy.png"
    Image.new('RGB', (64, 64), color='blue').save(img_path)
    
    scores = predict_image(img_path, model, classes, device)
    
    assert len(scores) == 2
    assert "Industrial" in scores
    assert "Forest" in scores
    assert sum(scores.values()) == pytest.approx(1.0)
    
def test_predict_image_missing_file() -> None:
    model = ContextClassifier()
    with pytest.raises(FileNotFoundError):
        predict_image(Path("missing_image.png"), model, EUROSAT_CLASSES, torch.device("cpu"))
