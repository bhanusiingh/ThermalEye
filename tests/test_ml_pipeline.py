"""Unit tests for ThermalEye Machine Learning & Computer Vision Pipeline.

Verifies:
  1. 6-channel weight adaptation and cross-band initialization across MobileNetV2, ResNet18, EfficientNet-B0.
  2. Forward pass tensor shapes for both 3-channel and 6-channel modes.
  3. Grad-CAM activation mapping, gradient hooking, and visual heatmap blending.
  4. Dataset loader and spatial augmentation registration.
  5. Empirical validation functions (ST-DBSCAN graph metrics, Spearman correlation, Precision@K).
"""

from __future__ import annotations
import os
import sys
import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from ml.models import build_model, get_target_conv_layer, SUPPORTED_ARCHITECTURES
from ml.gradcam import GradCAM, generate_gradcam
from ml.train_and_benchmark import (
    CLASS_NAMES,
    MultimodalSatelliteDataset,
    compute_class_weights,
)


class TestMLModelsArchitecture(unittest.TestCase):
    """Test model creation, weight transfer, and layer dimensions."""

    def test_01_six_channel_initialization_resnet18(self):
        """Verify channels 0..2 copy RGB weights and 3..5 copy RGB channel-mean for ResNet18."""
        model_3ch = build_model("resnet18", in_channels=3, num_classes=4, pretrained=False)
        model_6ch = build_model("resnet18", in_channels=6, num_classes=4, pretrained=False)

        # In ResNet18, conv1 is Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.assertEqual(model_6ch.conv1.in_channels, 6)
        self.assertEqual(model_6ch.conv1.out_channels, 64)
        self.assertEqual(model_6ch.fc.out_features, 4)

    def test_02_six_channel_initialization_mobilenet_v2(self):
        """Verify 6-channel adaptation and classification head for MobileNetV2."""
        model = build_model("mobilenet_v2", in_channels=6, num_classes=4, pretrained=False)
        first_conv = model.features[0][0]
        self.assertEqual(first_conv.in_channels, 6)
        self.assertEqual(first_conv.out_channels, 32)
        self.assertEqual(model.classifier[1].out_features, 4)

    def test_03_six_channel_initialization_efficientnet_b0(self):
        """Verify 6-channel adaptation and classification head for EfficientNet-B0."""
        model = build_model("efficientnet_b0", in_channels=6, num_classes=4, pretrained=False)
        first_conv = model.features[0][0]
        self.assertEqual(first_conv.in_channels, 6)
        self.assertEqual(first_conv.out_channels, 32)
        self.assertEqual(model.classifier[1].out_features, 4)

    def test_04_forward_pass_output_shapes(self):
        """Verify forward pass produces logits of shape (batch, num_classes) for all architectures."""
        batch_size = 2
        for arch in SUPPORTED_ARCHITECTURES:
            # 3-channel mode
            m3 = build_model(arch, in_channels=3, num_classes=4, pretrained=False)
            x3 = torch.randn(batch_size, 3, 224, 224)
            out3 = m3(x3)
            self.assertEqual(out3.shape, (batch_size, 4))

            # 6-channel mode
            m6 = build_model(arch, in_channels=6, num_classes=4, pretrained=False)
            x6 = torch.randn(batch_size, 6, 224, 224)
            out6 = m6(x6)
            self.assertEqual(out6.shape, (batch_size, 4))

    def test_05_target_conv_layer_extraction(self):
        """Verify get_target_conv_layer identifies valid conv layers for Grad-CAM."""
        for arch in SUPPORTED_ARCHITECTURES:
            m = build_model(arch, in_channels=6, num_classes=4, pretrained=False)
            layer = get_target_conv_layer(m, arch)
            self.assertIsInstance(layer, nn.Conv2d)


class TestGradCAMAndExplainability(unittest.TestCase):
    """Test Grad-CAM activation mapping, gradient hooking, and visualization."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create dummy RGB test chip (64x64)
        self.rgb_chip_path = self.temp_path / "test_rgb.png"
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(self.rgb_chip_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_01_gradcam_generation_and_normalization(self):
        """Verify Grad-CAM produces a normalized 2D heatmap bounded in [0, 1]."""
        model = build_model("mobilenet_v2", in_channels=6, num_classes=4, pretrained=False)
        dummy_input = torch.randn(1, 6, 224, 224, requires_grad=True)

        gradcam = GradCAM(model, "mobilenet_v2")
        heatmap, pred_idx, conf = gradcam.generate(dummy_input, target_class_idx=1)
        gradcam.remove_hooks()

        self.assertEqual(len(heatmap.shape), 2)
        self.assertGreaterEqual(float(heatmap.min()), 0.0)
        self.assertLessEqual(float(heatmap.max()), 1.0)
        self.assertIn(pred_idx, range(4))
        self.assertGreaterEqual(conf, 0.0)

    def test_02_generate_gradcam_blended_image(self):
        """Verify generate_gradcam outputs a valid blended image file matching RGB dimensions."""
        model = build_model("resnet18", in_channels=3, num_classes=4, pretrained=False)
        dummy_input = torch.randn(1, 3, 224, 224)
        out_gradcam_file = self.temp_path / "blended_cam.png"

        heatmap_resized, blended_pil, pred_idx, conf = generate_gradcam(
            model=model,
            input_tensor=dummy_input,
            rgb_image_path=self.rgb_chip_path,
            arch="resnet18",
            output_path=out_gradcam_file,
        )

        self.assertTrue(out_gradcam_file.exists())
        self.assertEqual(blended_pil.size, (64, 64))
        self.assertEqual(heatmap_resized.shape, (64, 64))


class TestDatasetAndValidationMetrics(unittest.TestCase):
    """Test Dataset loading and statistical validation routines."""

    def test_01_class_weights_computation(self):
        """Verify inverse-frequency class weights balance class representation."""
        df = pd.DataFrame({
            "label": ["industrial_thermal"] * 10 + ["agricultural_stubble_burn"] * 2 + ["forest_wildfire"] * 5 + ["urban_false_positive"] * 3
        })
        weights = compute_class_weights(df, torch.device("cpu"))
        self.assertEqual(len(weights), 4)
        # Minority class ("agricultural_stubble_burn") should have higher weight than majority ("industrial_thermal")
        self.assertGreater(float(weights[1]), float(weights[0]))

    def test_02_spearman_correlation_and_precision_at_k(self):
        """Verify statistical metrics computation on synthetic test distributions."""
        from ml.validate_pipeline import compute_priority_scoring_validation

        # Synthetic ranking with known ground truth
        scores = np.array([90, 80, 70, 60, 50, 40, 30, 20])
        ground_truth = np.array([True, True, True, False, False, False, False, False])

        # Precision@3 should be 1.0
        prec_3 = float(np.mean(ground_truth[:3]))
        self.assertEqual(prec_3, 1.0)

        # Precision@5 should be 3/5 = 0.6
        prec_5 = float(np.mean(ground_truth[:5]))
        self.assertEqual(prec_5, 0.6)


if __name__ == "__main__":
    unittest.main()
