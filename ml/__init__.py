"""
ThermalEye — Machine Learning & Computer Vision Subsystem
=========================================================

Package providing multimodal dataset construction, 6-channel convolutional
backbones (RGB + SWIR), fine-tuning pipelines, quantitative benchmarks,
Grad-CAM explainability, and experimental validation.
"""

from ml.models import build_model

__all__ = ["build_model"]
