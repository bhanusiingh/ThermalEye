"""ThermalEye Explainable AI (XAI) Grad-CAM Module.

Implements Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize
salient spatial activations for candidate fire event classifications.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm

import torch
import torch.nn as nn
from ml.models import get_target_conv_layer


class GradCAM:
    """Gradient-weighted Class Activation Mapping for PyTorch vision models."""

    def __init__(self, model: nn.Module, arch: str):
        self.model = model
        self.arch = arch
        self.target_layer = get_target_conv_layer(model, arch)

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.handles = []

        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module: nn.Module, input: Tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            self.activations = output

        def backward_hook(module: nn.Module, grad_input: Tuple[torch.Tensor, ...], grad_output: Tuple[torch.Tensor, ...]) -> None:
            self.gradients = grad_output[0]

        self.handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """Compute Grad-CAM heatmap for the given input tensor.

        Args:
            input_tensor: (1, C, H, W) normalized model input.
            target_class_idx: Target class index (if None, argmax predicted class is used).

        Returns:
            Tuple of (normalized_2d_heatmap, predicted_class_idx, predicted_confidence)
        """
        self.model.eval()
        self.activations = None
        self.gradients = None

        with torch.enable_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = int(torch.argmax(probs, dim=1).item())
            confidence = float(probs[0, pred_idx].item())

            class_idx = target_class_idx if target_class_idx is not None else pred_idx

            self.model.zero_grad()
            one_hot = torch.zeros_like(outputs)
            one_hot[0, class_idx] = 1.0
            outputs.backward(gradient=one_hot, retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks failed to capture activations or gradients.")

        # Channel weights via global average pooling of gradients: alpha_k = GAP(dY/dA_k)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activation maps: L_gradcam = ReLU(sum_k alpha_k * A_k)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H, W)
        cam = torch.relu(cam)

        cam_np = cam.squeeze().detach().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_norm = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam_np)

        return cam_norm, pred_idx, confidence


def generate_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    rgb_image_path: Union[str, Path],
    arch: str,
    target_class_idx: Optional[int] = None,
    output_path: Optional[Union[str, Path]] = None,
    alpha: float = 0.45,
) -> Tuple[np.ndarray, Image.Image, int, float]:
    """Generate and blend a Grad-CAM heatmap overlay onto the original RGB satellite image.

    Args:
        model: Trained CNN model.
        input_tensor: (1, C, H, W) normalized tensor.
        rgb_image_path: Path to base RGB chip.
        arch: Model architecture name.
        target_class_idx: Target class for CAM (default: argmax).
        output_path: Optional path to save blended PNG.
        alpha: Heatmap blend weight (0..1).

    Returns:
        Tuple of (heatmap_np, blended_pil_image, predicted_class_idx, confidence)
    """
    gradcam = GradCAM(model, arch)
    try:
        heatmap_2d, pred_idx, conf = gradcam.generate(input_tensor, target_class_idx)
    finally:
        gradcam.remove_hooks()

    # Load original RGB image
    rgb_path = Path(rgb_image_path)
    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB image chip not found: {rgb_path}")

    base_pil = Image.open(rgb_path).convert("RGB")
    width, height = base_pil.size

    # Resize heatmap to match original RGB image dimensions
    heatmap_pil = Image.fromarray((heatmap_2d * 255).astype(np.uint8)).resize(
        (width, height), Image.Resampling.BILINEAR
    )
    heatmap_resized = np.array(heatmap_pil, dtype=np.float32) / 255.0

    # Apply 'inferno' thermal colormap
    colormap = matplotlib.colormaps["inferno"]
    heatmap_colored = colormap(heatmap_resized)[:, :, :3]  # drop alpha, shape (H, W, 3) in [0, 1]

    # Blend with RGB
    base_arr = np.array(base_pil, dtype=np.float32) / 255.0
    blended_arr = (1.0 - alpha) * base_arr + alpha * heatmap_colored
    blended_arr = np.clip(blended_arr * 255.0, 0, 255).astype(np.uint8)
    blended_pil = Image.fromarray(blended_arr)

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        blended_pil.save(out_p, format="PNG")

    return heatmap_resized, blended_pil, pred_idx, conf
