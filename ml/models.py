"""ThermalEye Computer Vision Models Architecture Module.

Provides pre-trained convolutional backbones (MobileNetV2, ResNet18, EfficientNet-B0)
adapted for multimodal 6-band satellite imagery (RGB + SWIR2/SWIR1/NIR) with
physically grounded cross-band weight initialization.
"""

from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models


SUPPORTED_ARCHITECTURES = ("mobilenet_v2", "resnet18", "efficientnet_b0")


def _adapt_first_conv(
    old_conv: nn.Conv2d,
    in_channels: int,
) -> nn.Conv2d:
    """Create a new Conv2d layer with in_channels and initialize weights.

    For channels 0..2 (RGB), pretrained weights are preserved.
    For channels 3..5 (SWIR/NIR), weights are initialized from the channel-wise
    mean of the pretrained RGB kernels, providing a physically sound prior.
    """
    if in_channels == old_conv.in_channels:
        return old_conv

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=(old_conv.bias is not None),
        padding_mode=old_conv.padding_mode,
    )

    with torch.no_grad():
        # Copy RGB pretrained weights
        rgb_channels = min(old_conv.in_channels, in_channels)
        new_conv.weight[:, :rgb_channels] = old_conv.weight[:, :rgb_channels]

        if in_channels > old_conv.in_channels:
            # Replicate channel mean of RGB for remaining satellite channels (e.g., SWIR/NIR)
            mean_rgb = old_conv.weight.mean(dim=1, keepdim=True)
            remaining_channels = in_channels - rgb_channels
            new_conv.weight[:, rgb_channels:] = mean_rgb.repeat(1, remaining_channels, 1, 1)

        if old_conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    return new_conv


def build_model(
    arch: str,
    in_channels: int = 6,
    num_classes: int = 4,
    pretrained: bool = True,
) -> nn.Module:
    """Instantiate and adapt a deep learning model for ThermalEye classification.

    Args:
        arch: One of 'mobilenet_v2', 'resnet18', or 'efficientnet_b0'.
        in_channels: 3 for RGB ablation, 6 for paired RGB+SWIR.
        num_classes: Number of target land-cover classes (default 4).
        pretrained: Whether to load ImageNet pre-trained weights.

    Returns:
        Adapted nn.Module ready for fine-tuning.
    """
    arch_norm = arch.lower().strip()
    if arch_norm not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Unsupported architecture '{arch}'. Supported: {SUPPORTED_ARCHITECTURES}"
        )

    if arch_norm == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)

        # Adapt first conv: features[0][0] is Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2)...)
        if in_channels != 3:
            model.features[0][0] = _adapt_first_conv(model.features[0][0], in_channels)

        # Replace classification head: classifier[1] is Linear(1280, 1000)
        in_feats = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_feats, num_classes)

    elif arch_norm == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)

        # Adapt first conv: conv1 is Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2)...)
        if in_channels != 3:
            model.conv1 = _adapt_first_conv(model.conv1, in_channels)

        # Replace classification head: fc is Linear(512, 1000)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, num_classes)

    elif arch_norm == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)

        # Adapt first conv: features[0][0] is Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2)...)
        if in_channels != 3:
            model.features[0][0] = _adapt_first_conv(model.features[0][0], in_channels)

        # Replace classification head: classifier[1] is Linear(1280, 1000)
        in_feats = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_feats, num_classes)

    # Attach model metadata for reproducibility
    model.arch = arch_norm  # type: ignore[assignment]
    model.in_channels = in_channels  # type: ignore[assignment]
    model.num_classes = num_classes  # type: ignore[assignment]

    return model


def get_target_conv_layer(model: nn.Module, arch: str) -> nn.Module:
    """Retrieve the target convolutional layer for Grad-CAM interpretability.

    Args:
        model: PyTorch model instance.
        arch: Model architecture identifier.

    Returns:
        The penultimate or final convolutional layer nn.Module.
    """
    arch_norm = arch.lower().strip()
    if arch_norm == "mobilenet_v2":
        # features[18] is Conv2dNormActivation; features[18][0] is Conv2d(320, 1280)
        return model.features[-1][0]  # type: ignore[union-attr]
    elif arch_norm == "resnet18":
        # layer4[-1].conv2 is the final conv in the 4th residual stage
        return model.layer4[-1].conv2  # type: ignore[union-attr]
    elif arch_norm == "efficientnet_b0":
        # features[8][0] is Conv2d(320, 1280)
        return model.features[-1][0]  # type: ignore[union-attr]
    else:
        raise ValueError(f"Unknown architecture '{arch}' for Grad-CAM layer extraction")
