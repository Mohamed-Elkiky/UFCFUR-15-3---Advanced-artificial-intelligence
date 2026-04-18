"""CNN models for the CV quality grading task.

This module defines the baseline CustomCNN architecture and provides
utilities for transfer learning with pre-trained models (ResNet, EfficientNet, MobileNet).
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
from torchvision import models


class CustomCNN(nn.Module):
    """Baseline CNN for fresh/rotten classification.
    
    Architecture:
        - 3 convolutional blocks (Conv2d → ReLU → MaxPool2d)
        - AdaptiveAvgPool2d for variable input sizes
        - Flatten layer
        - 2 fully connected layers with Dropout
    
    This baseline model will be compared against transfer learning
    approaches (ResNet, EfficientNet) in the evaluation phase.
    
    Parameters
    ----------
    num_classes : int
        Number of output classes (28 for the full dataset).
    dropout_rate : float, optional
        Dropout probability for regularization (default: 0.5).
    
    Attributes
    ----------
    features : nn.Sequential
        Convolutional feature extraction blocks.
    avgpool : nn.AdaptiveAvgPool2d
        Adaptive pooling to fixed spatial size.
    classifier : nn.Sequential
        Fully connected classification head.
    
    Examples
    --------
    >>> model = CustomCNN(num_classes=28)
    >>> x = torch.randn(32, 3, 224, 224)  # Batch of 32 RGB images
    >>> output = model(x)
    >>> output.shape
    torch.Size([32, 28])
    """
    
    def __init__(self, num_classes: int, dropout_rate: float = 0.5):
        super().__init__()
        
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Feature extraction: 3 convolutional blocks
        self.features = nn.Sequential(
            # Block 1: 3 → 32 channels
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224×224 → 112×112
            
            # Block 2: 32 → 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112×112 → 56×56
            
            # Block 3: 64 → 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56×56 → 28×28
        )
        
        # Adaptive pooling: 28×28 → 7×7
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Flatten is handled in forward()
        
        # Classifier: Two FC layers with dropout
        # Input: 128 channels × 7 × 7 = 6272 features
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 3, height, width).
            Expected: height=width=224.
        
        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, num_classes).
        """
        x = self.features(x)      # (B, 128, 28, 28)
        x = self.avgpool(x)        # (B, 128, 7, 7)
        x = torch.flatten(x, 1)    # (B, 6272)
        x = self.classifier(x)     # (B, num_classes)
        return x
    
    def _initialize_weights(self) -> None:
        """Initialize network weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)


# ============================================================================
# TRANSFER LEARNING MODELS
# ============================================================================

def get_model(
    num_classes: int,
    model_type: Literal["resnet50", "efficientnet_b0", "mobilenet_v3_small", "mobilenet_v3_large", "custom_cnn"] = "resnet50",
    pretrained: bool = True,
    dropout_rate: float = 0.5,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Load a pre-trained model and adapt it for the quality grading task.
    
    Supports multiple architectures from torchvision with pre-trained ImageNet weights.
    The final classification layer is replaced with a custom head:
    Dropout → Linear(num_classes).
    
    Parameters
    ----------
    num_classes : int
        Number of output classes (28 for full dataset).
    model_type : str
        Model architecture to use. Options:
        - 'resnet50': ResNet-50 (25.6M params, good balance)
        - 'efficientnet_b0': EfficientNet-B0 (5.3M params, efficient)
        - 'mobilenet_v3_small': MobileNetV3-Small (2.5M params, fastest)
        - 'mobilenet_v3_large': MobileNetV3-Large (5.5M params)
        - 'custom_cnn': Baseline CustomCNN (1.7M params, no pretrained)
    pretrained : bool, optional
        If True, load ImageNet pre-trained weights (default: True).
    dropout_rate : float, optional
        Dropout probability in classification head (default: 0.5).
    freeze_backbone : bool, optional
        If True, freeze backbone weights (only train classification head).
        Useful for quick fine-tuning (default: False).
    
    Returns
    -------
    nn.Module
        Model ready for training/evaluation.
    
    Examples
    --------
    >>> # Load pre-trained ResNet50
    >>> model = get_model(num_classes=28, model_type='resnet50')
    >>> 
    >>> # Load EfficientNet without pre-training
    >>> model = get_model(num_classes=28, model_type='efficientnet_b0', pretrained=False)
    >>> 
    >>> # Freeze backbone, only train classification head
    >>> model = get_model(num_classes=28, model_type='resnet50', freeze_backbone=True)
    """
    if num_classes < 1:
        raise ValueError(f"num_classes must be >= 1, got {num_classes}")
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError(f"dropout_rate must be in [0, 1), got {dropout_rate}")
    
    # Special case: custom CNN (no transfer learning)
    if model_type == "custom_cnn":
        return CustomCNN(num_classes=num_classes, dropout_rate=dropout_rate)
    
    # Load pre-trained model
    weights = "DEFAULT" if pretrained else None
    
    if model_type == "resnet50":
        model = models.resnet50(weights=weights)
        # ResNet: Replace final fc layer
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, num_classes)
        )
    
    elif model_type == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        # EfficientNet: Replace classifier
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(num_features, num_classes)
        )
    
    elif model_type == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights)
        # MobileNetV3: Replace classifier
        num_features = model.classifier[3].in_features
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, model.classifier[0].out_features),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(num_features, num_classes)
        )
    
    elif model_type == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=weights)
        # MobileNetV3: Replace classifier
        num_features = model.classifier[3].in_features
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, model.classifier[0].out_features),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(num_features, num_classes)
        )
    
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Supported: resnet50, efficientnet_b0, mobilenet_v3_small, mobilenet_v3_large, custom_cnn"
        )
    
    # Optionally freeze backbone
    if freeze_backbone:
        freeze_model_backbone(model, model_type)
    
    return model


def freeze_model_backbone(model: nn.Module, model_type: str) -> None:
    """Freeze all parameters except the classification head.
    
    Parameters
    ----------
    model : nn.Module
        Model to freeze.
    model_type : str
        Model architecture type (determines which layers to freeze).
    """
    if model_type == "resnet50":
        # Freeze all layers except fc
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
    
    elif model_type in ["efficientnet_b0"]:
        # Freeze all layers except classifier
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
    
    elif model_type in ["mobilenet_v3_small", "mobilenet_v3_large"]:
        # Freeze all layers except classifier
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
    
    print(f"✓ Froze backbone for {model_type}")


def unfreeze_model(model: nn.Module) -> None:
    """Unfreeze all model parameters.
    
    Parameters
    ----------
    model : nn.Module
        Model to unfreeze.
    """
    for param in model.parameters():
        param.requires_grad = True
    print("✓ Unfroze all model parameters")


def create_model(
    num_classes: int,
    dropout_rate: float = 0.5,
    device: Optional[str] = None,
) -> CustomCNN:
    """Create and initialize a CustomCNN model (legacy compatibility).
    
    Parameters
    ----------
    num_classes : int
        Number of output classes.
    dropout_rate : float, optional
        Dropout probability (default: 0.5).
    device : str, optional
        Device to move model to ('cuda' or 'cpu'). If None, auto-detect.
    
    Returns
    -------
    CustomCNN
        Initialized model on the specified device.
    """
    model = CustomCNN(num_classes=num_classes, dropout_rate=dropout_rate)
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = model.to(device)
    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters in a model.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model to analyze.
    
    Returns
    -------
    tuple[int, int]
        (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_summary(model: nn.Module, model_type: Optional[str] = None, verbose: bool = True) -> None:
    """Print a summary of model architecture and parameter counts.
    
    Parameters
    ----------
    model : nn.Module
        Model to summarize.
    model_type : str, optional
        Model type name for display.
    verbose : bool, optional
        If True, print detailed layer information (default: True).
    """
    total, trainable = count_parameters(model)
    frozen = total - trainable
    
    print("=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    print(f"Architecture: {model_type or model.__class__.__name__}")
    if hasattr(model, "num_classes"):
        print(f"Classes:      {model.num_classes}")
    if hasattr(model, "dropout_rate"):
        print(f"Dropout:      {model.dropout_rate:.2f}")
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Frozen params:    {frozen:,}")
    if frozen > 0:
        print(f"Frozen ratio:     {100.0 * frozen / total:.1f}%")
    print("=" * 80)
    
    if verbose and hasattr(model, 'named_children'):
        print("\nLayer Groups:")
        print("-" * 80)
        for name, module in model.named_children():
            params_total = sum(p.numel() for p in module.parameters())
            params_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            frozen_marker = " [FROZEN]" if params_trainable == 0 and params_total > 0 else ""
            print(f"{name:<20} {params_total:>12,} params ({params_trainable:>12,} trainable){frozen_marker}")
        print("-" * 80)


if __name__ == "__main__":
    print("Testing Transfer Learning Models...")
    print("\n" + "=" * 80)
    
    # Test all models
    model_types = ["custom_cnn", "resnet50", "efficientnet_b0", "mobilenet_v3_small"]
    
    for model_type in model_types:
        print(f"\n{'='*80}")
        print(f"Testing: {model_type.upper()}")
        print(f"{'='*80}")
        
        # Create model
        pretrained = (model_type != "custom_cnn")
        model = get_model(
            num_classes=28,
            model_type=model_type,
            pretrained=pretrained,
            dropout_rate=0.5
        )
        
        # Print summary
        print_model_summary(model, model_type=model_type, verbose=False)
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        
        print(f"\nForward pass test:")
        print(f"  Input:  {tuple(x.shape)}")
        print(f"  Output: {tuple(output.shape)}")
        print(f"  ✓ {model_type} works correctly!")
    
    # Test freezing
    print("\n" + "=" * 80)
    print("Testing Backbone Freezing")
    print("=" * 80)
    
    model = get_model(num_classes=28, model_type="resnet50", freeze_backbone=True)
    total, trainable = count_parameters(model)
    print(f"ResNet50 with frozen backbone:")
    print(f"  Total: {total:,}, Trainable: {trainable:,}")
    print(f"  Only {100.0 * trainable / total:.1f}% trainable")
    
    print("\n✓ All models tested successfully!")