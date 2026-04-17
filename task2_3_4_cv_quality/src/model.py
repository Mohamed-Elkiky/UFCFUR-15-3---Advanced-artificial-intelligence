"""CNN models for the CV quality grading task.

This module defines the baseline CustomCNN architecture and provides
utilities for model creation, initialization, and summary display.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


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


def create_model(
    num_classes: int,
    dropout_rate: float = 0.5,
    device: Optional[str] = None,
) -> CustomCNN:
    """Create and initialize a CustomCNN model.
    
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
    
    Examples
    --------
    >>> model = create_model(num_classes=28, dropout_rate=0.3)
    >>> print(model.num_classes)
    28
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
    
    Examples
    --------
    >>> model = CustomCNN(num_classes=28)
    >>> total, trainable = count_parameters(model)
    >>> print(f"Total: {total:,}, Trainable: {trainable:,}")
    Total: 1,620,764, Trainable: 1,620,764
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_summary(model: nn.Module, verbose: bool = True) -> None:
    """Print a summary of model architecture and parameter counts.
    
    Parameters
    ----------
    model : nn.Module
        Model to summarize.
    verbose : bool, optional
        If True, print layer-by-layer details (default: True).
    """
    total, trainable = count_parameters(model)
    
    print("=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    print(f"Architecture: {model.__class__.__name__}")
    if hasattr(model, "num_classes"):
        print(f"Classes:      {model.num_classes}")
    if hasattr(model, "dropout_rate"):
        print(f"Dropout:      {model.dropout_rate:.2f}")
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Non-trainable:    {total - trainable:,}")
    print("=" * 80)
    
    if verbose:
        print("\nLayer Details:")
        print("-" * 80)
        for name, module in model.named_children():
            if isinstance(module, nn.Sequential):
                print(f"\n{name}:")
                for i, layer in enumerate(module):
                    params = sum(p.numel() for p in layer.parameters())
                    print(f"  {i}: {layer.__class__.__name__:<20} {params:>10,} params")
            else:
                params = sum(p.numel() for p in module.parameters())
                print(f"{name:<20} {params:>10,} params")
        print("-" * 80)


if __name__ == "__main__":
    # Demo: Create model and print summary
    print("Creating CustomCNN baseline model...")
    model = create_model(num_classes=28, dropout_rate=0.5)
    print_model_summary(model)
    
    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(4, 3, 224, 224)  # Batch of 4 images
    if torch.cuda.is_available():
        x = x.cuda()
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    print("\n✓ Model created and tested successfully!")