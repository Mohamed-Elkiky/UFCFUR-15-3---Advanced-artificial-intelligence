"""Grad-CAM implementation for visualising CV quality grade decisions (AA-46)."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Gradient-weighted Class Activation Mapping for CNN quality models.

    Registers forward and backward hooks on a target conv layer to capture
    activations and gradients, then builds a heatmap showing which image
    regions drove the model's grade decision.

    Parameters
    ----------
    model : nn.Module
        Trained CNN model (e.g. ResNet50, EfficientNet).
    target_layer : nn.Module
        The convolutional layer to attach hooks to. Usually the last conv
        layer before global pooling.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(
        self, module: nn.Module, input: tuple, output: torch.Tensor
    ) -> None:
        self._activations = output.detach()

    def _save_gradients(
        self, module: nn.Module, grad_input: tuple, grad_output: tuple
    ) -> None:
        self._gradients = grad_output[0].detach()

    def generate(
        self, input_tensor: torch.Tensor
    ) -> Tuple[np.ndarray, int]:
        """Run forward + backward pass and compute the Grad-CAM heatmap.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Shape (1, C, H, W). Must already be normalised.

        Returns
        -------
        cam : np.ndarray
            Float array in [0, 1] with same spatial size as input_tensor.
        pred_class : int
            Index of the predicted class.
        """
        self.model.eval()
        output = self.model(input_tensor)
        pred_class = int(output.argmax(dim=1).item())

        self.model.zero_grad()
        output[0, pred_class].backward()

        # Global-average-pool the gradients over spatial dims
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted sum of activation maps
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Resize to input spatial size
        h, w = input_tensor.shape[-2], input_tensor.shape[-1]
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, pred_class

    def overlay_on_image(
        self, img: Image.Image | np.ndarray, cam: np.ndarray, alpha: float = 0.4
    ) -> np.ndarray:
        """Blend the Grad-CAM heatmap onto the original image.

        Parameters
        ----------
        img : PIL.Image or np.ndarray (H, W, 3) uint8
            Original RGB image.
        cam : np.ndarray
            Float array in [0, 1] from :meth:`generate`.
        alpha : float
            Heatmap blend weight (0 = original only, 1 = heatmap only).

        Returns
        -------
        np.ndarray
            Blended RGB image as uint8.
        """
        if isinstance(img, Image.Image):
            img = np.array(img.convert("RGB"))

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        if heatmap.shape[:2] != img.shape[:2]:
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        overlay = (
            (1 - alpha) * img.astype(np.float32)
            + alpha * heatmap.astype(np.float32)
        )
        return np.clip(overlay, 0, 255).astype(np.uint8)

    def remove_hooks(self) -> None:
        """Remove registered hooks to avoid memory leaks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def get_target_layer(model: nn.Module, model_type: str = "resnet50") -> nn.Module:
    """Return the last conv layer suitable for Grad-CAM for known architectures.

    Parameters
    ----------
    model : nn.Module
        Loaded model instance.
    model_type : str
        One of 'resnet50', 'efficientnet_b0', 'mobilenet_v3_small',
        'mobilenet_v3_large', 'custom'.

    Returns
    -------
    nn.Module
        The target conv layer.
    """
    if model_type.startswith("resnet"):
        return model.layer4[-1].conv2  # type: ignore[attr-defined]
    if model_type.startswith("efficientnet"):
        return model.features[-1][0]  # type: ignore[index]
    if "mobilenet" in model_type:
        return model.features[-1][0]  # type: ignore[index]
    # Custom CNN: last conv in features
    for layer in reversed(list(model.features.children())):  # type: ignore[attr-defined]
        if isinstance(layer, nn.Sequential):
            for sub in reversed(list(layer.children())):
                if isinstance(sub, nn.Conv2d):
                    return sub
        if isinstance(layer, nn.Conv2d):
            return layer
    raise ValueError(f"Cannot auto-detect target layer for model_type={model_type!r}")
