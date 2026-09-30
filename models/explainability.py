"""
models/explainability.py
=========================
Grad-CAM and Grad-CAM++ explainability for OA detection models.

IMPORTANT
---------
Heatmaps produced by this module reflect gradient-weighted feature
activations inside the neural network.  They do **not** constitute
radiological evidence of any anatomical structure or pathology.
Every :class:`ExplainabilityResult` carries a mandatory warning string
to this effect.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import ExplainabilityResult
from preprocessing import PreprocessingResult
from preprocessing.normalization import prepare_for_model

logger = logging.getLogger(__name__)


class GradCAMExplainer:
    """Generate Grad-CAM and Grad-CAM++ explanations for CNN predictions.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to explain.  Must be in eval mode.
    target_layer_name : str
        Dot-notation attribute path of the convolutional layer to hook,
        e.g. ``'features.8'`` for EfficientNet-B4's last conv block.
    device : str | torch.device
        Device on which the model lives.

    Notes
    -----
    The explainer registers forward and backward hooks on
    ``target_layer_name`` at construction time.  Always call
    :meth:`remove_hooks` when finished to avoid memory leaks.

    References
    ----------
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", ICCV 2017.

    Chattopadhyay et al., "Grad-CAM++: Improved Visual Explanations for
    Deep Convolutional Networks", WACV 2018.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer_name: str,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.target_layer_name = target_layer_name
        self.device = torch.device(device)

        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._hooks: List = []

        self._register_hooks()
        logger.debug(
            "GradCAMExplainer initialised on layer '%s'.", target_layer_name
        )

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _get_target_layer(self) -> nn.Module:
        """Resolve dot-notation layer name to the actual module."""
        layer = self.model
        for attr in self.target_layer_name.split("."):
            layer = getattr(layer, attr)
        return layer

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target layer."""
        target = self._get_target_layer()

        def _forward_hook(module: nn.Module, inp, output: torch.Tensor) -> None:
            self._activations = output.detach()

        def _backward_hook(module: nn.Module, grad_in, grad_out: Tuple) -> None:
            self._gradients = grad_out[0].detach()

        self._hooks.append(target.register_forward_hook(_forward_hook))
        self._hooks.append(target.register_full_backward_hook(_backward_hook))

    def remove_hooks(self) -> None:
        """Remove all registered hooks to free resources."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        logger.debug("GradCAMExplainer hooks removed.")

    # ------------------------------------------------------------------
    # Grad-CAM
    # ------------------------------------------------------------------

    def generate_gradcam(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 1,
    ) -> np.ndarray:
        """Compute a Grad-CAM heatmap.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Shape ``(1, 3, H, W)``, already normalized, on the correct device.
        target_class : int
            Class index to explain.  Defaults to ``1`` (OA-related).

        Returns
        -------
        np.ndarray
            Float32 heatmap of shape ``(H, W)``, values in [0, 1].
        """
        self.model.eval()
        self._activations = None
        self._gradients = None

        input_tensor = input_tensor.to(self.device).requires_grad_(True)

        logits = self.model(input_tensor)  # forward pass → hooks fire
        score = logits[0, target_class]

        self.model.zero_grad()
        score.backward()  # backward pass → gradient hooks fire

        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks did not capture activations/gradients. "
                "Check that target_layer_name is correct."
            )

        # Global average pooling of gradients → weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of feature maps
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        return self._postprocess_cam(cam, input_tensor.shape[-2:])

    # ------------------------------------------------------------------
    # Grad-CAM++
    # ------------------------------------------------------------------

    def generate_gradcam_plusplus(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 1,
    ) -> np.ndarray:
        """Compute a Grad-CAM++ heatmap.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Shape ``(1, 3, H, W)``.
        target_class : int
            Class index to explain.

        Returns
        -------
        np.ndarray
            Float32 heatmap of shape ``(H, W)``, values in [0, 1].
        """
        self.model.eval()
        self._activations = None
        self._gradients = None

        input_tensor = input_tensor.to(self.device).requires_grad_(True)

        logits = self.model(input_tensor)
        score = logits[0, target_class]

        self.model.zero_grad()
        score.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "Grad-CAM++ hooks did not capture activations/gradients."
            )

        acts = self._activations  # (1, C, h, w)
        grads = self._gradients   # (1, C, h, w)

        # Grad-CAM++ alpha computation (second-order gradient approximation)
        grads_sq = grads ** 2
        grads_cb = grads ** 3
        sum_acts = acts.sum(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        denom = 2.0 * grads_sq + sum_acts * grads_cb
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha = grads_sq / denom  # (1, C, h, w)

        # Weights: alpha-weighted, ReLU-activated gradients
        weights = (alpha * F.relu(grads)).mean(dim=(2, 3), keepdim=True)

        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        return self._postprocess_cam(cam, input_tensor.shape[-2:])

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _postprocess_cam(
        self,
        cam: torch.Tensor,
        target_hw: Tuple[int, int],
    ) -> np.ndarray:
        """Resize CAM to target size and normalize to [0, 1]."""
        cam_np = cam.squeeze().cpu().numpy().astype(np.float32)

        # Resize to match input image
        h, w = target_hw
        cam_resized = cv2.resize(cam_np, (w, h), interpolation=cv2.INTER_LINEAR)

        # Min-max normalize to [0, 1]
        cam_min, cam_max = cam_resized.min(), cam_resized.max()
        if cam_max - cam_min < 1e-8:
            logger.warning("CAM is nearly constant — heatmap will be blank.")
            return np.zeros((h, w), dtype=np.float32)

        return ((cam_resized - cam_min) / (cam_max - cam_min)).astype(np.float32)

    def create_overlay(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Blend a heatmap over the original grayscale image.

        Parameters
        ----------
        original_image : np.ndarray
            Grayscale image of shape ``(H, W)``, dtype ``uint8``.
        heatmap : np.ndarray
            Float32 heatmap of shape ``(H, W)``, values in [0, 1].
        alpha : float
            Transparency of the heatmap overlay.  ``0`` = invisible,
            ``1`` = fully opaque heatmap.

        Returns
        -------
        np.ndarray
            RGB image of shape ``(H, W, 3)``, dtype ``uint8``.
        """
        # Convert grayscale to BGR
        if original_image.ndim == 2:
            img_bgr = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = original_image.copy()

        # Resize heatmap to image size if needed
        if heatmap.shape != original_image.shape[:2]:
            h, w = original_image.shape[:2]
            heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

        # Apply jet colormap to heatmap
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Blend
        overlay = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_colored, alpha, 0)
        return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB).astype(np.uint8)

    def explain(
        self,
        preprocessed: "PreprocessingResult",
        input_tensor: torch.Tensor,
        method: str = "gradcam",
    ) -> ExplainabilityResult:
        """Generate a complete explainability result.

        Parameters
        ----------
        preprocessed : PreprocessingResult
            Contains the processed image used for overlay generation.
        input_tensor : torch.Tensor
            Shape ``(1, 3, H, W)``, the model input.
        method : str
            ``'gradcam'`` or ``'gradcam++'``.

        Returns
        -------
        ExplainabilityResult
            Contains heatmap, RGB overlay, method, layer name, and warning.
        """
        if method not in {"gradcam", "gradcam++"}:
            raise ValueError(
                f"method must be 'gradcam' or 'gradcam++', got '{method}'"
            )

        try:
            if method == "gradcam":
                heatmap = self.generate_gradcam(input_tensor, target_class=1)
            else:
                heatmap = self.generate_gradcam_plusplus(input_tensor, target_class=1)
        except Exception as exc:
            logger.error("Grad-CAM generation failed: %s", exc)
            # Return blank heatmap rather than crashing the pipeline
            h, w = preprocessed.processed.shape[:2]
            heatmap = np.zeros((h, w), dtype=np.float32)

        overlay = self.create_overlay(preprocessed.processed, heatmap)

        return ExplainabilityResult(
            heatmap=heatmap,
            overlay=overlay,
            method=method,
            target_layer=self.target_layer_name,
            warning="Model explanation only. Not anatomical proof.",
        )
