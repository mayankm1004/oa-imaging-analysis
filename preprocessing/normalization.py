"""
preprocessing/normalization.py
================================
Image normalization utilities for OA detection.

Provides multiple normalization strategies:
- ImageNet mean/std normalization (for pretrained backbone compatibility)
- Z-score normalization
- Min-max normalization
- Model-ready tensor preparation
"""

from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ImageNet statistics (RGB order)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def imagenet_normalize(image: np.ndarray) -> np.ndarray:
    """Apply ImageNet mean/std normalization to a grayscale X-ray image.

    Converts a single-channel uint8 image to a 3-channel float32 array
    normalized with ImageNet statistics for compatibility with pretrained
    torchvision backbones.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image of shape ``(H, W)``, dtype ``uint8``.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(H, W, 3)`` with ImageNet normalization
        applied.  Values are **not** clipped after normalization.

    Notes
    -----
    X-ray images are inherently grayscale; they are replicated across three
    channels to satisfy the input contract of pretrained RGB models.
    ImageNet normalization here is a pragmatic approximation — it does
    **not** mean the model has been validated on X-ray data.
    """
    if image.ndim == 3:
        # Already 3-channel; convert to grayscale first for consistency
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Replicate single channel to 3 channels
    image_rgb = np.stack([image, image, image], axis=-1).astype(np.float32) / 255.0

    normalized = (image_rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    return normalized.astype(np.float32)


def zscore_normalize(image: np.ndarray) -> np.ndarray:
    """Apply per-image Z-score normalization.

    Parameters
    ----------
    image : np.ndarray
        Image of any shape and numeric dtype.

    Returns
    -------
    np.ndarray
        Float32 array of the same shape with zero mean and unit variance.
        If the image has zero standard deviation (constant), returns a
        zero array of the same shape.
    """
    arr = image.astype(np.float32)
    mean = arr.mean()
    std = arr.std()
    if std < 1e-8:
        logger.warning(
            "zscore_normalize: image has near-zero std (%.6f); returning zeros.", std
        )
        return np.zeros_like(arr)
    return (arr - mean) / std


def minmax_normalize(image: np.ndarray) -> np.ndarray:
    """Normalize image pixel values to the range [0, 1].

    Parameters
    ----------
    image : np.ndarray
        Image of any shape and numeric dtype.

    Returns
    -------
    np.ndarray
        Float32 array of the same shape with values in [0, 1].
        If ``min == max`` (constant image), returns a zero array.
    """
    arr = image.astype(np.float32)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        logger.warning(
            "minmax_normalize: image is constant (min=%.2f, max=%.2f); "
            "returning zeros.",
            lo,
            hi,
        )
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def prepare_for_model(
    image: np.ndarray,
    target_size: Tuple[int, int],
) -> np.ndarray:
    """Prepare a preprocessed grayscale image for PyTorch model inference.

    Steps:
    1. Resize to ``target_size`` if needed (LANCZOS interpolation).
    2. Replicate single channel → 3 channels.
    3. Apply ImageNet normalization.
    4. Add batch dimension → shape ``(1, 3, H, W)``.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image of shape ``(H, W)``, dtype ``uint8``.
    target_size : Tuple[int, int]
        ``(height, width)`` target dimensions expected by the model.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(1, 3, H, W)`` ready for
        ``torch.from_numpy()`` conversion.
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = target_size
    if image.shape != (h, w):
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # (H, W) → (H, W, 3)
    img_rgb = np.stack([image, image, image], axis=-1).astype(np.float32) / 255.0

    # Apply ImageNet normalization
    normalized = (img_rgb - _IMAGENET_MEAN) / _IMAGENET_STD  # (H, W, 3)

    # (H, W, 3) → (3, H, W) → (1, 3, H, W)
    chw = np.transpose(normalized, (2, 0, 1))
    return np.expand_dims(chw, axis=0).astype(np.float32)
