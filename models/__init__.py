"""
models/__init__.py
==================
Public API for the OA-detection models package.

Exports
-------
ModelPrediction
    Structured output from any ``OAModel.predict()`` call.
ExplainabilityResult
    Structured output from any ``OAModel.explain_prediction()`` call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

__all__ = ["ModelPrediction", "ExplainabilityResult"]


@dataclass
class ModelPrediction:
    """Structured prediction result returned by every OA model.

    Attributes
    ----------
    oa_score : float
        Probability in [0, 1] that the image contains OA-related
        abnormality (i.e. the softmax probability for class index 1).
    confidence_level : str
        Human-readable confidence tier derived from ``oa_score``:

        * ``'low'``      – score in [0.35, 0.65]
        * ``'moderate'`` – score in [0.25, 0.35) or (0.65, 0.75]
        * ``'high'``     – score < 0.25 or > 0.75

    raw_probabilities : Dict[str, float]
        Class-level softmax probabilities, e.g.
        ``{'normal': 0.22, 'oa_related': 0.78}``.
    feature_vector : Optional[np.ndarray]
        Penultimate-layer feature embedding (shape ``(D,)``).
        ``None`` when feature extraction is not supported.
    model_name : str
        Canonical name of the model that produced this prediction.
    model_version : str
        Version string of the model (e.g. ``'1.0.0'``).
    requires_finetuning : bool
        ``True`` when the model was loaded with ImageNet weights only
        and has **not** been fine-tuned on OA data.  All predictions
        from such a model must be treated as unvalidated demonstrations.

    Notes
    -----
    When ``requires_finetuning`` is ``True`` the prediction **must not**
    be used for clinical decision-making.  Task-specific fine-tuning and
    prospective validation are required before any clinical use.
    """

    oa_score: float
    confidence_level: str
    raw_probabilities: Dict[str, float]
    feature_vector: Optional[np.ndarray]
    model_name: str
    model_version: str
    requires_finetuning: bool

    def __post_init__(self) -> None:
        """Validate field ranges at construction time."""
        if not (0.0 <= self.oa_score <= 1.0):
            raise ValueError(
                f"oa_score must be in [0, 1], got {self.oa_score:.4f}"
            )
        valid_levels = {"low", "moderate", "high"}
        if self.confidence_level not in valid_levels:
            raise ValueError(
                f"confidence_level must be one of {valid_levels}, "
                f"got '{self.confidence_level}'"
            )

    def to_dict(self) -> Dict:
        """Serialise to a plain dictionary (JSON-safe, except ndarray)."""
        return {
            "oa_score": self.oa_score,
            "confidence_level": self.confidence_level,
            "raw_probabilities": self.raw_probabilities,
            "feature_vector": (
                self.feature_vector.tolist()
                if self.feature_vector is not None
                else None
            ),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "requires_finetuning": self.requires_finetuning,
        }


@dataclass
class ExplainabilityResult:
    """Structured explainability output from Grad-CAM / Grad-CAM++.

    Attributes
    ----------
    heatmap : np.ndarray
        Activation heatmap of shape ``(H, W)``, dtype ``float32``,
        values normalised to [0, 1].  Higher values indicate regions
        that most influenced the model's predicted class score.
    overlay : np.ndarray
        RGB overlay of shape ``(H, W, 3)``, dtype ``uint8``.
        The heatmap colourmap (jet) is blended with the original
        greyscale image for visual inspection.
    method : str
        Name of the explanation method used: ``'gradcam'`` or
        ``'gradcam++'``.
    target_layer : str
        Dot-notation name of the convolutional layer used for hooks,
        e.g. ``'features.8'``.
    warning : str
        **Always** set to the fixed disclaimer string:
        ``'Model explanation only. Not anatomical proof.'``

    Notes
    -----
    Grad-CAM heatmaps reflect gradient-weighted feature activations
    inside the neural network.  They do **not** constitute radiological
    evidence of any anatomical structure or pathology.
    """

    heatmap: np.ndarray
    overlay: np.ndarray
    method: str
    target_layer: str
    warning: str = field(
        default="Model explanation only. Not anatomical proof."
    )

    def __post_init__(self) -> None:
        """Validate shapes and enforce the mandatory warning string."""
        if self.heatmap.ndim != 2:
            raise ValueError(
                f"heatmap must be 2-D (H, W), got shape {self.heatmap.shape}"
            )
        if self.overlay.ndim != 3 or self.overlay.shape[2] != 3:
            raise ValueError(
                f"overlay must be 3-D (H, W, 3), got shape {self.overlay.shape}"
            )
        if self.method not in {"gradcam", "gradcam++"}:
            raise ValueError(
                f"method must be 'gradcam' or 'gradcam++', got '{self.method}'"
            )
        # Enforce the mandatory disclaimer – callers must not alter it.
        self.warning = "Model explanation only. Not anatomical proof."

    def to_dict(self) -> Dict:
        """Serialise to a plain dictionary (JSON-safe, except ndarrays)."""
        return {
            "heatmap": self.heatmap.tolist(),
            "overlay": self.overlay.tolist(),
            "method": self.method,
            "target_layer": self.target_layer,
            "warning": self.warning,
        }
