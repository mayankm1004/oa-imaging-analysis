"""
models/classifier.py
====================
EfficientNet-B4 based binary classifier for OA-related abnormality detection.

Architecture overview
---------------------
* Backbone : EfficientNet-B4 (pretrained on ImageNet-1K)
* Custom head :
    ``[Linear(1792, 512) → BatchNorm1d → Dropout(0.4) → Linear(512, 2)]``
* Output  : 2-class softmax — ``{0: 'normal', 1: 'oa_related'}``

.. warning::
    When ``checkpoint_path=None``, the model is loaded with
    **ImageNet-only weights** and has **not** been validated for OA
    detection.  All predictions in this mode are labelled with
    ``requires_finetuning=True`` and must be treated as unvalidated
    demonstrations.  Do NOT use for clinical decision-making.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import EfficientNet_B4_Weights, efficientnet_b4

from models import ExplainabilityResult, ModelPrediction
from models.base_model import OAModel

# ---------------------------------------------------------------------------
# Lazy import – preprocessing may not be installed in all environments.
# ---------------------------------------------------------------------------
try:
    from preprocessing import PreprocessingResult  # type: ignore[import]
except ImportError:  # pragma: no cover
    PreprocessingResult = Any  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

__all__ = ["EfficientNetOAClassifier"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MODEL_NAME: str = "EfficientNet-B4-OA"
_MODEL_VERSION: str = "1.0.0"

# EfficientNet-B4 MBConv output channels (penultimate before head)
_BACKBONE_OUT_FEATURES: int = 1792
_HEAD_HIDDEN: int = 512
_NUM_CLASSES: int = 2
_DROPOUT: float = 0.4

# ImageNet normalisation statistics
_IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# Input spatial resolution expected by EfficientNet-B4
_INPUT_SIZE: Tuple[int, int] = (380, 380)

# Confidence thresholds
_LOW_THRESH_LO: float = 0.35
_LOW_THRESH_HI: float = 0.65
_MOD_THRESH_LO: float = 0.25
_MOD_THRESH_HI: float = 0.75


def _get_device() -> torch.device:
    """Auto-detect the best available compute device.

    Priority: CUDA → Apple MPS → CPU.

    Returns
    -------
    torch.device
        The selected device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS device.")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device (no GPU available).")
    return device


def _derive_confidence(oa_score: float) -> str:
    """Map a scalar OA score to a human-readable confidence tier.

    Parameters
    ----------
    oa_score : float
        Predicted probability that the image contains OA-related
        abnormality, in [0, 1].

    Returns
    -------
    str
        ``'low'``      when score ∈ [0.35, 0.65]
        ``'moderate'`` when score ∈ [0.25, 0.35) ∪ (0.65, 0.75]
        ``'high'``     when score < 0.25 or > 0.75
    """
    if _LOW_THRESH_LO <= oa_score <= _LOW_THRESH_HI:
        return "low"
    if _MOD_THRESH_LO <= oa_score < _LOW_THRESH_LO or _LOW_THRESH_HI < oa_score <= _MOD_THRESH_HI:
        return "moderate"
    return "high"


class _OAClassifierHead(nn.Module):
    """Custom classification head attached on top of the EfficientNet backbone.

    Architecture
    ------------
    ``Linear(1792, 512) → BatchNorm1d(512) → Dropout(0.4) → Linear(512, 2)``

    Parameters
    ----------
    in_features : int
        Number of features from the backbone's pooled output (1792 for B4).
    hidden_dim : int
        Number of neurons in the intermediate linear layer.
    num_classes : int
        Number of output logits (2 for binary OA detection).
    dropout : float
        Dropout probability applied after BatchNorm.
    """

    def __init__(
        self,
        in_features: int = _BACKBONE_OUT_FEATURES,
        hidden_dim: int = _HEAD_HIDDEN,
        num_classes: int = _NUM_CLASSES,
        dropout: float = _DROPOUT,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Pooled feature tensor of shape ``(B, in_features)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, num_classes)``.
        """
        x = F.relu(self.bn1(self.fc1(x)), inplace=True)
        x = self.dropout(x)
        return self.fc2(x)


class EfficientNetOAClassifier(OAModel):
    """EfficientNet-B4 adapted for OA-related abnormality detection.

    .. important::
        When loaded with ImageNet weights only (``checkpoint_path=None``),
        this model has **NOT** been validated for OA detection.
        Task-specific fine-tuning is required before clinical use.
        All predictions in demo mode are marked with
        ``requires_finetuning=True`` and must not be used for diagnosis.

    Parameters
    ----------
    device : torch.device, optional
        Compute device.  Auto-detected (CUDA → MPS → CPU) when ``None``.

    Attributes
    ----------
    model : nn.Module
        The full EfficientNet-B4 network with custom head, on the
        selected device.
    _feature_maps : List[torch.Tensor]
        Intermediate feature maps captured by forward hook (last conv block).
    _loaded : bool
        Set to ``True`` after :py:meth:`load` completes successfully.
    _using_imagenet_only : bool
        ``True`` when no OA-specific checkpoint has been loaded.
    _checkpoint_path : Optional[str]
        Path to the checkpoint file that was used, or ``None``.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self._config = config or {}
        self.device: torch.device = device or _get_device()
        self.model: Optional[nn.Module] = None
        self._using_imagenet_only: bool = True
        self._checkpoint_path: Optional[str] = None

        # Hook state
        self._feature_maps: List[torch.Tensor] = []
        self._feature_hook_handle: Optional[torch.utils.hooks.RemovableHook] = None

    # ------------------------------------------------------------------
    # OAModel abstract properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Return the canonical model name."""
        return _MODEL_NAME

    @property
    def version(self) -> str:
        """Return the model version string."""
        return _MODEL_VERSION

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load(self, checkpoint_path: Optional[str] = None) -> None:
        """Load model weights.

        Parameters
        ----------
        checkpoint_path : str, optional
            Path to an OA-specific ``.pt`` / ``.pth`` checkpoint.
            When ``None``, ImageNet pretrained weights are used.

        Raises
        ------
        FileNotFoundError
            If ``checkpoint_path`` is provided but does not exist on disk.
        RuntimeError
            If the checkpoint cannot be deserialized.

        Notes
        -----
        The model is placed in ``eval()`` mode after loading, regardless
        of the weight source.
        """
        logger.info("Building EfficientNet-B4 backbone...")
        backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)

        # Replace the default classifier head
        backbone.classifier = _OAClassifierHead(
            in_features=_BACKBONE_OUT_FEATURES,
            hidden_dim=_HEAD_HIDDEN,
            num_classes=_NUM_CLASSES,
            dropout=_DROPOUT,
        )

        if checkpoint_path is not None:
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    f"Checkpoint not found at '{checkpoint_path}'. "
                    "Please provide a valid path or pass checkpoint_path=None "
                    "to use ImageNet weights for demonstration purposes."
                )
            logger.info("Loading OA-specific checkpoint: %s", checkpoint_path)
            try:
                state = torch.load(checkpoint_path, map_location=self.device)
                # Support both raw state_dict and {'model_state_dict': ...} checkpoints
                if isinstance(state, dict) and "model_state_dict" in state:
                    state = state["model_state_dict"]
                backbone.load_state_dict(state, strict=True)
                self._using_imagenet_only = False
                self._checkpoint_path = checkpoint_path
                logger.info("OA-specific checkpoint loaded successfully.")
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load checkpoint '{checkpoint_path}': {exc}"
                ) from exc
        else:
            self._using_imagenet_only = True
            self._checkpoint_path = None
            logger.warning(
                "═══════════════════════════════════════════════════════════\n"
                "  DEMO MODE – ImageNet weights ONLY.\n"
                "  This model has NOT been fine-tuned for OA detection.\n"
                "  All predictions are marked requires_finetuning=True.\n"
                "  DO NOT use output for clinical decision-making.\n"
                "═══════════════════════════════════════════════════════════"
            )

        self.model = backbone.to(self.device)
        self.model.eval()

        # Register feature-map capture hook on the last conv block
        self._register_feature_hook()

        self._loaded = True
        logger.info(
            "EfficientNet-B4 loaded on %s (ImageNet-only=%s).",
            self.device,
            self._using_imagenet_only,
        )

    # ------------------------------------------------------------------
    # Pre-processing helpers
    # ------------------------------------------------------------------

    def prepare_for_model(self, image: np.ndarray) -> torch.Tensor:
        """Convert a preprocessed grayscale image array to a model input tensor.

        The image is:
        1. Resized to ``(380, 380)`` using bilinear interpolation.
        2. Replicated across 3 channels (EfficientNet expects RGB).
        3. Normalised with ImageNet mean/std.
        4. Batched to shape ``(1, 3, 380, 380)`` and moved to device.

        Parameters
        ----------
        image : np.ndarray
            Preprocessed image of shape ``(H, W)`` or ``(H, W, 1)``,
            dtype ``float32``, values in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Batch tensor of shape ``(1, 3, 380, 380)`` on ``self.device``.

        Raises
        ------
        ValueError
            If ``image`` has an unexpected number of dimensions or channels.
        """
        if image.ndim == 3 and image.shape[2] == 1:
            image = image.squeeze(2)
        elif image.ndim == 3 and image.shape[2] == 3:
            # Already RGB – convert to grey for consistency
            image = np.mean(image, axis=2).astype(np.float32)
        elif image.ndim != 2:
            raise ValueError(
                f"Expected a 2-D (H,W) or 3-D (H,W,C) image, got shape {image.shape}."
            )

        # Ensure float32 in [0, 1]
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0

        # Resize to expected input size
        h_target, w_target = _INPUT_SIZE
        resized = cv2.resize(image, (w_target, h_target), interpolation=cv2.INTER_LINEAR)

        # Stack to 3 channels
        rgb = np.stack([resized, resized, resized], axis=0)  # (3, H, W)

        # ImageNet normalisation per channel
        mean = np.array(_IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
        std = np.array(_IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
        rgb = (rgb - mean) / std

        tensor = torch.from_numpy(rgb).unsqueeze(0)  # (1, 3, H, W)
        return tensor.to(self.device)

    # ------------------------------------------------------------------
    # Forward hook for feature extraction
    # ------------------------------------------------------------------

    def _register_feature_hook(self) -> None:
        """Register a forward hook on EfficientNet-B4's last conv block.

        The hook stores the output feature maps in ``self._feature_maps``
        for use by :class:`~models.explainability.GradCAMExplainer`.
        """
        if self._feature_hook_handle is not None:
            self._feature_hook_handle.remove()

        def _hook(
            module: nn.Module,  # noqa: ARG001
            input: Any,          # noqa: ARG001
            output: torch.Tensor,
        ) -> None:
            self._feature_maps = [output.detach()]

        # EfficientNet-B4: features[-1] is the last MBConv block output
        target_module = self.model.features[-1]  # type: ignore[index]
        self._feature_hook_handle = target_module.register_forward_hook(_hook)
        logger.debug("Feature-map hook registered on 'features[-1]'.")

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks.

        Call this when the model instance is no longer needed to avoid
        memory leaks from dangling hook handles.
        """
        if self._feature_hook_handle is not None:
            self._feature_hook_handle.remove()
            self._feature_hook_handle = None
            logger.debug("Feature-map hook removed.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, preprocessed: PreprocessingResult) -> ModelPrediction:
        """Run inference on a preprocessed image.

        Parameters
        ----------
        preprocessed : PreprocessingResult
            Preprocessing pipeline output.  ``preprocessed.processed``
            must be a ``float32`` array with values in ``[0, 1]``.

        Returns
        -------
        ModelPrediction
            Populated prediction object.  ``requires_finetuning`` is
            ``True`` whenever ImageNet-only weights are active.

        Raises
        ------
        RuntimeError
            If :py:meth:`load` has not been called.
        """
        self._require_loaded()

        input_tensor = self.prepare_for_model(preprocessed.processed)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)           # (1, 2)
            probs = F.softmax(logits, dim=1).squeeze()  # (2,)

        oa_score = float(probs[1].cpu().item())
        normal_score = float(probs[0].cpu().item())

        # Feature vector from the penultimate pooled output
        feature_vector = self._extract_feature_vector(input_tensor)

        confidence = _derive_confidence(oa_score)

        return ModelPrediction(
            oa_score=oa_score,
            confidence_level=confidence,
            raw_probabilities={"normal": normal_score, "oa_related": oa_score},
            feature_vector=feature_vector,
            model_name=self.name,
            model_version=self.version,
            requires_finetuning=self._using_imagenet_only,
        )

    def _extract_feature_vector(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Extract the pooled penultimate-layer feature vector.

        The EfficientNet features block output is globally average-pooled
        to produce a 1-D embedding vector.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The batch tensor already on ``self.device``.

        Returns
        -------
        np.ndarray
            1-D feature vector of shape ``(1792,)``.
        """
        with torch.no_grad():
            feat_out = self.model.features(input_tensor)      # (1, 1792, h, w)
            pooled = self.model.avgpool(feat_out)             # (1, 1792, 1, 1)
            vector = pooled.flatten(start_dim=1).cpu().numpy()  # (1, 1792)
        return vector.squeeze(0)  # (1792,)

    # ------------------------------------------------------------------
    # Explainability (delegates to GradCAMExplainer)
    # ------------------------------------------------------------------

    def explain_prediction(
        self,
        preprocessed: PreprocessingResult,
        prediction: ModelPrediction,
    ) -> ExplainabilityResult:
        """Generate a Grad-CAM explanation for the given prediction.

        Delegates to :class:`~models.explainability.GradCAMExplainer`
        targeting the last convolutional block (``'features.8'``).

        Parameters
        ----------
        preprocessed : PreprocessingResult
            The preprocessed image corresponding to ``prediction``.
        prediction : ModelPrediction
            The prediction whose class will be used as the Grad-CAM target.

        Returns
        -------
        ExplainabilityResult
            Heatmap and overlay with the mandatory clinical disclaimer.

        Raises
        ------
        RuntimeError
            If :py:meth:`load` has not been called.
        """
        self._require_loaded()

        # Import here to avoid circular imports at module level
        from models.explainability import GradCAMExplainer  # noqa: PLC0415

        target_layer = "features.8"
        explainer = GradCAMExplainer(model=self.model, target_layer=target_layer)
        try:
            input_tensor = self.prepare_for_model(preprocessed.processed)
            result = explainer.explain(
                preprocessed=preprocessed,
                input_tensor=input_tensor,
                method="gradcam",
            )
        finally:
            explainer.remove_hooks()

        return result

    # ------------------------------------------------------------------
    # OAModel.get_model_info override
    # ------------------------------------------------------------------

    def get_model_info(self) -> Dict[str, Any]:
        """Return an extended info dictionary including device and weight source.

        Returns
        -------
        dict
            All keys from :py:meth:`OAModel.get_model_info` plus:
            ``device``, ``backbone``, ``input_size``,
            ``using_imagenet_only``, ``checkpoint_path``.
        """
        info = super().get_model_info()
        info.update(
            {
                "device": str(self.device),
                "backbone": "efficientnet_b4",
                "input_size": _INPUT_SIZE,
                "using_imagenet_only": self._using_imagenet_only,
                "checkpoint_path": self._checkpoint_path,
                "num_classes": _NUM_CLASSES,
                "class_labels": {0: "normal", 1: "oa_related"},
            }
        )
        return info

    # ------------------------------------------------------------------
    # Public convenience accessors
    # ------------------------------------------------------------------

    def is_loaded(self) -> bool:
        """Return ``True`` if :meth:`load` has been called successfully."""
        return getattr(self, "_loaded", False) and self.model is not None

    @property
    def backbone(self) -> nn.Module:
        """Return the underlying torchvision EfficientNet-B4 model.

        This exposes the full model (backbone + head) for direct use
        in GradCAMExplainer.
        """
        self._require_loaded()
        return self.model  # type: ignore[return-value]

    @property
    def gradcam_target_layer(self) -> str:
        """Return the dot-notation name of the Grad-CAM target layer.

        For EfficientNet-B4 this is ``'features.8'`` — the final
        inverted-residual block, which produces the most semantically
        rich feature maps.
        """
        return "features.8"

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def __del__(self) -> None:
        """Clean up registered hooks on garbage collection."""
        try:
            self.remove_hooks()
        except Exception:  # noqa: BLE001
            pass
