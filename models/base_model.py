"""
models/base_model.py
====================
Abstract base class that every OA detection model must implement.

All concrete model classes (classifiers, segmenters, multi-modal networks)
should inherit from :class:`OAModel` and implement every abstract method.
This guarantees a uniform interface for the inference pipeline, evaluation
harness, and Gradio application layer.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from models import ExplainabilityResult, ModelPrediction

# ---------------------------------------------------------------------------
# Lazy import to avoid a hard dependency at module level – preprocessing may
# not be on the path in all deployment contexts.
# ---------------------------------------------------------------------------
try:
    from preprocessing import PreprocessingResult  # type: ignore[import]
except ImportError:  # pragma: no cover
    PreprocessingResult = Any  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

__all__ = ["OAModel"]


class OAModel(ABC):
    """Abstract base class for all OA-related detection / classification models.

    Sub-classes must implement the four abstract methods:
    :py:meth:`name`, :py:meth:`version`, :py:meth:`load`,
    :py:meth:`predict`, and :py:meth:`explain_prediction`.

    The class manages a simple ``_loaded`` flag that is set to ``True``
    inside :py:meth:`load` once weights have been successfully applied.
    Concrete implementations should call ``super().__init__()`` and then
    set ``self._loaded = True`` at the end of their ``load()`` method.

    Usage
    -----
    .. code-block:: python

        class MyModel(OAModel):
            @property
            def name(self) -> str:
                return "MyModel"

            @property
            def version(self) -> str:
                return "1.0.0"

            def load(self, checkpoint_path=None):
                ...
                self._loaded = True

            def predict(self, preprocessed):
                ...

            def explain_prediction(self, preprocessed, prediction):
                ...

        model = MyModel()
        model.load()
        result = model.predict(preprocessed_image)
    """

    def __init__(self) -> None:
        """Initialise shared state.  Sub-classes must call ``super().__init__()``."""
        self._loaded: bool = False
        logger.debug("Initialised OAModel base; waiting for load().")

    # ------------------------------------------------------------------
    # Abstract properties
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical human-readable name of this model (e.g. ``'EfficientNet-B4'``)."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Semantic version string (e.g. ``'1.0.0'``)."""
        ...

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def load(self, checkpoint_path: Optional[str] = None) -> None:
        """Load model weights into memory and prepare for inference.

        Parameters
        ----------
        checkpoint_path : str, optional
            Path to a ``.pt`` / ``.pth`` checkpoint file produced by the
            training pipeline.  When ``None``, the model falls back to
            ImageNet-pretrained weights (where available) and sets
            ``requires_finetuning = True`` on all subsequent predictions.

        Raises
        ------
        FileNotFoundError
            If ``checkpoint_path`` is provided but the file does not exist.
        RuntimeError
            If loading fails for any other reason (corrupt file, version
            mismatch, etc.).

        Notes
        -----
        * Implementations **must** set ``self._loaded = True`` on success.
        * Implementations should place the model in ``eval()`` mode after
          loading weights.
        * When falling back to ImageNet weights, a prominent warning must
          be emitted via the :mod:`logging` module.
        """
        ...

    @abstractmethod
    def predict(self, preprocessed: PreprocessingResult) -> ModelPrediction:
        """Run a forward pass and return a structured prediction.

        Parameters
        ----------
        preprocessed : PreprocessingResult
            Output from the preprocessing pipeline.  The field
            ``preprocessed.processed`` contains the normalised image
            array (H x W, float32, values in [0, 1]) ready for the
            model.

        Returns
        -------
        ModelPrediction
            Fully populated prediction object.  ``requires_finetuning``
            must be ``True`` whenever ImageNet weights are in use.

        Raises
        ------
        RuntimeError
            If called before :py:meth:`load`.
        ValueError
            If ``preprocessed.processed`` has an incompatible shape.
        """
        ...

    @abstractmethod
    def explain_prediction(
        self,
        preprocessed: PreprocessingResult,
        prediction: ModelPrediction,
    ) -> ExplainabilityResult:
        """Generate a Grad-CAM explainability heatmap for a prediction.

        Parameters
        ----------
        preprocessed : PreprocessingResult
            The same preprocessed image that was passed to
            :py:meth:`predict`.
        prediction : ModelPrediction
            The :class:`ModelPrediction` returned by :py:meth:`predict`
            for this image.  Used to select the target class for the
            backward pass.

        Returns
        -------
        ExplainabilityResult
            Heatmap and overlay with the mandatory clinical disclaimer.

        Raises
        ------
        RuntimeError
            If called before :py:meth:`load`.
        """
        ...

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def is_loaded(self) -> bool:
        """Return ``True`` if the model has been successfully loaded.

        Returns
        -------
        bool
            Whether :py:meth:`load` has been called and completed
            without error.
        """
        return self._loaded

    def get_model_info(self) -> Dict[str, Any]:
        """Return a summary dictionary describing this model instance.

        Returns
        -------
        dict
            Keys: ``name``, ``version``, ``is_loaded``, ``class``.
            Concrete subclasses should call ``super().get_model_info()``
            and merge in their own additional keys (e.g. device, backbone
            architecture, checkpoint path).

        Example
        -------
        .. code-block:: python

            info = model.get_model_info()
            print(info)
            # {'name': 'EfficientNet-B4', 'version': '1.0.0',
            #  'is_loaded': True, 'class': 'EfficientNetOAClassifier'}
        """
        return {
            "name": self.name,
            "version": self.version,
            "is_loaded": self._loaded,
            "class": type(self).__name__,
        }

    def _require_loaded(self) -> None:
        """Raise ``RuntimeError`` if the model has not been loaded yet.

        Intended as a guard at the top of :py:meth:`predict` and
        :py:meth:`explain_prediction`.

        Raises
        ------
        RuntimeError
            If ``self._loaded`` is ``False``.
        """
        if not self._loaded:
            raise RuntimeError(
                f"{type(self).__name__} is not loaded.  Call load() first."
            )
