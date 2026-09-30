"""
inference/predictor.py
======================
Top-level OA analysis pipeline orchestrator.

:class:`OAPredictor` wires together every stage of the pipeline:

1. Image loading and validation
2. Preprocessing (via ``preprocessing/``)
3. Joint-region segmentation (via ``models/``)
4. Model inference (via ``models/``)
5. Risk-marker extraction (via :class:`~inference.risk_markers.RiskMarkerExtractor`)
6. Explainability / Grad-CAM (via ``models/``)
7. Result assembly into :class:`~inference.OAAnalysisResult`

Usage
-----
.. code-block:: python

    predictor = OAPredictor(config_path='configs/oa_model.yaml')
    result = predictor.analyze('path/to/xray.png', joint_type='knee')
    print(result.summary())
    print(result.to_json())

.. warning::
    Outputs from this system are for **research / investigational use only**.
    They must **never** be used as the sole basis for a clinical decision.
"""

from __future__ import annotations

import datetime
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from inference import OAAnalysisResult, DISCLAIMER
from inference.risk_markers import RiskMarkerExtractor
from preprocessing import PreprocessingResult
from models import ModelPrediction, ExplainabilityResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default configuration (used when YAML values are absent)
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "name": "EfficientNetOAClassifier",
        "version": "1.0.0",
        "checkpoint": "models/best_model.pth",
        "input_size": [224, 224],
        "num_classes": 2,
        "device": "auto",
    },
    "preprocessing": {
        "study_type": "xray",
        "modality": "xray",
    },
    "risk_markers": {
        "model_weight": 0.60,
    },
    "inference": {
        "confidence_thresholds": {
            "high": 0.80,
            "moderate": 0.60,
            "low": 0.40,
        },
        "image_quality": {
            "min_mean_intensity": 20,
            "max_mean_intensity": 235,
            "min_std_intensity": 10,
        },
    },
}


class OAPredictor:
    """Orchestrates the full OA analysis pipeline.

    Coordinates: image loading → preprocessing → joint segmentation →
    model inference → risk marker extraction → explainability →
    result assembly.

    All heavy components (model, preprocessor, segmenter) are loaded once at
    construction time and reused across calls to :meth:`analyze`.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.  Defaults to
        ``'configs/oa_model.yaml'``.

    Raises
    ------
    FileNotFoundError
        If ``config_path`` does not exist.
    RuntimeError
        If a required pipeline component cannot be initialised.

    Example
    -------
    .. code-block:: python

        predictor = OAPredictor('configs/oa_model.yaml')
        result = predictor.analyze('knee_xray.png', joint_type='knee')
    """

    def __init__(self, config_path: str = "configs/oa_model.yaml", checkpoint_path: Optional[str] = None) -> None:
        self._config: Dict[str, Any] = self._load_config(config_path)
        self._checkpoint_path = checkpoint_path
        self._init_components()
        logger.info(
            "OAPredictor ready | model=%s",
            self._config["model"]["name"],
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyze(
        self,
        image_path: str,
        joint_type: str = "knee",
        run_gradcam: bool = True,
        gradcam_method: str = "gradcam",
    ) -> OAAnalysisResult:

        """Run the full OA analysis pipeline on a single image.

        Parameters
        ----------
        image_path : str
            Absolute or relative path to the input image
            (DICOM / JPEG / PNG / TIFF supported, depending on preprocessor).
        joint_type : str
            Anatomical joint; e.g. ``'knee'``, ``'hip'``, ``'hand'``.
            Passed through to the result for provenance.
        run_gradcam : bool
            Whether to generate a Grad-CAM explainability heatmap.  Adds
            ~100–300 ms per image depending on hardware.

        Returns
        -------
        OAAnalysisResult
            Fully populated result object, always including a disclaimer.

        Raises
        ------
        FileNotFoundError
            If the image file does not exist.
        ValueError
            If the image cannot be decoded or is in an unsupported format.
        """
        image_path = str(image_path)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        all_warnings: List[str] = []

        # ------- Stage 1: Preprocessing -------
        logger.info("[1/5] Preprocessing | path=%s", image_path)
        try:
            preprocessed: PreprocessingResult = self._preprocessor.load(image_path)
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(
                f"Preprocessing failed for '{image_path}': {exc}"
            ) from exc

        all_warnings.extend(preprocessed.warnings)

        # ------- Stage 2: Image quality validation -------
        logger.info("[2/5] Image quality check")
        quality_warnings = self._validate_image_quality(preprocessed)
        all_warnings.extend(quality_warnings)

        # ------- Stage 3: Segmentation -------
        logger.info("[3/5] Joint region segmentation")
        try:
            roi_image, roi_bbox = self._segmenter.segment(preprocessed.processed)
            if roi_image is not None:
                preprocessed.roi_image = roi_image
                preprocessed.roi_bbox = roi_bbox
        except Exception as exc:  # pylint: disable=broad-except
            msg = f"Segmentation failed (non-fatal): {exc}"
            logger.warning(msg)
            all_warnings.append(msg)

        # ------- Stage 4: Model inference -------
        logger.info("[4/5] Model inference")
        try:
            prediction: ModelPrediction = self._model.predict(preprocessed)
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(
                f"Model inference failed for '{image_path}': {exc}"
            ) from exc

        if prediction.requires_finetuning:
            all_warnings.append(
                "Model indicates it requires task-specific fine-tuning. "
                "OA score and risk markers should be interpreted with caution."
            )

        # ------- Stage 4b: Risk marker extraction -------
        logger.info("[4b/5] Risk marker extraction")
        risk_markers = self._risk_extractor.extract(
            preprocessed, prediction, self._segmenter
        )

        # ------- Stage 5: Explainability -------
        logger.info("[5/5] Explainability | run_gradcam=%s", run_gradcam)
        if run_gradcam:
            try:
                explainability: ExplainabilityResult = self._model.explain_prediction(
                    preprocessed, prediction
                )
            except Exception as exc:  # pylint: disable=broad-except
                msg = f"Grad-CAM failed (non-fatal): {exc}"
                logger.warning(msg)
                all_warnings.append(msg)
                explainability = self._null_explainability(preprocessed)
        else:
            explainability = self._null_explainability(preprocessed)


        # ------- Assembly -------
        all_warnings.append(
            "RESEARCH USE ONLY: All outputs require review by a qualified clinician."
        )

        result = OAAnalysisResult(
            preprocessing=preprocessed,
            prediction=prediction,
            risk_markers=risk_markers,
            explainability=explainability,
            kl_grade=None,  # Only set for KL-grading fine-tuned models
            kl_grade_confidence=None,
            requires_clinical_review=True,
            warnings=all_warnings,
            study_type=preprocessed.study_type,
            joint_type=joint_type,
            timestamp=timestamp,
            model_name=self._config["model"]["name"],
            model_version=self._config["model"]["version"],
            disclaimer=DISCLAIMER,
        )

        logger.info(
            "Analysis complete | oa_score=%.4f | confidence=%s | warnings=%d",
            result.prediction.oa_score,
            result.prediction.confidence_level,
            len(result.warnings),
        )
        return result

    def analyze_batch(
        self,
        image_paths: List[str],
        joint_type: str = "knee",
        run_gradcam: bool = False,
    ) -> List[OAAnalysisResult]:
        """Run the full pipeline on a batch of images.

        Grad-CAM is disabled by default in batch mode to improve throughput.
        Results are returned in the same order as the input paths.  Failed
        images are logged and a partial :class:`OAAnalysisResult` with error
        information is **not** included; instead the exception is re-raised
        unless ``ignore_errors`` is set (see note below).

        Parameters
        ----------
        image_paths : List[str]
            Ordered list of paths to process.
        joint_type : str
            Anatomical joint type, applied uniformly to all images.
        run_gradcam : bool
            Whether to run Grad-CAM.  Defaults to ``False`` for batch speed.

        Returns
        -------
        List[OAAnalysisResult]
            Results in the same order as ``image_paths``.  If an image fails,
            it is skipped and a warning is logged; the returned list may be
            shorter than the input list.

        .. note::
            To force strict mode (raise on any error), wrap in a try/except
            and pass a single-element list.
        """
        results: List[OAAnalysisResult] = []
        n = len(image_paths)
        logger.info("Starting batch analysis | n_images=%d | gradcam=%s", n, run_gradcam)

        for i, path in enumerate(image_paths):
            logger.info("Batch [%d/%d] | %s", i + 1, n, path)
            try:
                result = self.analyze(
                    image_path=path,
                    joint_type=joint_type,
                    run_gradcam=run_gradcam,
                )
                results.append(result)
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("Batch [%d/%d] FAILED | %s | %s", i + 1, n, path, exc)

        logger.info(
            "Batch complete | processed=%d / %d", len(results), n
        )
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration, falling back to defaults for missing keys.

        Parameters
        ----------
        config_path : str
            Path to ``oa_model.yaml``.

        Returns
        -------
        dict
            Merged configuration dictionary.

        Raises
        ------
        FileNotFoundError
            If the config file does not exist.
        """
        path = Path(config_path)
        if not path.is_file():
            logger.warning(
                "Config not found at '%s'. Using built-in defaults.", config_path
            )
            return _DEFAULT_CONFIG.copy()

        with path.open("r") as fh:
            user_cfg = yaml.safe_load(fh) or {}

        # Deep-merge user config over defaults
        merged = _deep_merge(_DEFAULT_CONFIG, user_cfg)
        logger.info("Configuration loaded from '%s'", config_path)
        return merged

    def _init_components(self) -> None:
        """Initialise all pipeline components.

        Imports are deferred here to keep module-level load fast and to
        allow each component's package to be optional if not installed.

        Components initialised:

        - ``self._preprocessor`` – image preprocessor
        - ``self._segmenter``    – joint-region segmenter
        - ``self._model``        – EfficientNet OA classifier
        - ``self._risk_extractor`` – risk marker extractor

        Raises
        ------
        RuntimeError
            If the model checkpoint cannot be loaded.
        """
        # --- Preprocessor ---
        try:
            from preprocessing.image_loader import ImageLoader

            self._preprocessor = ImageLoader(config=self._config)
            logger.debug("ImageLoader initialised.")
        except ImportError as exc:
            raise RuntimeError(
                "Could not import ImageLoader from 'preprocessing.image_loader'. "
                "Ensure the preprocessing package is installed."
            ) from exc

        # --- Segmenter ---
        try:
            from models.segmentation import JointRegionSegmenter

            self._segmenter = JointRegionSegmenter()
            logger.debug("Segmenter initialised.")
        except ImportError as exc:
            raise RuntimeError(
                "Could not import JointRegionSegmenter from 'models.segmentation'."
            ) from exc

        # --- Model ---
        try:
            from models.classifier import EfficientNetOAClassifier

            self._model = EfficientNetOAClassifier(config=self._config)
            checkpoint = getattr(self, "_checkpoint_path", None) or self._config.get("model", {}).get("checkpoint")
            self._model.load(checkpoint_path=checkpoint)
            logger.debug("Model '%s' loaded.", self._config["model"]["name"])
        except ImportError as exc:
            raise RuntimeError(
                "Could not import EfficientNetOAClassifier from 'models.classifier'."
            ) from exc

        # --- Risk marker extractor ---
        self._risk_extractor = RiskMarkerExtractor(self._config)
        logger.debug("RiskMarkerExtractor initialised.")


    def _validate_image_quality(
        self, preprocessed: PreprocessingResult
    ) -> List[str]:
        """Check basic image quality metrics and return a list of warnings.

        Checks performed:

        - Mean pixel intensity within plausible range (avoids over/under-exposed).
        - Standard deviation of pixel intensity above a minimum (avoids blank images).
        - Image dimensions above a minimum threshold.

        Parameters
        ----------
        preprocessed : PreprocessingResult
            Output from the preprocessing stage.

        Returns
        -------
        List[str]
            Zero or more human-readable warning strings.
        """
        warnings: List[str] = []
        qcfg = self._config.get("inference", {}).get("image_quality", {})
        img = preprocessed.processed

        # Intensity range check
        mean_i = float(np.mean(img))
        min_mean = float(qcfg.get("min_mean_intensity", 20))
        max_mean = float(qcfg.get("max_mean_intensity", 235))
        if mean_i < min_mean:
            warnings.append(
                f"Image appears underexposed (mean intensity={mean_i:.1f} < {min_mean}). "
                "Analysis quality may be reduced."
            )
        if mean_i > max_mean:
            warnings.append(
                f"Image appears overexposed (mean intensity={mean_i:.1f} > {max_mean}). "
                "Analysis quality may be reduced."
            )

        # Contrast check
        std_i = float(np.std(img))
        min_std = float(qcfg.get("min_std_intensity", 10))
        if std_i < min_std:
            warnings.append(
                f"Image has very low contrast (std={std_i:.1f} < {min_std}). "
                "The image may be blank or corrupted."
            )

        # Dimension check
        h, w = img.shape[:2]
        if h < 64 or w < 64:
            warnings.append(
                f"Image is very small ({w}×{h} px). Results may be unreliable."
            )

        return warnings

    @staticmethod
    def _null_explainability(preprocessed: PreprocessingResult) -> ExplainabilityResult:
        """Return a placeholder :class:`ExplainabilityResult` when Grad-CAM is skipped.

        Parameters
        ----------
        preprocessed : PreprocessingResult
            Used to determine the output image shape.

        Returns
        -------
        ExplainabilityResult
            Zero-valued heatmap with an informative warning.
        """
        h, w = preprocessed.processed.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)
        # Overlay must always be (H, W, 3) uint8
        if preprocessed.processed.ndim == 2:
            overlay = np.stack(
                [preprocessed.processed, preprocessed.processed, preprocessed.processed],
                axis=-1
            ).astype(np.uint8)
        else:
            overlay = preprocessed.processed[:, :, :3].astype(np.uint8)

        return ExplainabilityResult(
            heatmap=heatmap,
            overlay=overlay,
            method="gradcam",  # must be 'gradcam' or 'gradcam++' per dataclass validation
            target_layer="none",
            warning="Grad-CAM was not run (disabled or failed).",
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into *base*, returning a new dict.

    Scalar and list values in *override* take precedence over those in *base*.
    Nested dicts are merged recursively.

    Parameters
    ----------
    base : dict
        Default values.
    override : dict
        User-supplied values.

    Returns
    -------
    dict
        Merged configuration.
    """
    result = base.copy()
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


__all__ = ["OAPredictor"]
