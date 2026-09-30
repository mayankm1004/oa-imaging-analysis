"""
inference/risk_markers.py
=========================
Structured OA risk-marker extraction.

Combines deep-learning model predictions with classical computer-vision
measurements to produce per-marker probability estimates for six canonical
osteoarthritis risk indicators.

.. important::
    All measurements produced by this module are **proxy estimates** derived
    from CV algorithms applied to 2-D image data.  They are **not**
    clinically validated measurements and are **not** equivalent to
    radiological findings made by a qualified clinician.  Results must
    always be reviewed by an appropriate healthcare professional.

Classes
-------
RiskMarkerExtractor
    Main class; instantiate once and call :meth:`extract` per image.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from inference import RiskMarkerResult, RiskMarkersReport
from preprocessing import PreprocessingResult
from models import ModelPrediction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default marker thresholds (overridden by config)
# ---------------------------------------------------------------------------

_DEFAULT_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "joint_space_narrowing": {"detected": 0.65, "possible": 0.40, "uncertain": 0.20},
    "osteophytes":           {"detected": 0.60, "possible": 0.35, "uncertain": 0.15},
    "subchondral_sclerosis": {"detected": 0.65, "possible": 0.40, "uncertain": 0.20},
    "bone_deformity":        {"detected": 0.70, "possible": 0.45, "uncertain": 0.20},
    "alignment_abnormality": {"detected": 0.60, "possible": 0.35, "uncertain": 0.15},
    "cystic_changes":        {"detected": 0.60, "possible": 0.35, "uncertain": 0.15},
}

_MODEL_WEIGHT: float = 0.60   # weight of model OA score vs CV measurement


class RiskMarkerExtractor:
    """Extracts structured OA risk marker assessments from model predictions
    and image analysis measurements.

    Each of the six OA risk markers is estimated by a dedicated ``_assess_*``
    method that:

    1. Performs a classical CV measurement on the (optionally ROI-cropped)
       preprocessed image.
    2. Blends the CV-derived score with the model's overall ``oa_score`` using
       a configurable weight (``risk_markers.model_weight`` in the YAML config).
    3. Converts the blended score to a status string via
       :meth:`_score_to_status`.

    .. important::
        **IMPORTANT**: Risk markers are model/image-derived estimates.  They are
        **not** equivalent to radiological findings by a qualified clinician.
        Marker status combines deep-learning features with classical CV
        measurements.

    Parameters
    ----------
    config : dict
        Full configuration dictionary loaded from ``configs/oa_model.yaml``.
        The relevant sub-key is ``risk_markers``.  If absent, defaults are
        used.

    Example
    -------
    >>> extractor = RiskMarkerExtractor(config)
    >>> report = extractor.extract(preprocessed, prediction, segmenter)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        rm_cfg: Dict[str, Any] = config.get("risk_markers", {})
        self._model_weight: float = float(
            rm_cfg.get("model_weight", _MODEL_WEIGHT)
        )
        self._cv_weight: float = 1.0 - self._model_weight

        # Per-marker thresholds
        self._thresholds: Dict[str, Dict[str, float]] = {}
        for marker, defaults in _DEFAULT_THRESHOLDS.items():
            self._thresholds[marker] = {
                level: float(
                    rm_cfg.get("thresholds", {}).get(marker, {}).get(level, val)
                )
                for level, val in defaults.items()
            }

        # Kernel sizes for CV operations
        self._canny_low: int = int(rm_cfg.get("canny_low", 50))
        self._canny_high: int = int(rm_cfg.get("canny_high", 150))
        self._blob_min_area: int = int(rm_cfg.get("blob_min_area", 20))
        self._blob_max_area: int = int(rm_cfg.get("blob_max_area", 500))

        logger.debug(
            "RiskMarkerExtractor initialised | model_weight=%.2f | "
            "cv_weight=%.2f | thresholds=%s",
            self._model_weight,
            self._cv_weight,
            self._thresholds,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        preprocessed: PreprocessingResult,
        prediction: ModelPrediction,
        segmenter: Any,  # JointRegionSegmenter – typed as Any to avoid circular import
    ) -> RiskMarkersReport:
        """Extract all six OA risk markers.

        Uses the ROI image when available, otherwise falls back to the full
        preprocessed image.

        Parameters
        ----------
        preprocessed : PreprocessingResult
            Result from the preprocessing stage.
        prediction : ModelPrediction
            Model prediction containing the raw OA score.
        segmenter : JointRegionSegmenter
            Segmentation helper (may be queried for region masks in future
            versions; currently kept for API compatibility).

        Returns
        -------
        RiskMarkersReport
            Structured per-marker assessments.
        """
        # Select working image: prefer ROI crop
        image: np.ndarray = (
            preprocessed.roi_image
            if preprocessed.roi_image is not None
            else preprocessed.processed
        )

        # Convert to uint8 grayscale for CV operations
        gray = self._to_gray_uint8(image)

        oa_score: float = float(prediction.oa_score)

        logger.info(
            "Extracting risk markers | oa_score=%.4f | image_shape=%s",
            oa_score,
            gray.shape,
        )

        return RiskMarkersReport(
            joint_space_narrowing=self._assess_joint_space_narrowing(gray, oa_score),
            osteophytes=self._assess_osteophytes(gray, oa_score),
            subchondral_sclerosis=self._assess_subchondral_sclerosis(gray, oa_score),
            bone_deformity=self._assess_bone_deformity(gray, oa_score),
            alignment_abnormality=self._assess_alignment(gray, oa_score),
            cystic_changes=self._assess_cystic_changes(gray, oa_score),
        )

    # ------------------------------------------------------------------
    # Per-marker assessment methods
    # ------------------------------------------------------------------

    def _assess_joint_space_narrowing(
        self, image: np.ndarray, oa_score: float
    ) -> RiskMarkerResult:
        """Combine joint-space proxy measurement with model OA score.

        Analyses intensity in the central horizontal band of the image as a
        proxy for the joint space.  A lower mean intensity in the central band
        (relative to the overall image) suggests reduced joint spacing on
        plain radiographs.

        Parameters
        ----------
        image : np.ndarray
            Grayscale uint8 image (H × W).
        oa_score : float
            Overall OA probability from the model.

        Returns
        -------
        RiskMarkerResult
        """
        h, w = image.shape[:2]
        # Central horizontal band: rows 40–60 % of height
        band_top = int(h * 0.40)
        band_bot = int(h * 0.60)
        central_band = image[band_top:band_bot, :]

        global_mean = float(np.mean(image)) + 1e-8
        band_mean = float(np.mean(central_band))

        # Inverted ratio: low band intensity relative to global -> narrowing proxy
        # Normalise to [0, 1]; clamp
        ratio = 1.0 - min(band_mean / (global_mean * 1.5), 1.0)
        ratio = float(np.clip(ratio, 0.0, 1.0))

        blended = self._blend(ratio, oa_score)
        status = self._score_to_status(blended, "joint_space_narrowing")
        confidence = self._status_confidence(blended, "joint_space_narrowing")

        return RiskMarkerResult(
            status=status,
            confidence=confidence,
            source="image_analysis",
            note=(
                f"Proxy via central-band intensity analysis "
                f"(cv_score={ratio:.3f}, oa_score={oa_score:.3f}, "
                f"blended={blended:.3f}). "
                "NOT a calibrated joint-space-width measurement."
            ),
        )

    def _assess_osteophytes(
        self, image: np.ndarray, oa_score: float
    ) -> RiskMarkerResult:
        """Edge-density analysis near bone margins as an osteophyte proxy.

        Applies Canny edge detection to the upper and lower 25 % of the image
        (approximating bone-margin regions for knee X-rays).  Higher edge
        density in these regions may indicate bony spurring.

        Parameters
        ----------
        image : np.ndarray
            Grayscale uint8 image (H × W).
        oa_score : float
            Overall OA probability from the model.

        Returns
        -------
        RiskMarkerResult
        """
        h, w = image.shape[:2]
        # Upper and lower margin bands
        upper_band = image[: int(h * 0.25), :]
        lower_band = image[int(h * 0.75) :, :]
        margin_band = np.vstack([upper_band, lower_band])

        edges = cv2.Canny(margin_band, self._canny_low, self._canny_high)
        edge_density = float(np.mean(edges > 0))  # fraction of edge pixels

        # Normalise: typical edge densities 0.02–0.30; scale to [0,1]
        cv_score = float(np.clip(edge_density / 0.25, 0.0, 1.0))
        blended = self._blend(cv_score, oa_score)
        status = self._score_to_status(blended, "osteophytes")
        confidence = self._status_confidence(blended, "osteophytes")

        return RiskMarkerResult(
            status=status,
            confidence=confidence,
            source="image_analysis",
            note=(
                f"Proxy via Canny edge density in bone-margin regions "
                f"(edge_density={edge_density:.4f}, cv_score={cv_score:.3f}, "
                f"oa_score={oa_score:.3f}, blended={blended:.3f}). "
                "Edge density is an indirect indicator; soft tissue and "
                "image quality affect results. NOT a clinical osteophyte measurement."
            ),
        )

    def _assess_subchondral_sclerosis(
        self, image: np.ndarray, oa_score: float
    ) -> RiskMarkerResult:
        """Intensity histogram analysis as a subchondral sclerosis proxy.

        High-intensity pixels clustered near the joint line are a proxy for
        increased subchondral bone density (sclerosis) on radiographs.
        Analyses the top 10 % of pixel intensities in the central band.

        Parameters
        ----------
        image : np.ndarray
            Grayscale uint8 image (H × W).
        oa_score : float
            Overall OA probability from the model.

        Returns
        -------
        RiskMarkerResult
        """
        h, w = image.shape[:2]
        # Near-joint bands: 30–45 % and 55–70 % of height
        upper_jt = image[int(h * 0.30) : int(h * 0.45), :]
        lower_jt = image[int(h * 0.55) : int(h * 0.70), :]
        jt_band = np.vstack([upper_jt, lower_jt])

        # Fraction of pixels above the 90th percentile of the full image
        p90 = float(np.percentile(image, 90))
        high_intensity_frac = float(np.mean(jt_band > p90))

        cv_score = float(np.clip(high_intensity_frac / 0.30, 0.0, 1.0))
        blended = self._blend(cv_score, oa_score)
        status = self._score_to_status(blended, "subchondral_sclerosis")
        confidence = self._status_confidence(blended, "subchondral_sclerosis")

        return RiskMarkerResult(
            status=status,
            confidence=confidence,
            source="image_analysis",
            note=(
                f"Proxy via high-intensity fraction near joint line "
                f"(hi_frac={high_intensity_frac:.4f}, cv_score={cv_score:.3f}, "
                f"oa_score={oa_score:.3f}, blended={blended:.3f}). "
                "Intensity alone does not confirm sclerosis; beam hardening and "
                "exposure settings affect values. NOT a clinical measurement."
            ),
        )

    def _assess_bone_deformity(
        self, image: np.ndarray, oa_score: float
    ) -> RiskMarkerResult:
        """Shape analysis proxy for gross bone deformity.

        Thresholds the image to isolate bright bony regions, finds the largest
        contour, and computes its solidity (area / convex-hull area).  Low
        solidity indicates an irregular / non-convex bone silhouette, which
        may proxy for deformity.

        Parameters
        ----------
        image : np.ndarray
            Grayscale uint8 image (H × W).
        oa_score : float
            Overall OA probability from the model.

        Returns
        -------
        RiskMarkerResult
        """
        # Adaptive threshold to get bony regions
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        solidity = 1.0  # default: assume convex (no deformity signal)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            hull = cv2.convexHull(largest)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = float(area / hull_area)

        # Inverted: low solidity -> potential deformity
        irregularity = float(np.clip(1.0 - solidity, 0.0, 1.0))
        blended = self._blend(irregularity, oa_score)
        status = self._score_to_status(blended, "bone_deformity")
        confidence = self._status_confidence(blended, "bone_deformity")

        return RiskMarkerResult(
            status=status,
            confidence=confidence,
            source="image_analysis",
            note=(
                f"Proxy via bone-silhouette solidity analysis "
                f"(solidity={solidity:.3f}, irregularity={irregularity:.3f}, "
                f"oa_score={oa_score:.3f}, blended={blended:.3f}). "
                "Contour analysis is sensitive to image quality and thresholding. "
                "NOT a validated bone-deformity measurement."
            ),
        )

    def _assess_alignment(
        self, image: np.ndarray, oa_score: float
    ) -> RiskMarkerResult:
        """Hough-line analysis proxy for mechanical-axis alignment.

        Detects dominant line orientations in the image as a rough proxy for
        limb alignment.  High angular variance may indicate varus/valgus
        deviation.

        .. note::
            **Reliable alignment measurement requires calibrated full-limb
            standing radiographs.**  This is a proxy estimate only and should
            be treated with extreme caution.

        Parameters
        ----------
        image : np.ndarray
            Grayscale uint8 image (H × W).
        oa_score : float
            Overall OA probability from the model.

        Returns
        -------
        RiskMarkerResult
        """
        edges = cv2.Canny(image, self._canny_low, self._canny_high)
        lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=80)

        angular_variance = 0.0
        n_lines = 0
        if lines is not None:
            angles = [line[0][1] for line in lines]  # theta in radians
            # Convert to degrees in [0, 90] (orientation, not direction)
            angles_deg = [np.degrees(a) % 90 for a in angles]
            n_lines = len(angles_deg)
            angular_variance = float(np.var(angles_deg)) if n_lines > 1 else 0.0

        # Normalise variance: 0 = all lines parallel (normal), high = chaotic
        cv_score = float(np.clip(angular_variance / 900.0, 0.0, 1.0))  # 900 = 30°² var
        blended = self._blend(cv_score, oa_score)
        status = self._score_to_status(blended, "alignment_abnormality")
        confidence = self._status_confidence(blended, "alignment_abnormality")

        return RiskMarkerResult(
            status=status,
            confidence=confidence,
            source="image_analysis",
            note=(
                f"Proxy via Hough-line angular variance "
                f"(n_lines={n_lines}, ang_var={angular_variance:.2f}°², "
                f"cv_score={cv_score:.3f}, oa_score={oa_score:.3f}, "
                f"blended={blended:.3f}). "
                "IMPORTANT: Reliable alignment measurement requires calibrated "
                "full-limb standing radiographs.  This estimate is highly "
                "unreliable and should NOT be used for clinical alignment decisions."
            ),
        )

    def _assess_cystic_changes(
        self, image: np.ndarray, oa_score: float
    ) -> RiskMarkerResult:
        """Blob-detection proxy for possible subchondral cystic changes.

        Detects dark circular blobs near the joint line as a proxy for
        subchondral cysts.  Uses SimpleBlobDetector with area and circularity
        constraints.

        Parameters
        ----------
        image : np.ndarray
            Grayscale uint8 image (H × W).
        oa_score : float
            Overall OA probability from the model.

        Returns
        -------
        RiskMarkerResult
        """
        h, w = image.shape[:2]
        # Focus on central third of image (joint region)
        roi = image[int(h * 0.35) : int(h * 0.65), :]

        # Configure blob detector
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = float(self._blob_min_area)
        params.maxArea = float(self._blob_max_area)
        params.filterByCircularity = True
        params.minCircularity = 0.5
        params.filterByConvexity = True
        params.minConvexity = 0.7
        params.filterByInertia = True
        params.minInertiaRatio = 0.4

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(roi)
        n_blobs = len(keypoints)

        # Normalise: 0 blobs -> 0, ≥5 blobs -> 1
        cv_score = float(np.clip(n_blobs / 5.0, 0.0, 1.0))
        blended = self._blend(cv_score, oa_score)
        status = self._score_to_status(blended, "cystic_changes")
        confidence = self._status_confidence(blended, "cystic_changes")

        return RiskMarkerResult(
            status=status,
            confidence=confidence,
            source="image_analysis",
            note=(
                f"Proxy via blob detection in joint ROI "
                f"(n_blobs={n_blobs}, cv_score={cv_score:.3f}, "
                f"oa_score={oa_score:.3f}, blended={blended:.3f}). "
                "Blob detection is sensitive to noise and image artefacts. "
                "NOT a validated cystic-change measurement."
            ),
        )

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _score_to_status(self, score: float, marker: str) -> str:
        """Convert a blended [0, 1] score to a status string.

        Uses per-marker thresholds from the configuration:

        ===================== =================
        Condition             Status
        ===================== =================
        score ≥ detected_thr  ``'detected'``
        score ≥ possible_thr  ``'possible'``
        score ≥ uncertain_thr ``'uncertain'``
        score <  uncertain    ``'not_detected'``
        ===================== =================

        Parameters
        ----------
        score : float
            Blended [0, 1] score.
        marker : str
            One of the six canonical marker names (must be a key in
            :attr:`_thresholds`).

        Returns
        -------
        str
            Status string.
        """
        thr = self._thresholds.get(marker, _DEFAULT_THRESHOLDS.get(marker, {}))
        if score >= thr.get("detected", 0.65):
            return "detected"
        if score >= thr.get("possible", 0.40):
            return "possible"
        if score >= thr.get("uncertain", 0.20):
            return "uncertain"
        return "not_detected"

    def _status_confidence(self, score: float, marker: str) -> float:
        """Derive a confidence value for the assigned status.

        Confidence reflects how far the score lies from the nearest decision
        boundary, expressed as a value in [0.5, 1.0].

        Parameters
        ----------
        score : float
            Blended [0, 1] score.
        marker : str
            Marker name.

        Returns
        -------
        float
            Confidence in [0.5, 1.0].
        """
        thr = self._thresholds.get(marker, _DEFAULT_THRESHOLDS.get(marker, {}))
        boundaries = sorted(thr.values())  # ascending
        # Distance to nearest boundary (normalised to [0, 0.5] -> add 0.5)
        min_dist = min(abs(score - b) for b in boundaries) if boundaries else 0.0
        confidence = float(np.clip(0.5 + min_dist, 0.5, 1.0))
        return confidence

    def _blend(self, cv_score: float, oa_score: float) -> float:
        """Weighted blend of CV-derived score and model OA score.

        Parameters
        ----------
        cv_score : float
            Score derived from classical CV measurement, in [0, 1].
        oa_score : float
            Model-predicted overall OA probability, in [0, 1].

        Returns
        -------
        float
            Blended score in [0, 1].
        """
        return float(
            np.clip(
                self._model_weight * oa_score + self._cv_weight * cv_score,
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _to_gray_uint8(image: np.ndarray) -> np.ndarray:
        """Convert an arbitrary image array to grayscale uint8.

        Handles:
        - Already grayscale (2-D or 3-D single channel)
        - RGB / BGR 3-channel
        - Float images (normalised to [0, 1] or arbitrary range)

        Parameters
        ----------
        image : np.ndarray
            Input image.

        Returns
        -------
        np.ndarray
            Grayscale uint8 image.
        """
        img = image.copy()

        # Squeeze single-channel dim if present
        if img.ndim == 3 and img.shape[2] == 1:
            img = img[:, :, 0]

        # Convert colour to gray
        if img.ndim == 3:
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

        # Normalise float images
        if img.dtype != np.uint8:
            img = img.astype(np.float64)
            lo, hi = img.min(), img.max()
            if hi > lo:
                img = (img - lo) / (hi - lo) * 255.0
            else:
                img = np.zeros_like(img)
            img = img.astype(np.uint8)

        return img


__all__ = ["RiskMarkerExtractor"]
