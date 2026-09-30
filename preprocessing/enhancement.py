"""
preprocessing/enhancement.py
=============================
OpenCV-based image enhancement pipeline for OA Detection.

The :class:`OAPreprocessingPipeline` orchestrates a fixed sequence of
transforms designed to improve the visibility of bone and joint structures
in knee X-rays:

1. **Grayscale** — ensure single-channel input.
2. **Denoise** — non-local means denoising to reduce sensor noise.
3. **CLAHE** — contrast-limited adaptive histogram equalisation.
4. **Sharpen** — unsharp masking to enhance edges.
5. **Resize** — scale to the model's expected input size.
6. **Normalize** — linear min-max normalisation to ``[0, 255]``.

All intermediate step outputs are stored and returned for inspection /
reporting.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration values (used when config dict is absent/incomplete)
# ---------------------------------------------------------------------------
_DEFAULTS: Dict[str, Any] = {
    "target_size": [384, 384],
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid": [8, 8],
    "denoise_h": 10,
    "denoise_template_window": 7,
    "denoise_search_window": 21,
    "sharpen_kernel_size": 3,
}

# Morphological structuring element size for ROI detection
_MORPH_KERNEL_SIZE: int = 15
# Minimum fraction of total image area a contour must occupy to be considered
_MIN_ROI_AREA_FRACTION: float = 0.05
# Maximum fraction — avoids treating the entire image as the ROI
_MAX_ROI_AREA_FRACTION: float = 0.90


class OAPreprocessingPipeline:
    """Full preprocessing pipeline for knee X-ray images.

    Parameters
    ----------
    config:
        Dictionary containing preprocessing hyperparameters (see
        ``configs/oa_model.yaml`` → ``preprocessing`` section).
        Missing keys fall back to :data:`_DEFAULTS`.

    Examples
    --------
    >>> pipeline = OAPreprocessingPipeline(config=cfg["preprocessing"])
    >>> steps = pipeline.run(raw_image)
    >>> steps.keys()
    dict_keys(['grayscale', 'denoised', 'clahe', 'sharpened', 'resized', 'normalized'])
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._cfg: Dict[str, Any] = {**_DEFAULTS, **config}
        target_raw = self._cfg["target_size"]
        self._target_size: Tuple[int, int] = (int(target_raw[0]), int(target_raw[1]))
        self._clahe_clip: float = float(self._cfg["clahe_clip_limit"])
        tile_raw = self._cfg["clahe_tile_grid"]
        self._clahe_tile: Tuple[int, int] = (int(tile_raw[0]), int(tile_raw[1]))
        self._denoise_h: int = int(self._cfg["denoise_h"])
        self._denoise_template_win: int = int(self._cfg["denoise_template_window"])
        self._denoise_search_win: int = int(self._cfg["denoise_search_window"])
        self._sharpen_ksize: int = int(self._cfg["sharpen_kernel_size"])

        # Lazily create CLAHE object (not picklable, so keep as instance attr)
        self._clahe = cv2.createCLAHE(
            clipLimit=self._clahe_clip,
            tileGridSize=self._clahe_tile,
        )
        logger.debug(
            "OAPreprocessingPipeline: target=%s clahe_clip=%.1f denoise_h=%d",
            self._target_size,
            self._clahe_clip,
            self._denoise_h,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Execute the full preprocessing sequence on *image*.

        Parameters
        ----------
        image:
            Input image array.  May be grayscale ``(H, W)`` or colour
            ``(H, W, C)`` uint8.

        Returns
        -------
        Dict[str, np.ndarray]
            Ordered dictionary with keys:
            ``'grayscale'``, ``'denoised'``, ``'clahe'``,
            ``'sharpened'``, ``'resized'``, ``'normalized'``.
            All arrays are uint8 with shape ``(H, W)`` (or the
            target size for ``'resized'``/``'normalized'``).

        Raises
        ------
        ValueError
            If the input array is empty or has unexpected dimensions.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None.")
        if image.ndim not in (2, 3):
            raise ValueError(
                f"Input image must be 2-D or 3-D, got shape {image.shape}."
            )

        self._warn_unusual(image)

        steps: Dict[str, np.ndarray] = {}

        gray = self._to_grayscale(image)
        steps["grayscale"] = gray.copy()

        denoised = self._denoise(gray)
        steps["denoised"] = denoised.copy()

        clahe_img = self._apply_clahe(denoised)
        steps["clahe"] = clahe_img.copy()

        sharpened = self._sharpen(clahe_img)
        steps["sharpened"] = sharpened.copy()

        resized = self._resize(sharpened)
        steps["resized"] = resized.copy()

        normalized = self._normalize(resized)
        steps["normalized"] = normalized.copy()

        logger.debug("Pipeline complete. Steps: %s", list(steps.keys()))
        return steps

    def locate_joint_region(
        self,
        image: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """Heuristically locate the knee joint region of interest.

        The algorithm:
        1. Threshold the image with Otsu's method.
        2. Apply morphological closing to fill holes in bone regions.
        3. Find contours and select the largest one within area bounds.
        4. Return the bounding-box crop.

        Parameters
        ----------
        image:
            Preprocessed grayscale uint8 image of shape ``(H, W)``.

        Returns
        -------
        Tuple[Optional[np.ndarray], Optional[Tuple[int,int,int,int]]]
            ``(roi_image, (x, y, w, h))`` where the bbox is in pixel
            coordinates on *image*, or ``(None, None)`` if the ROI
            cannot be reliably located.
        """
        if image is None or image.size == 0:
            logger.warning("locate_joint_region: received empty image.")
            return None, None

        gray = self._to_grayscale(image)
        h_img, w_img = gray.shape
        total_area = h_img * w_img

        # ---- Threshold ----
        try:
            _, thresh = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        except cv2.error as exc:
            logger.warning("Otsu thresholding failed: %s", exc)
            return None, None

        # ---- Morphological closing ----
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (_MORPH_KERNEL_SIZE, _MORPH_KERNEL_SIZE)
        )
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        dilated = cv2.dilate(closed, kernel, iterations=1)

        # ---- Contour detection ----
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            logger.warning("No contours found during ROI detection.")
            return None, None

        # Filter by area fraction and pick the largest valid contour
        valid_contours = [
            c for c in contours
            if _MIN_ROI_AREA_FRACTION * total_area
            <= cv2.contourArea(c)
            <= _MAX_ROI_AREA_FRACTION * total_area
        ]
        if not valid_contours:
            logger.warning(
                "No contour within area fraction [%.2f, %.2f]; "
                "falling back to full image.",
                _MIN_ROI_AREA_FRACTION,
                _MAX_ROI_AREA_FRACTION,
            )
            return None, None

        largest = max(valid_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        # Add a small margin (5 %) clamped to image bounds
        margin_x = int(w * 0.05)
        margin_y = int(h * 0.05)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w_img, x + w + margin_x)
        y2 = min(h_img, y + h + margin_y)

        roi = image[y1:y2, x1:x2]
        bbox: Tuple[int, int, int, int] = (x1, y1, x2 - x1, y2 - y1)
        logger.debug("ROI located: bbox=%s on image %s", bbox, image.shape)
        return roi, bbox

    # ------------------------------------------------------------------
    # Pipeline steps (private)
    # ------------------------------------------------------------------

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert *image* to a single-channel uint8 grayscale array.

        Parameters
        ----------
        image:
            Input array ``(H, W)`` or ``(H, W, C)``, uint8.

        Returns
        -------
        np.ndarray
            Grayscale array ``(H, W)``, uint8.
        """
        if image.ndim == 2:
            return image.astype(np.uint8)
        if image.ndim == 3:
            if image.shape[2] == 1:
                return image[:, :, 0].astype(np.uint8)
            if image.shape[2] == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image.shape[2] == 4:
                bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        raise ValueError(f"Cannot convert array of shape {image.shape} to grayscale.")

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply OpenCV Non-Local Means denoising.

        Parameters
        ----------
        image:
            Grayscale uint8 array ``(H, W)``.

        Returns
        -------
        np.ndarray
            Denoised grayscale uint8 array ``(H, W)``.

        Notes
        -----
        If denoising fails (e.g. very small images), the original is
        returned with a logged warning.
        """
        try:
            denoised = cv2.fastNlMeansDenoising(
                image,
                h=float(self._denoise_h),
                templateWindowSize=self._denoise_template_win,
                searchWindowSize=self._denoise_search_win,
            )
            return denoised
        except cv2.error as exc:
            logger.warning(
                "Denoising failed (%s); returning original image.", exc
            )
            return image.copy()

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast-Limited Adaptive Histogram Equalisation).

        Parameters
        ----------
        image:
            Grayscale uint8 array ``(H, W)``.

        Returns
        -------
        np.ndarray
            Contrast-enhanced grayscale uint8 array ``(H, W)``.
        """
        try:
            return self._clahe.apply(image)
        except cv2.error as exc:
            logger.warning("CLAHE failed (%s); returning denoised image.", exc)
            return image.copy()

    def _sharpen(self, image: np.ndarray) -> np.ndarray:
        """Sharpen the image using unsharp masking.

        A Gaussian blur is subtracted from the original with a weighted
        sum to enhance high-frequency (edge) detail.

        Parameters
        ----------
        image:
            Grayscale uint8 array ``(H, W)``.

        Returns
        -------
        np.ndarray
            Sharpened grayscale uint8 array ``(H, W)``.
        """
        ksize = self._sharpen_ksize
        # Kernel size must be odd
        if ksize % 2 == 0:
            ksize += 1
        try:
            blurred = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=0)
            # unsharp mask: sharpened = 1.5 * original - 0.5 * blurred
            sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
            return sharpened
        except cv2.error as exc:
            logger.warning("Sharpening failed (%s); returning CLAHE image.", exc)
            return image.copy()

    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Resize *image* to the configured target size.

        Uses :attr:`cv2.INTER_LANCZOS4` for downscaling and
        :attr:`cv2.INTER_CUBIC` for upscaling.

        Parameters
        ----------
        image:
            Grayscale uint8 array ``(H, W)``.

        Returns
        -------
        np.ndarray
            Resized grayscale uint8 array ``(target_h, target_w)``.
        """
        target_h, target_w = self._target_size
        src_h, src_w = image.shape[:2]

        if (src_h, src_w) == (target_h, target_w):
            return image.copy()

        interpolation = (
            cv2.INTER_LANCZOS4
            if (src_h > target_h or src_w > target_w)
            else cv2.INTER_CUBIC
        )
        resized = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
        logger.debug(
            "Resized %s → %s (interp=%s).", image.shape, resized.shape, interpolation
        )
        return resized

    @staticmethod
    def _normalize(image: np.ndarray) -> np.ndarray:
        """Min-max normalise *image* to the full uint8 range ``[0, 255]``.

        Parameters
        ----------
        image:
            Grayscale uint8 array ``(H, W)``.

        Returns
        -------
        np.ndarray
            Normalised grayscale uint8 array ``(H, W)``.
        """
        min_val = float(image.min())
        max_val = float(image.max())

        if max_val == min_val:
            # Constant image — return zeros to avoid division by zero
            logger.warning(
                "Normalization: image is constant (min=max=%.1f); returning zeros.",
                min_val,
            )
            return np.zeros_like(image, dtype=np.uint8)

        normalized = (image.astype(np.float32) - min_val) / (max_val - min_val) * 255.0
        return normalized.astype(np.uint8)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _warn_unusual(self, image: np.ndarray) -> None:
        """Log warnings for unusual image characteristics.

        Parameters
        ----------
        image:
            Input image before any processing.
        """
        h, w = image.shape[:2]
        aspect = w / max(h, 1)

        if aspect > 3.0 or aspect < 0.33:
            logger.warning(
                "Unusual aspect ratio %.2f for image of shape %s; "
                "expected close to 1:1 for knee X-rays.",
                aspect,
                image.shape,
            )

        mean_val = float(np.mean(image))
        std_val = float(np.std(image))

        if std_val < 5.0:
            logger.warning(
                "Very low pixel std-dev (%.2f) — image may be nearly uniform.",
                std_val,
            )

        if mean_val < 5.0:
            logger.warning(
                "Very low mean intensity (%.2f) — image appears near-black.",
                mean_val,
            )

        if mean_val > 250.0:
            logger.warning(
                "Very high mean intensity (%.2f) — image appears near-white.",
                mean_val,
            )

        if h < 100 or w < 100:
            logger.warning(
                "Image is very small (%d×%d px); results may be unreliable.",
                h,
                w,
            )
