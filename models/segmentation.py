"""
models/segmentation.py
======================
Classical CV-based joint-region localiser for X-ray images.

This is an **MVP** implementation using OpenCV morphological operations
and contour detection.  For production use, replace this module with a
trained segmentation network (U-Net, nnU-Net, etc.) fine-tuned on
annotated joint X-ray data.

The module exposes one public class:

* :class:`JointRegionSegmenter` – detects the joint region, returns the
  cropped ROI and its bounding box, and computes a proxy joint-space
  width score.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

__all__ = ["JointRegionSegmenter"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunable defaults
# ---------------------------------------------------------------------------

# Minimum fraction of image area that a contour must occupy to be considered
# a candidate bone region.
_MIN_AREA_FRACTION: float = 0.05

# Padding (pixels) added around the detected bounding box before cropping.
_BBOX_PADDING: int = 20

# Fraction of the bounding-box width / height used as the central analysis
# window for joint-space estimation.
_JOINT_SPACE_WINDOW_FRACTION: float = 0.2

# Minimum number of candidate contours required before we give up.
_MIN_CONTOURS: int = 1


class JointRegionSegmenter:
    """Localise the joint region in an X-ray image using classical CV.

    The pipeline is:

    1. Adaptive thresholding to binarise the bone signal.
    2. Morphological *close* (fill small intra-bone gaps) then *open*
       (remove noise).
    3. Find external contours and select the largest (presumed bone).
    4. Add padding and crop the region of interest.

    .. note::
        This implementation does not use a learned model.  It will fail on
        unusual projections, severe artefacts, or very small joints.  The
        ``segment()`` method returns ``(None, None)`` gracefully in those
        cases rather than raising an exception.

    Parameters
    ----------
    min_area_fraction : float
        Minimum fraction of the total image area that the largest contour
        must occupy to be accepted (default: 0.05).
    bbox_padding : int
        Pixel padding added on all sides of the detected bounding box
        before cropping the ROI (default: 20).
    """

    def __init__(
        self,
        min_area_fraction: float = _MIN_AREA_FRACTION,
        bbox_padding: int = _BBOX_PADDING,
    ) -> None:
        self.min_area_fraction = min_area_fraction
        self.bbox_padding = bbox_padding

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment(
        self,
        image: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """Localise the joint region and return the cropped ROI.

        Parameters
        ----------
        image : np.ndarray
            Grayscale X-ray image of shape ``(H, W)`` or ``(H, W, 1)``,
            dtype ``uint8`` or ``float32``.  Float images are expected to
            have values in ``[0, 1]`` and will be scaled to ``[0, 255]``.

        Returns
        -------
        roi_image : np.ndarray or None
            Cropped region of interest, dtype ``uint8``.  ``None`` when
            no suitable region is found.
        bbox : Tuple[int, int, int, int] or None
            ``(x, y, w, h)`` bounding box in the original image coordinate
            system.  ``None`` when no region is found.
        """
        gray = self._to_uint8(image)

        bbox = self._find_bone_region(gray)
        if bbox is None:
            logger.warning(
                "JointRegionSegmenter: No bone region found. "
                "Returning full image as fallback."
            )
            return None, None

        x, y, w, h = bbox
        roi = gray[y : y + h, x : x + w]
        logger.debug(
            "Joint ROI detected: x=%d y=%d w=%d h=%d", x, y, w, h
        )
        return roi, bbox

    def get_joint_space_score(self, image: np.ndarray) -> float:
        """Compute a proxy score for joint-space width.

        A higher score suggests more preserved joint space; a lower score
        suggests narrowing.  This is a coarse heuristic and **must not**
        be interpreted as a clinical measurement.

        Parameters
        ----------
        image : np.ndarray
            Grayscale X-ray image ``(H, W)``, uint8 or float32.

        Returns
        -------
        float
            Proxy score in ``[0, 1]`` where 1.0 = maximum estimated
            joint space.
        """
        gray = self._to_uint8(image)
        bbox = self._find_bone_region(gray)
        if bbox is None:
            logger.warning(
                "JointRegionSegmenter.get_joint_space_score: "
                "Could not find bone region; returning 0.5 (neutral)."
            )
            return 0.5
        return self._estimate_joint_space(gray, bbox)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_uint8(image: np.ndarray) -> np.ndarray:
        """Normalise and convert an image to uint8 grayscale.

        Parameters
        ----------
        image : np.ndarray
            Input image.  Accepted shapes: ``(H, W)``, ``(H, W, 1)``,
            ``(H, W, 3)``.  Accepted dtypes: ``uint8``, ``float32``,
            ``float64``.

        Returns
        -------
        np.ndarray
            Grayscale uint8 image of shape ``(H, W)``.

        Raises
        ------
        ValueError
            If the input has an unsupported shape or dtype.
        """
        if image.ndim == 3:
            if image.shape[2] == 1:
                image = image[:, :, 0]
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                raise ValueError(
                    f"Unsupported channel count {image.shape[2]}; expected 1 or 3."
                )
        elif image.ndim != 2:
            raise ValueError(
                f"Image must be 2-D or 3-D, got {image.ndim}-D array."
            )

        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).clip(0, 255).astype(np.uint8)
            else:
                image = image.clip(0, 255).astype(np.uint8)

        return image

    def _find_bone_region(
        self,
        image: np.ndarray,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Detect the dominant bone region via morphological filtering.

        Algorithm
        ---------
        1. Adaptive threshold (Gaussian, block size = 1/8 image width,
           rounded to nearest odd; C = -5).
        2. Morphological *close* with a 15x15 elliptical kernel to fill
           intra-bone gaps.
        3. Morphological *open* with a 5x5 elliptical kernel to remove
           small noise blobs.
        4. Find external contours; pick the largest by area.
        5. Accept only if the largest contour's area exceeds
           ``min_area_fraction`` of the total image area.
        6. Return the padded bounding box clamped to image boundaries.

        Parameters
        ----------
        image : np.ndarray
            Grayscale uint8 image.

        Returns
        -------
        Tuple[int, int, int, int] or None
            ``(x, y, w, h)`` padded bounding box, or ``None``.
        """
        h_img, w_img = image.shape
        total_area = h_img * w_img

        # ---- 1. Adaptive threshold ----------------------------------------
        block_size = max(3, (w_img // 8) | 1)  # ensure odd, at least 3
        binary = cv2.adaptiveThreshold(
            image,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=block_size,
            C=-5,
        )

        # ---- 2. Morphological close (fill gaps) ----------------------------
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close, iterations=2)

        # ---- 3. Morphological open (remove noise) --------------------------
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_open, iterations=1)

        # ---- 4. Find contours ---------------------------------------------
        contours, _ = cv2.findContours(
            opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) < _MIN_CONTOURS:
            return None

        # Largest contour by area
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        # ---- 5. Area threshold check --------------------------------------
        if area < self.min_area_fraction * total_area:
            logger.debug(
                "Largest contour area %.0f < %.0f (min fraction threshold). "
                "Skipping.",
                area,
                self.min_area_fraction * total_area,
            )
            return None

        # ---- 6. Padded bounding box --------------------------------------
        x, y, w, h = cv2.boundingRect(largest)
        p = self.bbox_padding
        x_pad = max(0, x - p)
        y_pad = max(0, y - p)
        w_pad = min(w_img - x_pad, w + 2 * p)
        h_pad = min(h_img - y_pad, h + 2 * p)

        return (x_pad, y_pad, w_pad, h_pad)

    def _estimate_joint_space(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> float:
        """Estimate joint space via intensity profile analysis.

        The heuristic works as follows:

        1. Crop the central ``_JOINT_SPACE_WINDOW_FRACTION`` of the
           bounding-box region in both axes.
        2. Compute a column-wise mean intensity profile.
        3. The joint space corresponds to a bright (high intensity) band
           between two darker bone regions.  The mean intensity of the
           central row band is used as a proxy.
        4. Normalise to ``[0, 1]`` using the global image statistics.

        Parameters
        ----------
        image : np.ndarray
            Full-size grayscale uint8 image.
        bbox : Tuple[int, int, int, int]
            ``(x, y, w, h)`` bounding box of the bone region.

        Returns
        -------
        float
            Score in ``[0, 1]``.  Higher → more preserved joint space.
        """
        x, y, w, h = bbox

        # Central window within the bounding box
        cx = x + w // 2
        cy = y + h // 2
        half_w = max(1, int(w * _JOINT_SPACE_WINDOW_FRACTION / 2))
        half_h = max(1, int(h * _JOINT_SPACE_WINDOW_FRACTION / 2))

        y1 = max(0, cy - half_h)
        y2 = min(image.shape[0], cy + half_h)
        x1 = max(0, cx - half_w)
        x2 = min(image.shape[1], cx + half_w)

        central_patch = image[y1:y2, x1:x2]
        if central_patch.size == 0:
            return 0.5  # neutral fallback

        # Mean intensity of the central patch vs. global range
        patch_mean = float(central_patch.mean())
        global_mean = float(image.mean())
        global_std = float(image.std()) + 1e-6  # avoid division by zero

        # Z-score relative to global, mapped to [0, 1] via sigmoid
        z = (patch_mean - global_mean) / global_std
        score = float(1.0 / (1.0 + np.exp(-z)))  # sigmoid in [0, 1]

        logger.debug(
            "Joint-space score: patch_mean=%.1f global_mean=%.1f z=%.3f score=%.3f",
            patch_mean,
            global_mean,
            z,
            score,
        )
        return score
