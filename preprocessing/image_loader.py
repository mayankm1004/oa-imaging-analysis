"""
preprocessing/image_loader.py
==============================
High-level image loading for the OA Detection preprocessing pipeline.

Supports standard image formats (.jpg, .jpeg, .png) and DICOM (.dcm).
All errors are raised explicitly — nothing is silently swallowed.

Exceptions hierarchy
--------------------
::

    OAImageError (base)
    ├── UnsupportedFormatError
    ├── CorruptedImageError
    ├── ImageDimensionError
    └── ImageModalityWarning   (Warning, not Exception)
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import cv2
import numpy as np

from preprocessing import PreprocessingResult
from preprocessing.dicom import DICOMLoadError, load_dicom
from preprocessing.enhancement import OAPreprocessingPipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported extensions
# ---------------------------------------------------------------------------
_STANDARD_IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png"}
_DICOM_EXTENSIONS: Set[str] = {".dcm"}
_ALL_SUPPORTED_EXTENSIONS: Set[str] = _STANDARD_IMAGE_EXTENSIONS | _DICOM_EXTENSIONS

# ---------------------------------------------------------------------------
# Grayscale detection heuristic
# ---------------------------------------------------------------------------
_GRAYSCALE_CHANNEL_STD_THRESHOLD: float = 5.0  # max per-channel std-dev diff
_MIN_MEAN_INTENSITY: float = 10.0              # images shouldn't be nearly black
_MAX_MEAN_INTENSITY: float = 245.0             # images shouldn't be nearly white


# ---------------------------------------------------------------------------
# Custom exceptions / warnings
# ---------------------------------------------------------------------------


class OAImageError(Exception):
    """Base class for all image-loading errors in this module."""


class UnsupportedFormatError(OAImageError):
    """Raised when the file extension is not supported.

    Attributes
    ----------
    path:
        Path that was attempted.
    extension:
        The unsupported file extension.
    """

    def __init__(self, path: str, extension: str) -> None:
        self.path = path
        self.extension = extension
        super().__init__(
            f"Unsupported file format '{extension}' for '{path}'. "
            f"Supported: {sorted(_ALL_SUPPORTED_EXTENSIONS)}."
        )


class CorruptedImageError(OAImageError):
    """Raised when a file cannot be decoded as a valid image.

    Attributes
    ----------
    path:
        Path that was attempted.
    reason:
        Human-readable description of the decode failure.
    """

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Cannot decode image '{path}': {reason}")


class ImageDimensionError(OAImageError):
    """Raised when an image's spatial dimensions are outside allowed bounds.

    Attributes
    ----------
    path:
        Path that was attempted.
    shape:
        The actual ``(height, width)`` of the loaded image.
    min_size:
        Configured minimum ``(height, width)``.
    max_size:
        Configured maximum ``(height, width)``.
    """

    def __init__(
        self,
        path: str,
        shape: Tuple[int, int],
        min_size: Tuple[int, int],
        max_size: Tuple[int, int],
    ) -> None:
        self.path = path
        self.shape = shape
        self.min_size = min_size
        self.max_size = max_size
        super().__init__(
            f"Image dimensions {shape} for '{path}' are outside allowed bounds "
            f"[min={min_size}, max={max_size}]."
        )


class ImageModalityWarning(UserWarning):
    """Emitted when an image does not appear to be a grayscale X-ray.

    This is a :class:`Warning`, not an :class:`Exception`.  It is issued
    with :func:`warnings.warn` so callers can choose to filter or elevate it.
    """


# ---------------------------------------------------------------------------
# ImageLoader
# ---------------------------------------------------------------------------


class ImageLoader:
    """Load images from disk and run the full OA preprocessing pipeline.

    Parameters
    ----------
    config:
        The ``preprocessing`` section of the parsed ``oa_model.yaml``
        configuration dictionary.  If ``None``, default values are used.

    Examples
    --------
    >>> loader = ImageLoader(config=cfg["preprocessing"])
    >>> result = loader.load("/path/to/knee.dcm")
    >>> result.processed.shape
    (384, 384)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._cfg: Dict[str, Any] = config or {}
        min_raw = self._cfg.get("min_image_size", [100, 100])
        max_raw = self._cfg.get("max_image_size", [4096, 4096])
        self._min_size: Tuple[int, int] = (int(min_raw[0]), int(min_raw[1]))
        self._max_size: Tuple[int, int] = (int(max_raw[0]), int(max_raw[1]))
        self._pipeline = OAPreprocessingPipeline(self._cfg)
        logger.debug(
            "ImageLoader initialised: min_size=%s max_size=%s",
            self._min_size,
            self._max_size,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, path: str) -> PreprocessingResult:
        """Load an image file and return a fully populated
        :class:`~preprocessing.PreprocessingResult`.

        Parameters
        ----------
        path:
            Absolute or relative path to the image file.

        Returns
        -------
        PreprocessingResult
            Populated with the original image, all pipeline steps,
            ROI crop, safe metadata, and any accumulated warnings.

        Raises
        ------
        UnsupportedFormatError
            If the file extension is not one of the supported types.
        CorruptedImageError
            If the file cannot be decoded as a valid image.
        ImageDimensionError
            If the image dimensions fall outside the configured bounds.
        DICOMLoadError
            If a ``.dcm`` file is invalid or cannot be decoded.
        FileNotFoundError
            If the path does not point to an existing file.
        """
        file_path = Path(path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: '{file_path}'")
        if not file_path.is_file():
            raise CorruptedImageError(str(file_path), "Path is not a regular file.")

        ext = file_path.suffix.lower()
        if ext not in _ALL_SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(str(file_path), ext)

        run_warnings: list[str] = []

        # ---- Load raw image ----
        if ext in _DICOM_EXTENSIONS:
            image, metadata = self._load_dicom(str(file_path))
            modality = "dicom"
        else:
            image, metadata = self._load_standard_image(str(file_path))
            modality = "xray"

        # ---- Validate dimensions ----
        h, w = image.shape[:2]
        self._validate_dimensions(str(file_path), h, w)

        # ---- Modality check ----
        self._check_xray_likelihood(image, str(file_path), run_warnings)

        # Keep an immutable copy of the original
        original = image.copy()

        # ---- Run preprocessing pipeline ----
        steps = self._pipeline.run(image)
        processed: np.ndarray = steps.get("normalized", steps.get("resized", image))

        # ---- Locate joint ROI ----
        roi_image, roi_bbox = self._pipeline.locate_joint_region(processed)

        if roi_image is None:
            run_warnings.append(
                "Joint ROI could not be automatically located; full image will be used."
            )

        result = PreprocessingResult(
            original=original,
            processed=processed,
            roi_image=roi_image,
            roi_bbox=roi_bbox,
            steps=steps,
            metadata=metadata,
            warnings=run_warnings,
            image_path=str(file_path),
            modality=modality,
            study_type="knee_xray",
        )
        logger.info(
            "Loaded '%s': shape=%s modality=%s warnings=%d",
            file_path.name,
            original.shape,
            modality,
            len(run_warnings),
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_dicom(self, path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load a DICOM file, delegating to :mod:`preprocessing.dicom`.

        Parameters
        ----------
        path:
            Path to the ``.dcm`` file.

        Returns
        -------
        Tuple[np.ndarray, Dict[str, Any]]
            ``(grayscale_uint8_image, safe_metadata)``.

        Raises
        ------
        DICOMLoadError
            Propagated from :func:`~preprocessing.dicom.load_dicom`.
        CorruptedImageError
            If the returned array is not 2-D.
        """
        image, metadata = load_dicom(path)
        if image.ndim != 2:
            raise CorruptedImageError(
                path, f"DICOM loader returned non-2D array with shape {image.shape}."
            )
        return image, metadata

    def _load_standard_image(self, path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load a JPEG/PNG file with OpenCV and convert to grayscale uint8.

        Parameters
        ----------
        path:
            Path to the image file.

        Returns
        -------
        Tuple[np.ndarray, Dict[str, Any]]
            ``(grayscale_uint8_image, minimal_metadata)``.

        Raises
        ------
        CorruptedImageError
            If OpenCV cannot decode the file.
        """
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise CorruptedImageError(
                path,
                "OpenCV returned None — file is corrupted or not a valid image.",
            )

        if raw.dtype != np.uint8:
            # Normalise 16-bit etc. to uint8
            logger.debug("Converting image dtype %s → uint8.", raw.dtype)
            raw = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if raw.ndim == 3 and raw.shape[2] == 4:
            # RGBA → BGR drop alpha
            raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

        if raw.ndim == 3:
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        elif raw.ndim == 2:
            gray = raw
        else:
            raise CorruptedImageError(
                path, f"Unexpected array dimensions: {raw.shape}."
            )

        metadata: Dict[str, Any] = {
            "source_path": path,
            "original_shape": gray.shape,
            "original_dtype": str(raw.dtype),
            "original_channels": raw.shape[2] if raw.ndim == 3 else 1,
        }
        return gray, metadata

    def _validate_dimensions(self, path: str, h: int, w: int) -> None:
        """Raise :class:`ImageDimensionError` if ``(h, w)`` are out of bounds.

        Parameters
        ----------
        path:
            Image path (for error messages).
        h:
            Image height in pixels.
        w:
            Image width in pixels.

        Raises
        ------
        ImageDimensionError
            If ``h`` or ``w`` fall outside the configured min/max range.
        """
        min_h, min_w = self._min_size
        max_h, max_w = self._max_size

        if h < min_h or w < min_w or h > max_h or w > max_w:
            raise ImageDimensionError(
                path,
                shape=(h, w),
                min_size=self._min_size,
                max_size=self._max_size,
            )

    @staticmethod
    def _check_xray_likelihood(
        image: np.ndarray,
        path: str,
        warnings_out: list[str],
    ) -> None:
        """Emit an :class:`ImageModalityWarning` if the image is unlikely to be an X-ray.

        Heuristics applied:
        - Very low mean intensity (near-black image).
        - Very high mean intensity (near-white image).
        - If the original file was colour, log a note (already converted).

        Parameters
        ----------
        image:
            Grayscale ``uint8`` image array.
        path:
            Image path (for warning messages).
        warnings_out:
            Mutable list; any generated warning strings are appended here.
        """
        mean_intensity = float(np.mean(image))
        std_intensity = float(np.std(image))

        if mean_intensity < _MIN_MEAN_INTENSITY:
            msg = (
                f"Image '{Path(path).name}' has very low mean intensity "
                f"({mean_intensity:.1f}) — may not be a valid X-ray."
            )
            warnings.warn(msg, ImageModalityWarning, stacklevel=4)
            warnings_out.append(msg)
            logger.warning(msg)

        if mean_intensity > _MAX_MEAN_INTENSITY:
            msg = (
                f"Image '{Path(path).name}' has very high mean intensity "
                f"({mean_intensity:.1f}) — may be an overexposed or inverted image."
            )
            warnings.warn(msg, ImageModalityWarning, stacklevel=4)
            warnings_out.append(msg)
            logger.warning(msg)

        if std_intensity < 5.0:
            msg = (
                f"Image '{Path(path).name}' has very low pixel standard deviation "
                f"({std_intensity:.1f}) — image may be uniform or blank."
            )
            warnings.warn(msg, ImageModalityWarning, stacklevel=4)
            warnings_out.append(msg)
            logger.warning(msg)

        logger.debug(
            "X-ray likelihood check — mean=%.1f std=%.1f for '%s'.",
            mean_intensity,
            std_intensity,
            path,
        )
