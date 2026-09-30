"""
preprocessing/dicom.py
======================
DICOM file loading, windowing, and safe metadata extraction for the
OA Detection preprocessing pipeline.

All public functions raise :class:`DICOMLoadError` on unrecoverable
failures and never silently swallow exceptions.

.. warning::
   PII fields are **never** included in the returned metadata dictionary.
   See :func:`extract_safe_metadata` and ``configs/oa_model.yaml``
   (``dicom.pii_fields``) for the full exclusion list.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pydicom
import pydicom.errors
from pydicom.pixel_data_handlers.util import apply_voi_lut

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PII fields that must never appear in returned metadata.
# Kept in sync with configs/oa_model.yaml → dicom.pii_fields
# ---------------------------------------------------------------------------
_DEFAULT_PII_FIELDS: frozenset[str] = frozenset(
    {
        "PatientName",
        "PatientID",
        "PatientBirthDate",
        "PatientSex",
        "PatientAge",
        "PatientWeight",
        "InstitutionName",
        "ReferringPhysicianName",
        "PerformingPhysicianName",
        "OperatorsName",
        "StudyID",
        "AccessionNumber",
    }
)

# Safe DICOM tags to include in returned metadata
_SAFE_TAGS: Tuple[str, ...] = (
    "Modality",
    "StudyDescription",
    "SeriesDescription",
    "BodyPartExamined",
    "ViewPosition",
    "ImageComments",
    "Manufacturer",
    "ManufacturerModelName",
    "SoftwareVersions",
    "KVP",
    "ExposureTime",
    "XRayTubeCurrent",
    "Exposure",
    "FocalSpots",
    "DistanceSourceToDetector",
    "DistanceSourceToPatient",
    "PixelSpacing",
    "ImagerPixelSpacing",
    "Rows",
    "Columns",
    "BitsAllocated",
    "BitsStored",
    "HighBit",
    "PixelRepresentation",
    "PhotometricInterpretation",
    "SamplesPerPixel",
    "WindowCenter",
    "WindowWidth",
    "RescaleIntercept",
    "RescaleSlope",
    "RescaleType",
    "SOPClassUID",
    "SOPInstanceUID",
    "StudyDate",
    "SeriesDate",
    "AcquisitionDate",
    "ContentDate",
    "StudyTime",
    "SeriesTime",
    "AcquisitionTime",
    "ContentTime",
    "InstanceNumber",
    "AcquisitionNumber",
    "SliceLocation",
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "SliceThickness",
    "SpacingBetweenSlices",
)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class DICOMLoadError(Exception):
    """Raised when a DICOM file cannot be loaded or parsed.

    Attributes
    ----------
    path:
        Filesystem path of the DICOM file that triggered the error.
    reason:
        Human-readable explanation of the failure.
    """

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to load DICOM '{path}': {reason}")


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def load_dicom(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load a DICOM file and return a normalised grayscale uint8 image.

    The function applies windowing (VOI LUT if present, otherwise the
    ``WindowCenter``/``WindowWidth`` attributes, or the configured
    defaults) and scales the result to ``[0, 255]``.

    Parameters
    ----------
    path:
        Absolute or relative filesystem path to the ``.dcm`` file.

    Returns
    -------
    image:
        Grayscale image array of shape ``(H, W)``, dtype ``uint8``.
    metadata:
        Safe (non-PII) metadata dictionary.  See
        :func:`extract_safe_metadata`.

    Raises
    ------
    DICOMLoadError
        If the file cannot be opened, has no pixel data, uses an
        unsupported compression scheme, or any other irrecoverable error
        occurs.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise DICOMLoadError(path, "File does not exist.")
    if not file_path.is_file():
        raise DICOMLoadError(path, "Path is not a regular file.")

    logger.info("Loading DICOM file: %s", path)

    try:
        ds = pydicom.dcmread(str(file_path), force=False)
    except pydicom.errors.InvalidDicomError as exc:
        raise DICOMLoadError(path, f"Not a valid DICOM file: {exc}") from exc
    except FileNotFoundError as exc:
        raise DICOMLoadError(path, f"File not found: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise DICOMLoadError(path, f"Unexpected read error: {exc}") from exc

    # Validate pixel data existence
    if not hasattr(ds, "PixelData"):
        raise DICOMLoadError(path, "DICOM dataset contains no PixelData element.")

    # Decompress if necessary
    try:
        pixel_array: np.ndarray = ds.pixel_array  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        # Common causes: missing JPEG2000/JPEG Lossless transfer syntax handler
        msg = (
            f"Cannot decode pixel data — transfer syntax "
            f"'{getattr(ds, 'file_meta', {}).get('TransferSyntaxUID', 'unknown')}' "
            f"may require an optional handler (e.g. pylibjpeg, gdcm): {exc}"
        )
        raise DICOMLoadError(path, msg) from exc

    logger.debug(
        "Pixel array shape=%s dtype=%s", pixel_array.shape, pixel_array.dtype
    )

    # Handle multi-frame — take the first frame only
    if pixel_array.ndim == 3:
        logger.warning(
            "Multi-frame DICOM detected (%d frames); using first frame only.", pixel_array.shape[0]
        )
        pixel_array = pixel_array[0]

    if pixel_array.ndim != 2:
        raise DICOMLoadError(
            path,
            f"Unexpected pixel array dimensionality after frame selection: {pixel_array.shape}.",
        )

    # Apply windowing
    window_center, window_width = _get_window_params(ds)
    windowed = apply_windowing(pixel_array.astype(np.float64), window_center, window_width)

    # Photometric inversion (MONOCHROME1 → invert to match MONOCHROME2)
    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2").strip()
    if photometric == "MONOCHROME1":
        logger.debug("Inverting MONOCHROME1 image to MONOCHROME2 convention.")
        windowed = 255 - windowed

    image = windowed.astype(np.uint8)
    metadata = extract_safe_metadata(ds)
    metadata["source_path"] = str(file_path)
    metadata["original_shape"] = pixel_array.shape
    metadata["window_center_used"] = window_center
    metadata["window_width_used"] = window_width

    logger.info("DICOM loaded successfully: shape=%s", image.shape)
    return image, metadata


def apply_windowing(
    pixel_array: np.ndarray,
    window_center: float,
    window_width: float,
) -> np.ndarray:
    """Apply linear windowing to a raw DICOM pixel array.

    Clips and linearly maps pixel values within the window to ``[0, 255]``.

    Parameters
    ----------
    pixel_array:
        Raw pixel values as a float64 array of shape ``(H, W)``.
        Rescale slope/intercept should already be applied before calling
        this function if needed.
    window_center:
        Centre of the display window.
    window_width:
        Total width of the display window.  Must be > 0.

    Returns
    -------
    np.ndarray
        Windowed image as ``uint8`` array of shape ``(H, W)``,
        values in ``[0, 255]``.

    Raises
    ------
    ValueError
        If ``window_width`` is not positive.
    """
    if window_width <= 0:
        raise ValueError(f"window_width must be positive, got {window_width}.")

    lower = window_center - window_width / 2.0
    upper = window_center + window_width / 2.0

    clipped = np.clip(pixel_array, lower, upper)
    scaled = (clipped - lower) / (upper - lower) * 255.0
    return scaled.astype(np.uint8)


def extract_safe_metadata(
    ds: pydicom.Dataset,
    extra_pii_fields: Optional[frozenset[str]] = None,
) -> Dict[str, Any]:
    """Extract a PII-free metadata dictionary from a pydicom Dataset.

    Only attributes listed in :data:`_SAFE_TAGS` are included.  Any
    attribute whose keyword matches a PII field is unconditionally
    excluded, even if it somehow appears in ``_SAFE_TAGS``.

    Parameters
    ----------
    ds:
        Parsed pydicom :class:`pydicom.Dataset`.
    extra_pii_fields:
        Optional additional field names to suppress beyond the defaults.

    Returns
    -------
    Dict[str, Any]
        Flat dictionary mapping DICOM keyword strings to their Python
        values.  Sequence elements and binary blobs are skipped.
    """
    pii_exclusions = _DEFAULT_PII_FIELDS
    if extra_pii_fields:
        pii_exclusions = pii_exclusions | extra_pii_fields

    metadata: Dict[str, Any] = {}
    for keyword in _SAFE_TAGS:
        if keyword in pii_exclusions:
            # Belt-and-suspenders: should never happen given our constants.
            logger.error(
                "BUG: keyword '%s' is both in _SAFE_TAGS and PII exclusion list — skipping.",
                keyword,
            )
            continue
        if not hasattr(ds, keyword):
            continue
        try:
            value = getattr(ds, keyword)
            # Skip raw bytes / Sequence elements
            if isinstance(value, (bytes, pydicom.sequence.Sequence)):
                continue
            # Convert pydicom-specific types to plain Python types
            if hasattr(value, "__iter__") and not isinstance(value, str):
                value = list(value)
            metadata[keyword] = value
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skipping DICOM attribute '%s': %s", keyword, exc)

    return metadata


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_window_params(ds: pydicom.Dataset) -> Tuple[float, float]:
    """Extract window centre and width from a DICOM dataset.

    Falls back to configured defaults if the attributes are absent or
    cannot be parsed.

    Parameters
    ----------
    ds:
        Parsed pydicom :class:`pydicom.Dataset`.

    Returns
    -------
    Tuple[float, float]
        ``(window_center, window_width)``
    """
    # Hardcoded defaults (matching configs/oa_model.yaml dicom section)
    default_center: float = 400.0
    default_width: float = 2000.0

    try:
        wc = ds.WindowCenter
        ww = ds.WindowWidth
        # These can be pydicom DSfloat or a list of values
        if hasattr(wc, "__iter__"):
            wc = list(wc)[0]
        if hasattr(ww, "__iter__"):
            ww = list(ww)[0]
        center = float(wc)
        width = float(ww)
        if width <= 0:
            raise ValueError(f"WindowWidth must be positive, got {width}.")
        logger.debug("Using DICOM window: center=%.1f width=%.1f", center, width)
        return center, width
    except (AttributeError, IndexError, ValueError, TypeError) as exc:
        logger.warning(
            "Could not read WindowCenter/WindowWidth (%s). "
            "Falling back to defaults: center=%.1f, width=%.1f.",
            exc,
            default_center,
            default_width,
        )
        return default_center, default_width
