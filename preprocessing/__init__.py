"""
preprocessing/__init__.py
=========================
Public API for the OA Detection preprocessing package.

Exports the canonical :class:`PreprocessingResult` dataclass that all
downstream modules (inference, reporting, UI) must consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = ["PreprocessingResult"]


@dataclass
class PreprocessingResult:
    """Container for all artefacts produced by the preprocessing pipeline.

    Attributes
    ----------
    original:
        The raw image exactly as loaded from disk, shape ``(H, W)``,
        dtype ``uint8`` (grayscale).  Never modified in-place.
    processed:
        The fully-preprocessed image ready for model input,
        shape ``(H, W)``, dtype ``uint8``.
    roi_image:
        The joint region-of-interest crop, shape ``(H', W')``, dtype
        ``uint8``, or ``None`` if the ROI could not be located.
    roi_bbox:
        Bounding box of the ROI as ``(x, y, w, h)`` in pixel coordinates
        relative to *original*, or ``None`` if ROI detection failed.
    steps:
        Ordered mapping of intermediate processing step names to their
        output images.  Keys: ``'grayscale'``, ``'denoised'``,
        ``'clahe'``, ``'sharpened'``, ``'resized'``, ``'normalized'``.
    metadata:
        Safe image / DICOM metadata.  **Must never contain PII fields**
        (see ``configs/oa_model.yaml`` → ``dicom.pii_fields``).
    warnings:
        Human-readable quality or processing warnings emitted during the
        pipeline run.  Empty list means no issues detected.
    image_path:
        Absolute path (as string) of the source image file.
    modality:
        Image modality: ``'xray'`` for standard image files,
        ``'dicom'`` for DICOM files.
    study_type:
        Clinical study type.  Defaults to ``'knee_xray'``.
    """

    original: np.ndarray
    processed: np.ndarray
    roi_image: Optional[np.ndarray]
    roi_bbox: Optional[Tuple[int, int, int, int]]
    steps: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    image_path: str = ""
    modality: str = "xray"
    study_type: str = "knee_xray"
