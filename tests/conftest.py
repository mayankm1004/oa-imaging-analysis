"""
tests/conftest.py
==================
Shared pytest fixtures for OA detection tests.

All fixtures generate synthetic data — no real patient images are used.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Synthetic image fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def synthetic_xray_image() -> np.ndarray:
    """Create a 512×512 synthetic grayscale image resembling a knee X-ray.

    The image contains:
    - Two bright rectangular 'bone' regions separated by a gap (joint space).
    - Gaussian noise to simulate X-ray grain.
    """
    img = np.zeros((512, 512), dtype=np.uint8)

    # Upper 'femur' bone region
    img[60:220, 140:370] = 195
    # Lower 'tibia' bone region
    img[265:420, 140:370] = 185
    # Thin cartilage/joint space gap between 220 and 265 (left intentionally dark)

    # Add realistic noise
    rng = np.random.default_rng(42)
    noise = rng.integers(0, 25, img.shape, dtype=np.uint8)
    img = np.clip(img.astype(np.int16) + noise - 12, 0, 255).astype(np.uint8)

    return img


@pytest.fixture()
def synthetic_xray_path(synthetic_xray_image: np.ndarray, tmp_path: Path) -> str:
    """Save synthetic X-ray to a temporary PNG file and return the path."""
    path = tmp_path / "test_xray.png"
    cv2.imwrite(str(path), synthetic_xray_image)
    return str(path)


@pytest.fixture()
def synthetic_jpeg_path(synthetic_xray_image: np.ndarray, tmp_path: Path) -> str:
    """Save synthetic X-ray as JPEG."""
    path = tmp_path / "test_xray.jpg"
    cv2.imwrite(str(path), synthetic_xray_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return str(path)


@pytest.fixture()
def synthetic_dicom_path(synthetic_xray_image: np.ndarray, tmp_path: Path) -> str:
    """Create a minimal synthetic DICOM file for testing."""
    try:
        import pydicom
        from pydicom import Dataset, FileDataset
        from pydicom.uid import ExplicitVRLittleEndian, generate_uid
        import pydicom._storage_sopclass_uids as _sc
    except ImportError:
        pytest.skip("pydicom not installed")

    path = tmp_path / "test.dcm"

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = _sc.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.is_implicit_VR = False
    ds.is_little_endian = True

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = 512
    ds.Columns = 512
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = synthetic_xray_image.tobytes()
    ds.WindowCenter = 128
    ds.WindowWidth = 255
    ds.Modality = "DX"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = generate_uid()
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.file_meta = file_meta

    ds.save_as(str(path), write_like_original=False)
    return str(path)


@pytest.fixture()
def corrupted_image_path(tmp_path: Path) -> str:
    """Create a file with garbage bytes (simulates a corrupted image)."""
    path = tmp_path / "corrupted.png"
    path.write_bytes(b"\x00\x01\x02\x03\x04\x05NOTANIMAGE")
    return str(path)


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sample_config() -> Dict[str, Any]:
    """Load the project config or fall back to a minimal test config."""
    import yaml

    config_path = Path(__file__).resolve().parents[1] / "configs" / "oa_model.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # Minimal fallback
    return {
        "project": {
            "study_type": "knee_xray",
            "disclaimer": "Research tool. Not a diagnosis.",
        },
        "model": {
            "name": "efficientnet_b4_oa",
            "backbone": "efficientnet_b4",
            "num_classes": 2,
            "pretrained": True,
            "checkpoint": None,
            "input_size": [224, 224],
            "dropout": 0.3,
            "requires_finetuning": True,
        },
        "preprocessing": {
            "target_size": [224, 224],
            "clahe_clip_limit": 2.0,
            "clahe_tile_grid": [8, 8],
            "denoise_h": 10,
            "denoise_template_window": 7,
            "denoise_search_window": 21,
            "sharpen_kernel_size": 3,
            "min_image_size": [64, 64],
            "max_image_size": [4096, 4096],
        },
        "risk_markers": {
            "joint_space_narrowing": {"threshold_possible": 0.45, "threshold_detected": 0.65},
            "osteophytes": {"threshold_possible": 0.40, "threshold_detected": 0.60},
            "subchondral_sclerosis": {"threshold_possible": 0.45, "threshold_detected": 0.65},
            "bone_deformity": {"threshold_possible": 0.40, "threshold_detected": 0.60},
            "alignment_abnormality": {"threshold_possible": 0.35, "threshold_detected": 0.55},
            "cystic_changes": {"threshold_possible": 0.40, "threshold_detected": 0.60},
        },
    }
