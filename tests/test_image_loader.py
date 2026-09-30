"""
tests/test_image_loader.py
===========================
Unit tests for preprocessing/image_loader.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing import PreprocessingResult
from preprocessing.image_loader import ImageLoader, UnsupportedFormatError, CorruptedImageError


class TestImageLoader:
    """Tests for the ImageLoader class."""

    def test_load_png_returns_preprocessing_result(
        self, synthetic_xray_path: str, sample_config: dict
    ) -> None:
        loader = ImageLoader(config=sample_config)
        result = loader.load(synthetic_xray_path)
        assert isinstance(result, PreprocessingResult)

    def test_load_jpeg_returns_preprocessing_result(
        self, synthetic_jpeg_path: str, sample_config: dict
    ) -> None:
        loader = ImageLoader(config=sample_config)
        result = loader.load(synthetic_jpeg_path)
        assert isinstance(result, PreprocessingResult)

    def test_load_dicom_returns_preprocessing_result(
        self, synthetic_dicom_path: str, sample_config: dict
    ) -> None:
        loader = ImageLoader(config=sample_config)
        result = loader.load(synthetic_dicom_path)
        assert isinstance(result, PreprocessingResult)
        assert result.modality == "dicom"

    def test_result_original_is_ndarray(
        self, synthetic_xray_path: str, sample_config: dict
    ) -> None:
        loader = ImageLoader(config=sample_config)
        result = loader.load(synthetic_xray_path)
        assert isinstance(result.original, np.ndarray)
        assert result.original.dtype == np.uint8

    def test_result_processed_is_ndarray(
        self, synthetic_xray_path: str, sample_config: dict
    ) -> None:
        loader = ImageLoader(config=sample_config)
        result = loader.load(synthetic_xray_path)
        assert isinstance(result.processed, np.ndarray)

    def test_result_modality_xray(
        self, synthetic_xray_path: str, sample_config: dict
    ) -> None:
        loader = ImageLoader(config=sample_config)
        result = loader.load(synthetic_xray_path)
        assert result.modality == "xray"

    def test_result_study_type_default(
        self, synthetic_xray_path: str, sample_config: dict
    ) -> None:
        loader = ImageLoader(config=sample_config)
        result = loader.load(synthetic_xray_path)
        assert result.study_type == "knee_xray"

    def test_image_path_preserved(
        self, synthetic_xray_path: str, sample_config: dict
    ) -> None:
        loader = ImageLoader(config=sample_config)
        result = loader.load(synthetic_xray_path)
        assert result.image_path == synthetic_xray_path

    def test_nonexistent_file_raises(self, sample_config: dict) -> None:
        loader = ImageLoader(config=sample_config)
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/image.png")

    def test_unsupported_format_raises(self, tmp_path: Path, sample_config: dict) -> None:
        # Create a .txt file
        fake = tmp_path / "image.txt"
        fake.write_text("not an image")
        loader = ImageLoader(config=sample_config)
        with pytest.raises(UnsupportedFormatError):
            loader.load(str(fake))

    def test_corrupted_image_raises(
        self, corrupted_image_path: str, sample_config: dict
    ) -> None:
        loader = ImageLoader(config=sample_config)
        with pytest.raises(CorruptedImageError):
            loader.load(corrupted_image_path)

    def test_warnings_list_present(
        self, synthetic_xray_path: str, sample_config: dict
    ) -> None:
        loader = ImageLoader(config=sample_config)
        result = loader.load(synthetic_xray_path)
        assert isinstance(result.warnings, list)

    def test_metadata_no_pii(
        self, synthetic_dicom_path: str, sample_config: dict
    ) -> None:
        """DICOM metadata must not contain any PII fields."""
        loader = ImageLoader(config=sample_config)
        result = loader.load(synthetic_dicom_path)
        pii_fields = {
            "PatientName", "PatientID", "PatientBirthDate",
            "PatientSex", "PatientAge", "InstitutionName",
        }
        for field in pii_fields:
            assert field not in result.metadata, f"PII field '{field}' found in metadata"
