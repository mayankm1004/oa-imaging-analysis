"""
tests/test_preprocessing.py
=============================
Unit tests for preprocessing/enhancement.py and preprocessing/normalization.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing.enhancement import OAPreprocessingPipeline
from preprocessing.normalization import (
    imagenet_normalize,
    minmax_normalize,
    prepare_for_model,
    zscore_normalize,
)

_PIPELINE_STEP_KEYS = {"grayscale", "denoised", "clahe", "sharpened", "resized"}


class TestOAPreprocessingPipeline:
    """Tests for the OpenCV preprocessing pipeline."""

    def test_run_returns_dict_with_expected_keys(
        self, synthetic_xray_image: np.ndarray, sample_config: dict
    ) -> None:
        pipeline = OAPreprocessingPipeline(config=sample_config["preprocessing"])
        steps = pipeline.run(synthetic_xray_image)
        assert isinstance(steps, dict)
        assert _PIPELINE_STEP_KEYS.issubset(set(steps.keys()))

    def test_grayscale_output_is_2d(
        self, synthetic_xray_image: np.ndarray, sample_config: dict
    ) -> None:
        pipeline = OAPreprocessingPipeline(config=sample_config["preprocessing"])
        steps = pipeline.run(synthetic_xray_image)
        assert steps["grayscale"].ndim == 2

    def test_resized_output_matches_target_size(
        self, synthetic_xray_image: np.ndarray, sample_config: dict
    ) -> None:
        pipeline = OAPreprocessingPipeline(config=sample_config["preprocessing"])
        steps = pipeline.run(synthetic_xray_image)
        h, w = sample_config["preprocessing"]["target_size"]
        assert steps["resized"].shape == (h, w)

    def test_clahe_increases_contrast(
        self, synthetic_xray_image: np.ndarray, sample_config: dict
    ) -> None:
        """CLAHE output should have equal or greater std than grayscale."""
        pipeline = OAPreprocessingPipeline(config=sample_config["preprocessing"])
        steps = pipeline.run(synthetic_xray_image)
        std_gray = float(steps["grayscale"].std())
        std_clahe = float(steps["clahe"].std())
        # CLAHE should not reduce overall std significantly
        assert std_clahe >= std_gray * 0.5

    def test_all_outputs_are_uint8(
        self, synthetic_xray_image: np.ndarray, sample_config: dict
    ) -> None:
        pipeline = OAPreprocessingPipeline(config=sample_config["preprocessing"])
        steps = pipeline.run(synthetic_xray_image)
        for key, arr in steps.items():
            assert arr.dtype == np.uint8, f"Step '{key}' dtype is {arr.dtype}"

    def test_locate_joint_region_returns_valid_or_none(
        self, synthetic_xray_image: np.ndarray, sample_config: dict
    ) -> None:
        pipeline = OAPreprocessingPipeline(config=sample_config["preprocessing"])
        roi_img, roi_bbox = pipeline.locate_joint_region(synthetic_xray_image)
        if roi_bbox is not None:
            x, y, w, h = roi_bbox
            assert w > 0 and h > 0
            assert isinstance(roi_img, np.ndarray)
        else:
            assert roi_img is None

    def test_rgb_input_also_works(
        self, synthetic_xray_image: np.ndarray, sample_config: dict
    ) -> None:
        import cv2
        rgb = cv2.cvtColor(synthetic_xray_image, cv2.COLOR_GRAY2BGR)
        pipeline = OAPreprocessingPipeline(config=sample_config["preprocessing"])
        steps = pipeline.run(rgb)
        assert steps["grayscale"].ndim == 2


class TestNormalization:
    """Tests for normalization utilities."""

    def test_imagenet_normalize_output_shape(self) -> None:
        img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        out = imagenet_normalize(img)
        assert out.shape == (256, 256, 3)
        assert out.dtype == np.float32

    def test_minmax_normalize_range(self) -> None:
        img = np.array([0, 50, 100, 200, 255], dtype=np.uint8)
        out = minmax_normalize(img)
        assert float(out.min()) == pytest.approx(0.0)
        assert float(out.max()) == pytest.approx(1.0)

    def test_zscore_normalize_mean_std(self) -> None:
        rng = np.random.default_rng(0)
        img = rng.integers(50, 200, (64, 64), dtype=np.uint8)
        out = zscore_normalize(img)
        assert abs(float(out.mean())) < 0.01
        assert abs(float(out.std()) - 1.0) < 0.01

    def test_minmax_constant_image_returns_zeros(self) -> None:
        img = np.full((32, 32), 128, dtype=np.uint8)
        out = minmax_normalize(img)
        assert (out == 0).all()

    def test_prepare_for_model_output_shape(self) -> None:
        img = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        out = prepare_for_model(img, target_size=(224, 224))
        assert out.shape == (1, 3, 224, 224)
        assert out.dtype == np.float32
