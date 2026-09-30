"""
tests/test_model.py
====================
Unit tests for models/classifier.py and models/explainability.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import ExplainabilityResult, ModelPrediction
from models.classifier import EfficientNetOAClassifier
from models.explainability import GradCAMExplainer
from preprocessing import PreprocessingResult


# ---------------------------------------------------------------------------
# Helper: create a minimal PreprocessingResult for testing
# ---------------------------------------------------------------------------


def _make_preprocessing_result(h: int = 224, w: int = 224) -> PreprocessingResult:
    rng = np.random.default_rng(99)
    img = rng.integers(0, 255, (h, w), dtype=np.uint8)
    return PreprocessingResult(
        original=img,
        processed=img.copy(),
        roi_image=None,
        roi_bbox=None,
        steps={"grayscale": img, "denoised": img, "clahe": img, "sharpened": img, "resized": img},
        metadata={},
        warnings=[],
        image_path="/synthetic/test.png",
        modality="xray",
        study_type="knee_xray",
    )


class TestEfficientNetOAClassifier:
    """Tests for the EfficientNet-B4 classifier."""

    @pytest.fixture(scope="class")
    def loaded_model(self, sample_config: dict) -> EfficientNetOAClassifier:
        # Use small input size for speed
        sample_config = dict(sample_config)
        sample_config["model"] = dict(sample_config["model"])
        sample_config["model"]["input_size"] = [224, 224]
        model = EfficientNetOAClassifier(config=sample_config)
        model.load(checkpoint_path=None)  # ImageNet weights
        return model

    def test_model_loads_without_checkpoint(self, loaded_model: EfficientNetOAClassifier) -> None:
        assert loaded_model.is_loaded()

    def test_predict_returns_model_prediction(
        self, loaded_model: EfficientNetOAClassifier
    ) -> None:
        prep = _make_preprocessing_result(224, 224)
        pred = loaded_model.predict(prep)
        assert isinstance(pred, ModelPrediction)

    def test_oa_score_in_range(self, loaded_model: EfficientNetOAClassifier) -> None:
        prep = _make_preprocessing_result(224, 224)
        pred = loaded_model.predict(prep)
        assert 0.0 <= pred.oa_score <= 1.0

    def test_confidence_level_valid(self, loaded_model: EfficientNetOAClassifier) -> None:
        prep = _make_preprocessing_result(224, 224)
        pred = loaded_model.predict(prep)
        assert pred.confidence_level in {"low", "moderate", "high"}

    def test_requires_finetuning_true_without_checkpoint(
        self, loaded_model: EfficientNetOAClassifier
    ) -> None:
        prep = _make_preprocessing_result(224, 224)
        pred = loaded_model.predict(prep)
        assert pred.requires_finetuning is True

    def test_raw_probabilities_sum_to_one(
        self, loaded_model: EfficientNetOAClassifier
    ) -> None:
        prep = _make_preprocessing_result(224, 224)
        pred = loaded_model.predict(prep)
        total = sum(pred.raw_probabilities.values())
        assert abs(total - 1.0) < 1e-5

    def test_model_name_is_string(self, loaded_model: EfficientNetOAClassifier) -> None:
        assert isinstance(loaded_model.name, str)
        assert len(loaded_model.name) > 0


class TestGradCAMExplainer:
    """Tests for the GradCAM explainability module."""

    @pytest.fixture(scope="class")
    def explainer_and_tensor(self, sample_config: dict):
        sample_config = dict(sample_config)
        sample_config["model"] = dict(sample_config["model"])
        sample_config["model"]["input_size"] = [224, 224]
        model = EfficientNetOAClassifier(config=sample_config)
        model.load(checkpoint_path=None)

        # Get target layer name from model
        target_layer = model.gradcam_target_layer
        explainer = GradCAMExplainer(
            model=model.backbone,
            target_layer_name=target_layer,
            device=str(model.device),
        )

        from preprocessing.normalization import prepare_for_model
        img = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        tensor = torch.from_numpy(prepare_for_model(img, (224, 224)))

        yield explainer, tensor
        explainer.remove_hooks()

    def test_gradcam_output_shape(self, explainer_and_tensor) -> None:
        explainer, tensor = explainer_and_tensor
        heatmap = explainer.generate_gradcam(tensor, target_class=1)
        assert heatmap.ndim == 2
        assert heatmap.shape == (224, 224)

    def test_gradcam_values_in_range(self, explainer_and_tensor) -> None:
        explainer, tensor = explainer_and_tensor
        heatmap = explainer.generate_gradcam(tensor, target_class=1)
        assert float(heatmap.min()) >= 0.0
        assert float(heatmap.max()) <= 1.0

    def test_gradcam_dtype_float32(self, explainer_and_tensor) -> None:
        explainer, tensor = explainer_and_tensor
        heatmap = explainer.generate_gradcam(tensor, target_class=1)
        assert heatmap.dtype == np.float32

    def test_overlay_output_shape(self, explainer_and_tensor) -> None:
        explainer, tensor = explainer_and_tensor
        heatmap = explainer.generate_gradcam(tensor, target_class=1)
        img = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        overlay = explainer.create_overlay(img, heatmap)
        assert overlay.shape == (224, 224, 3)
        assert overlay.dtype == np.uint8

    def test_explain_returns_explainability_result(self, explainer_and_tensor) -> None:
        explainer, tensor = explainer_and_tensor
        prep = _make_preprocessing_result(224, 224)
        result = explainer.explain(prep, tensor, method="gradcam")
        assert isinstance(result, ExplainabilityResult)

    def test_warning_is_mandatory_string(self, explainer_and_tensor) -> None:
        explainer, tensor = explainer_and_tensor
        prep = _make_preprocessing_result(224, 224)
        result = explainer.explain(prep, tensor, method="gradcam")
        assert "model explanation" in result.warning.lower()
        assert len(result.warning) > 10
