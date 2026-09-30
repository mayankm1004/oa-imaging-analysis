"""
tests/test_predictor.py
========================
Integration test: full end-to-end pipeline with a synthetic X-ray image.

This test exercises every stage of the pipeline:
  ImageLoader → Preprocessing → Segmentation → Model → Risk Markers →
  Grad-CAM → OAAnalysisResult

No real patient data is used.  All images are generated synthetically.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference import OAAnalysisResult
from inference.predictor import OAPredictor

_VALID_STATUSES = {"detected", "possible", "uncertain", "not_detected"}
_MARKER_NAMES = {
    "joint_space_narrowing",
    "osteophytes",
    "subchondral_sclerosis",
    "bone_deformity",
    "alignment_abnormality",
    "cystic_changes",
}


class TestOAPredictorIntegration:
    """End-to-end integration tests for OAPredictor."""

    @pytest.fixture()
    def predictor(self, sample_config: dict) -> OAPredictor:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "oa_model.yaml"
        return OAPredictor(config_path=str(config_path) if config_path.exists() else None)

    @pytest.fixture()
    def analysis_result(self, predictor: OAPredictor, synthetic_xray_path: str) -> OAAnalysisResult:
        return predictor.analyze(
            image_path=synthetic_xray_path,
            joint_type="knee",
            run_gradcam=True,
        )

    def test_returns_oa_analysis_result(self, analysis_result: OAAnalysisResult) -> None:
        assert isinstance(analysis_result, OAAnalysisResult)

    def test_requires_clinical_review_always_true(
        self, analysis_result: OAAnalysisResult
    ) -> None:
        assert analysis_result.requires_clinical_review is True

    def test_kl_grade_is_none(self, analysis_result: OAAnalysisResult) -> None:
        """KL grade must always be None unless a fine-tuned model is loaded."""
        assert analysis_result.kl_grade is None

    def test_oa_score_in_range(self, analysis_result: OAAnalysisResult) -> None:
        score = analysis_result.prediction.oa_score
        assert 0.0 <= score <= 1.0

    def test_confidence_level_valid(self, analysis_result: OAAnalysisResult) -> None:
        assert analysis_result.prediction.confidence_level in {"low", "moderate", "high"}

    def test_requires_finetuning_true(self, analysis_result: OAAnalysisResult) -> None:
        assert analysis_result.prediction.requires_finetuning is True

    def test_all_six_risk_markers_present(self, analysis_result: OAAnalysisResult) -> None:
        markers_dict = analysis_result.risk_markers.to_dict()
        assert _MARKER_NAMES == set(markers_dict.keys())

    def test_all_marker_statuses_valid(self, analysis_result: OAAnalysisResult) -> None:
        markers_dict = analysis_result.risk_markers.to_dict()
        for name, marker in markers_dict.items():
            assert marker["status"] in _VALID_STATUSES, (
                f"Marker '{name}' has invalid status '{marker['status']}'"
            )

    def test_all_marker_confidences_in_range(self, analysis_result: OAAnalysisResult) -> None:
        markers_dict = analysis_result.risk_markers.to_dict()
        for name, marker in markers_dict.items():
            conf = marker["confidence"]
            assert 0.0 <= conf <= 1.0, (
                f"Marker '{name}' confidence {conf} out of range"
            )

    def test_disclaimer_non_empty(self, analysis_result: OAAnalysisResult) -> None:
        assert len(analysis_result.disclaimer) > 20

    def test_explainability_heatmap_shape(self, analysis_result: OAAnalysisResult) -> None:
        heatmap = analysis_result.explainability.heatmap
        assert heatmap.ndim == 2
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0

    def test_explainability_warning_present(self, analysis_result: OAAnalysisResult) -> None:
        assert "model explanation" in analysis_result.explainability.warning.lower()

    def test_to_dict_is_json_serializable(self, analysis_result: OAAnalysisResult) -> None:
        import json
        d = analysis_result.to_dict()
        # Should not raise
        json_str = json.dumps(d)
        assert len(json_str) > 50

    def test_batch_analyze_returns_list(
        self, predictor: OAPredictor, synthetic_xray_path: str
    ) -> None:
        results = predictor.analyze_batch(
            image_paths=[synthetic_xray_path, synthetic_xray_path],
            joint_type="knee",
            run_gradcam=False,
        )
        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, OAAnalysisResult)
