"""
tests/test_report.py
=====================
Unit tests for reports/report_generator.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference import OAAnalysisResult, RiskMarkerResult, RiskMarkersReport
from models import ExplainabilityResult, ModelPrediction
from preprocessing import PreprocessingResult
from reports.report_generator import ReportGenerator


# ---------------------------------------------------------------------------
# Helper: construct a minimal OAAnalysisResult
# ---------------------------------------------------------------------------


def _make_marker(status: str = "uncertain") -> RiskMarkerResult:
    return RiskMarkerResult(
        status=status,
        confidence=0.5,
        source="model",
        note="Proxy measurement from image analysis.",
    )


def _make_analysis_result() -> OAAnalysisResult:
    img = np.zeros((224, 224), dtype=np.uint8)
    prep = PreprocessingResult(
        original=img,
        processed=img.copy(),
        roi_image=None,
        roi_bbox=None,
        steps={},
        metadata={"Modality": "DX"},
        warnings=[],
        image_path="/synthetic/test.png",
        modality="xray",
        study_type="knee_xray",
    )
    pred = ModelPrediction(
        oa_score=0.78,
        confidence_level="moderate",
        raw_probabilities={"normal": 0.22, "oa_related": 0.78},
        feature_vector=None,
        model_name="efficientnet_b4_oa",
        model_version="0.1.0",
        requires_finetuning=True,
    )
    markers = RiskMarkersReport(
        joint_space_narrowing=_make_marker("possible"),
        osteophytes=_make_marker("detected"),
        subchondral_sclerosis=_make_marker("uncertain"),
        bone_deformity=_make_marker("not_detected"),
        alignment_abnormality=_make_marker("uncertain"),
        cystic_changes=_make_marker("not_detected"),
    )
    heatmap = np.zeros((224, 224), dtype=np.float32)
    overlay = np.zeros((224, 224, 3), dtype=np.uint8)
    expl = ExplainabilityResult(
        heatmap=heatmap,
        overlay=overlay,
        method="gradcam",
        target_layer="features.8",
        warning="Model explanation only. Not anatomical proof.",
    )
    return OAAnalysisResult(
        preprocessing=prep,
        prediction=pred,
        risk_markers=markers,
        explainability=expl,
        kl_grade=None,
        kl_grade_confidence=None,
        requires_clinical_review=True,
        warnings=[],
        study_type="knee_xray",
        joint_type="knee",
        timestamp="2026-09-29T12:00:00Z",
        model_name="efficientnet_b4_oa",
        model_version="0.1.0",
        disclaimer="Research tool. Not a diagnosis.",
    )


class TestReportGenerator:
    """Tests for JSON and HTML report generation."""

    @pytest.fixture()
    def generator(self, tmp_path: Path) -> ReportGenerator:
        return ReportGenerator(output_dir=str(tmp_path))

    @pytest.fixture()
    def result(self) -> OAAnalysisResult:
        return _make_analysis_result()

    def test_generate_json_report_returns_dict(
        self, generator: ReportGenerator, result: OAAnalysisResult
    ) -> None:
        report = generator.generate_json_report(result)
        assert isinstance(report, dict)

    def test_json_report_has_required_keys(
        self, generator: ReportGenerator, result: OAAnalysisResult
    ) -> None:
        report = generator.generate_json_report(result)
        required = {
            "report_id", "generated_at", "study_type", "joint_type",
            "model", "model_version", "oa_related_score", "confidence_level",
            "risk_markers", "kl_grade", "requires_clinical_review", "disclaimer",
        }
        missing = required - set(report.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_json_requires_clinical_review_always_true(
        self, generator: ReportGenerator, result: OAAnalysisResult
    ) -> None:
        report = generator.generate_json_report(result)
        assert report["requires_clinical_review"] is True

    def test_json_kl_grade_is_null(
        self, generator: ReportGenerator, result: OAAnalysisResult
    ) -> None:
        report = generator.generate_json_report(result)
        assert report["kl_grade"] is None

    def test_json_disclaimer_non_empty(
        self, generator: ReportGenerator, result: OAAnalysisResult
    ) -> None:
        report = generator.generate_json_report(result)
        assert isinstance(report["disclaimer"], str)
        assert len(report["disclaimer"]) > 20

    def test_json_oa_score_in_range(
        self, generator: ReportGenerator, result: OAAnalysisResult
    ) -> None:
        report = generator.generate_json_report(result)
        assert 0.0 <= report["oa_related_score"] <= 1.0

    def test_save_json_writes_valid_file(
        self, generator: ReportGenerator, result: OAAnalysisResult, tmp_path: Path
    ) -> None:
        path = generator.save_json_report(result)
        assert Path(path).exists()
        with open(path, "r") as f:
            data = json.load(f)
        assert "requires_clinical_review" in data

    def test_generate_html_returns_string(
        self, generator: ReportGenerator, result: OAAnalysisResult
    ) -> None:
        html = generator.generate_html_report(result)
        assert isinstance(html, str)
        assert len(html) > 100

    def test_html_contains_disclaimer(
        self, generator: ReportGenerator, result: OAAnalysisResult
    ) -> None:
        html = generator.generate_html_report(result)
        assert "NOT A MEDICAL DIAGNOSIS" in html.upper() or "disclaimer" in html.lower()

    def test_html_contains_oa_score(
        self, generator: ReportGenerator, result: OAAnalysisResult
    ) -> None:
        html = generator.generate_html_report(result)
        assert "0.78" in html or "78" in html

    def test_json_risk_markers_has_six_markers(
        self, generator: ReportGenerator, result: OAAnalysisResult
    ) -> None:
        report = generator.generate_json_report(result)
        markers = report["risk_markers"]
        expected = {
            "joint_space_narrowing", "osteophytes", "subchondral_sclerosis",
            "bone_deformity", "alignment_abnormality", "cystic_changes",
        }
        assert expected == set(markers.keys())
