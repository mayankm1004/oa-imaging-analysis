"""
reports/report_generator.py — OA Imaging Research Report Generator.

Generates structured JSON and human-readable HTML reports from
OAAnalysisResult objects. All reports include full medical disclaimers
and clearly distinguish model predictions from clinical diagnoses.
"""

from __future__ import annotations

import json
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inline disclaimer helper (mirrors app/components/disclaimer.py)
# ---------------------------------------------------------------------------

_DISCLAIMER_TEXT: str = (
    "This system is a research/educational decision-support prototype. "
    "It has NOT been validated for clinical use, approved by any "
    "regulatory body, or tested for diagnostic accuracy in any "
    "specific population. All model outputs are predictions that "
    "require review and interpretation by a qualified radiologist "
    "or clinician. The presence or absence of model-derived risk "
    "markers does NOT constitute a medical diagnosis of "
    "osteoarthritis or any other condition."
)

_LIMITATIONS: List[str] = [
    "Model trained on ImageNet weights only — task-specific fine-tuning required "
    "before any research or clinical application.",
    "No prospective clinical validation has been performed.",
    "KL grade prediction requires dedicated fine-tuning on labelled OA datasets "
    "(e.g., OAI, MOST); it is NOT available in this prototype.",
    "Performance on DICOM images from sources other than the training distribution "
    "is unknown.",
    "Risk-marker detection is heuristic/proxy-based and must not be used as a "
    "standalone diagnostic criterion.",
    "The system does not account for patient history, symptoms, or laboratory findings.",
    "Explainability heat-maps (Grad-CAM) highlight regions that influenced the model "
    "prediction — they do NOT represent anatomical ground truth.",
    "Results may vary across hardware, software versions, and random seeds.",
]


class ReportGenerator:
    """Generates JSON and HTML research reports from OA analysis results.

    Reports include full medical disclaimers and clearly distinguish
    between model predictions and clinical diagnoses.

    Args:
        output_dir: Directory where reports will be saved. Created if absent.
        template_dir: Directory containing Jinja2 HTML templates.
            Defaults to the ``reports/`` folder alongside this module.
    """

    def __init__(
        self,
        output_dir: str = "reports/output",
        template_dir: Optional[str] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Resolve template directory
        if template_dir is None:
            template_dir = str(Path(__file__).parent)
        self.template_dir = Path(template_dir)

        # Lazy-import Jinja2 so the module remains importable without it
        self._jinja_env: Optional[Any] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_json_report(self, result: Any) -> Dict[str, Any]:
        """Generate a structured JSON report from an OAAnalysisResult.

        The returned dictionary matches the canonical report schema and is
        suitable for serialisation, storage in a research database, or
        downstream processing.

        Args:
            result: An ``OAAnalysisResult`` instance.

        Returns:
            Dictionary conforming to the OA report JSON schema, including
            ``report_id``, ``oa_related_score``, ``risk_markers``,
            ``requires_clinical_review``, ``disclaimer``, and
            ``limitations`` fields.
        """
        markers_dict: Dict[str, Dict[str, Any]] = {}
        try:
            rm_dict = result.risk_markers.to_dict()
            for marker_name, marker_data in rm_dict.items():
                if isinstance(marker_data, dict):
                    markers_dict[marker_name] = {
                        "status": marker_data.get("status", "uncertain"),
                        "confidence": float(marker_data.get("confidence", 0.0)),
                        "note": marker_data.get("note", ""),
                    }
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not extract risk markers: %s", exc)

        report: Dict[str, Any] = {
            "report_id": str(uuid.uuid4()),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "study_type": getattr(result, "study_type", "unknown"),
            "joint_type": getattr(result, "joint_type", "unknown"),
            "model": getattr(result, "model_name", "unknown"),
            "model_version": getattr(result, "model_version", "unknown"),
            "requires_finetuning": getattr(
                result.prediction, "requires_finetuning", True
            ),
            "oa_related_score": float(
                getattr(result.prediction, "oa_score", 0.0)
            ),
            "confidence_level": getattr(
                result.prediction, "confidence_level", "low"
            ),
            "risk_markers": markers_dict,
            # KL grade: always null unless a task-specific fine-tuned model is used
            "kl_grade": None,
            "kl_grade_confidence": None,
            "requires_clinical_review": True,
            "processing_warnings": list(getattr(result, "warnings", [])),
            "explainability_method": getattr(
                result.explainability, "method", "gradcam"
            ),
            "disclaimer": _DISCLAIMER_TEXT,
            "limitations": self._get_limitations(),
        }
        return report

    def save_json_report(
        self,
        result: Any,
        filename: Optional[str] = None,
    ) -> str:
        """Save a JSON report to disk.

        Args:
            result: An ``OAAnalysisResult`` instance.
            filename: Optional filename (without path). If omitted, a
                timestamped filename is generated automatically.

        Returns:
            Absolute path string to the saved JSON file.
        """
        if filename is None:
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            filename = f"oa_report_{ts}.json"

        report_dict = self.generate_json_report(result)
        out_path = self.output_dir / filename

        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(report_dict, fh, indent=2, ensure_ascii=False)

        logger.info("JSON report saved: %s", out_path)
        return str(out_path)

    def generate_html_report(
        self,
        result: Any,
        heatmap_path: Optional[str] = None,
    ) -> str:
        """Generate a human-readable HTML report.

        If Jinja2 is available and ``reports/template.html`` exists, the
        template is rendered. Otherwise a plain-HTML fallback is produced.

        Args:
            result: An ``OAAnalysisResult`` instance.
            heatmap_path: Optional path to a Grad-CAM heat-map image that
                will be embedded as a ``<img>`` tag in the report.

        Returns:
            HTML document as a string.
        """
        report_dict = self.generate_json_report(result)

        # Build template context
        context = self._build_template_context(result, report_dict, heatmap_path)

        # Try Jinja2 rendering first
        try:
            env = self._get_jinja_env()
            template = env.get_template("template.html")
            return template.render(**context)
        except Exception as exc:
            logger.warning(
                "Jinja2 rendering failed (%s); using plain-HTML fallback.", exc
            )
            return self._render_plain_html(context)

    def save_html_report(
        self,
        result: Any,
        filename: Optional[str] = None,
    ) -> str:
        """Save an HTML report to disk.

        Args:
            result: An ``OAAnalysisResult`` instance.
            filename: Optional filename. Auto-generated if omitted.

        Returns:
            Absolute path string to the saved HTML file.
        """
        if filename is None:
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            filename = f"oa_report_{ts}.html"

        html = self.generate_html_report(result)
        out_path = self.output_dir / filename

        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(html)

        logger.info("HTML report saved: %s", out_path)
        return str(out_path)

    def _get_limitations(self) -> List[str]:
        """Return the standard limitations list included in all reports.

        Returns:
            List of limitation strings covering model constraints,
            validation status, and appropriate use.
        """
        return list(_LIMITATIONS)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_jinja_env(self) -> Any:
        """Lazily initialise and return the Jinja2 Environment.

        Raises:
            ImportError: If ``jinja2`` is not installed.
            FileNotFoundError: If the template directory does not exist.
        """
        if self._jinja_env is None:
            from jinja2 import Environment, FileSystemLoader  # noqa: PLC0415

            self._jinja_env = Environment(
                loader=FileSystemLoader(str(self.template_dir)),
                autoescape=True,
            )
        return self._jinja_env

    def _build_template_context(
        self,
        result: Any,
        report_dict: Dict[str, Any],
        heatmap_path: Optional[str],
    ) -> Dict[str, Any]:
        """Construct the Jinja2 / plain-HTML template context dictionary.

        Args:
            result: An ``OAAnalysisResult`` instance.
            report_dict: Pre-computed JSON report dictionary.
            heatmap_path: Optional path to the heat-map image.

        Returns:
            Context dictionary for template rendering.
        """
        # Format risk markers for display
        marker_rows: List[Dict[str, Any]] = []
        for name, data in report_dict["risk_markers"].items():
            label = name.replace("_", " ").title()
            marker_rows.append(
                {
                    "name": label,
                    "status": data["status"],
                    "confidence": f"{data['confidence'] * 100:.1f}%",
                    "note": data.get("note", ""),
                }
            )

        return {
            "result": {
                "oa_score": f"{report_dict['oa_related_score']:.3f}",
                "oa_score_pct": f"{report_dict['oa_related_score'] * 100:.1f}",
                "confidence_level": report_dict["confidence_level"],
                "markers": marker_rows,
                "timestamp": report_dict["generated_at"],
                "study_type": report_dict["study_type"],
                "joint_type": report_dict["joint_type"],
                "model": report_dict["model"],
                "model_version": report_dict["model_version"],
                "requires_finetuning": report_dict["requires_finetuning"],
                "kl_grade": report_dict["kl_grade"],
                "kl_grade_confidence": report_dict["kl_grade_confidence"],
                "explainability_method": report_dict["explainability_method"],
                "warnings": report_dict["processing_warnings"],
                "disclaimer": report_dict["disclaimer"],
                "limitations": report_dict["limitations"],
                "report_id": report_dict["report_id"],
                "heatmap_path": heatmap_path,
                "requires_clinical_review": True,
            }
        }

    def _render_plain_html(self, context: Dict[str, Any]) -> str:
        """Produce a self-contained HTML report without Jinja2.

        This fallback guarantees a usable HTML report even when the
        template file or Jinja2 package is unavailable.

        Args:
            context: Template context built by ``_build_template_context``.

        Returns:
            A complete HTML document string.
        """
        r = context["result"]

        marker_rows_html = ""
        status_colors = {
            "detected": "#FF8C00",
            "possible": "#FFD700",
            "uncertain": "#A9A9A9",
            "not_detected": "#228B22",
        }
        for m in r["markers"]:
            color = status_colors.get(m["status"], "#A9A9A9")
            marker_rows_html += (
                f"<tr>"
                f"<td>{m['name']}</td>"
                f"<td style='color:{color};font-weight:bold'>{m['status'].replace('_',' ').upper()}</td>"
                f"<td>{m['confidence']}</td>"
                f"<td>{m['note']}</td>"
                f"</tr>"
            )

        limitations_html = "".join(
            f"<li>{lim}</li>" for lim in r["limitations"]
        )
        warnings_html = (
            "".join(f"<li>{w}</li>" for w in r["warnings"])
            if r["warnings"]
            else "<li>None</li>"
        )

        heatmap_html = ""
        if r.get("heatmap_path"):
            heatmap_html = (
                f"<img src='{r['heatmap_path']}' alt='Grad-CAM heat-map' "
                f"style='max-width:400px;border:1px solid #ccc;'/>"
            )

        score_pct = r["oa_score_pct"]
        kl_display = (
            "Not available — requires task-specific fine-tuning"
            if r["kl_grade"] is None
            else str(r["kl_grade"])
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>OA Imaging Research Report</title>
<style>
  body{{font-family:Arial,sans-serif;max-width:900px;margin:40px auto;padding:0 20px;color:#222;}}
  h1{{color:#1a1a2e;border-bottom:3px solid #e74c3c;padding-bottom:10px;}}
  h2{{color:#1a1a2e;border-left:4px solid #3498db;padding-left:10px;margin-top:30px;}}
  .header-banner{{background:#fff3cd;border:2px solid #ffc107;padding:15px;border-radius:6px;margin-bottom:20px;font-weight:bold;text-align:center;font-size:1.1em;}}
  .disclaimer-box{{background:#fff0f0;border:2px solid #e74c3c;padding:15px;border-radius:6px;margin:20px 0;}}
  .disclaimer-box h3{{color:#c0392b;margin-top:0;}}
  table{{width:100%;border-collapse:collapse;margin:15px 0;}}
  th{{background:#2c3e50;color:#fff;padding:10px;text-align:left;}}
  td{{padding:9px;border-bottom:1px solid #ddd;}}
  tr:nth-child(even){{background:#f8f9fa;}}
  .score-bar-outer{{background:#ddd;border-radius:8px;height:22px;width:100%;margin:8px 0;}}
  .score-bar-inner{{background:linear-gradient(90deg,#27ae60,#e67e22,#e74c3c);height:22px;border-radius:8px;}}
  .badge{{display:inline-block;padding:4px 12px;border-radius:12px;font-weight:bold;font-size:.9em;}}
  .badge-low{{background:#d4edda;color:#155724;}}
  .badge-medium{{background:#fff3cd;color:#856404;}}
  .badge-high{{background:#f8d7da;color:#721c24;}}
  .meta-table td:first-child{{font-weight:bold;width:35%;color:#555;}}
  .kl-unavail{{color:#888;font-style:italic;}}
  footer{{margin-top:40px;padding:15px;background:#f0f0f0;border-radius:6px;font-size:.85em;color:#555;text-align:center;}}
</style>
</head>
<body>
<div class="header-banner">
  ⚠️ OA Imaging Research Report — NOT A MEDICAL DIAGNOSIS<br/>
  Clinical interpretation by a qualified radiologist is required.
</div>

<h1>🦴 Osteoarthritis Imaging Research Report</h1>

<h2>Study Information</h2>
<table class="meta-table">
  <tr><td>Report ID</td><td>{r['report_id']}</td></tr>
  <tr><td>Generated At</td><td>{r['timestamp']}</td></tr>
  <tr><td>Study Type</td><td>{r['study_type']}</td></tr>
  <tr><td>Joint Type</td><td>{r['joint_type']}</td></tr>
  <tr><td>Model</td><td>{r['model']}</td></tr>
  <tr><td>Model Version</td><td>{r['model_version']}</td></tr>
  <tr><td>Fine-tuning Required</td><td>{'Yes' if r['requires_finetuning'] else 'No'}</td></tr>
  <tr><td>Patient</td><td><em>REDACTED — No PII stored</em></td></tr>
</table>

<h2>OA-Related Score</h2>
<div class="score-bar-outer">
  <div class="score-bar-inner" style="width:{score_pct}%"></div>
</div>
<p><strong>Score:</strong> {r['oa_score']} &nbsp;&nbsp;
   <span class="badge badge-{r['confidence_level'].lower()}">{r['confidence_level'].upper()} CONFIDENCE</span>
</p>
<p><em>⚠️ This score reflects model output from ImageNet-pretrained weights only.
Task-specific fine-tuning is required for research-grade results.</em></p>

<h2>Risk Markers</h2>
<table>
  <tr><th>Marker</th><th>Status</th><th>Confidence</th><th>Note</th></tr>
  {marker_rows_html}
</table>

<h2>KL Grade</h2>
<p class="kl-unavail">
  ⛔ Not available — requires task-specific fine-tuning on labelled OA datasets
  (e.g., OAI / MOST). KL grade prediction is NOT supported in this prototype.
</p>

<h2>Explainability ({r['explainability_method'].upper()})</h2>
<p>The heat-map below highlights image regions that contributed to the model's
prediction. This is an algorithmic explanation only — it does NOT represent
anatomical ground truth or clinical findings.</p>
{heatmap_html if heatmap_html else '<p><em>Heat-map image not available in this report.</em></p>'}

<h2>Processing Warnings</h2>
<ul>{warnings_html}</ul>

<div class="disclaimer-box">
  <h3>⚠️ Medical Disclaimer</h3>
  <p>{r['disclaimer']}</p>
</div>

<h2>Limitations</h2>
<ul>{limitations_html}</ul>

<footer>
  Research prototype. Requires clinical validation before any diagnostic use.<br/>
  Report ID: {r['report_id']}
</footer>
</body>
</html>"""
