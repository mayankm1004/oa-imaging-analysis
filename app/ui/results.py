"""
app/ui/results.py — Analysis results rendering for the OA Streamlit application.

Renders OAAnalysisResult objects as interactive Streamlit components,
including preprocessing comparison, risk marker tables, model score
displays, explainability visualisations, and report download buttons.
"""

from __future__ import annotations

import io
import json
import logging
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import streamlit as st

if TYPE_CHECKING:
    from reports.report_generator import ReportGenerator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Marker display helpers
# ---------------------------------------------------------------------------

_STATUS_EMOJI: Dict[str, str] = {
    "detected": "🟠",
    "possible": "🟡",
    "uncertain": "⚪",
    "not_detected": "🟢",
}

_STATUS_COLOR: Dict[str, str] = {
    "detected": "#FF8C00",
    "possible": "#DAA520",
    "uncertain": "#808080",
    "not_detected": "#228B22",
}

_MARKER_LABELS: Dict[str, str] = {
    "joint_space_narrowing": "Joint Space Narrowing",
    "osteophytes": "Osteophytes",
    "subchondral_sclerosis": "Subchondral Sclerosis",
    "bone_deformity": "Bone Deformity",
    "alignment_abnormality": "Alignment Abnormality",
    "cystic_changes": "Cystic Changes",
}


def _ndarray_to_png_bytes(arr: np.ndarray) -> bytes:
    """Convert a numpy image array to PNG bytes for st.image().

    Args:
        arr: Grayscale (H, W) or colour (H, W, 3) uint8 numpy array.

    Returns:
        PNG-encoded bytes.
    """
    import cv2  # noqa: PLC0415

    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)

    success, buf = cv2.imencode(".png", arr)
    if not success:
        raise ValueError("cv2.imencode failed for array with shape %s" % str(arr.shape))
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Public rendering functions
# ---------------------------------------------------------------------------

def render_preprocessing_comparison(result: Any) -> None:
    """Show original vs preprocessed image side-by-side and preprocessing steps.

    Displays the original input image alongside the fully preprocessed
    image in a two-column layout. All intermediate pipeline steps are
    shown inside a collapsible expander.

    Args:
        result: An ``OAAnalysisResult`` instance.

    Returns:
        None. Side-effects: renders Streamlit elements.
    """
    st.subheader("🔬 Image Preprocessing")

    prep = result.preprocessing

    col_orig, col_proc = st.columns(2)

    with col_orig:
        st.markdown("**Original Image**")
        try:
            orig_bytes = _ndarray_to_png_bytes(prep.original)
            st.image(orig_bytes, caption="Original upload", use_container_width=True)
        except Exception as exc:
            logger.warning("Could not display original image: %s", exc)
            st.warning("Original image not available for display.")

    with col_proc:
        st.markdown("**Preprocessed Image**")
        try:
            proc_bytes = _ndarray_to_png_bytes(prep.processed)
            st.image(proc_bytes, caption="After preprocessing pipeline", use_container_width=True)
        except Exception as exc:
            logger.warning("Could not display processed image: %s", exc)
            st.warning("Preprocessed image not available for display.")

    # ROI preview
    if prep.roi_image is not None:
        with st.expander("🎯 Region of Interest (ROI)", expanded=False):
            try:
                roi_bytes = _ndarray_to_png_bytes(prep.roi_image)
                st.image(roi_bytes, caption="Detected joint region", use_container_width=False)
                if prep.roi_bbox:
                    x, y, w, h = prep.roi_bbox
                    st.caption(f"ROI bounding box: x={x}, y={y}, w={w}, h={h}")
            except Exception as exc:
                logger.warning("Could not display ROI: %s", exc)

    # Preprocessing steps expander
    if prep.steps:
        with st.expander("🔧 Preprocessing Pipeline Steps", expanded=False):
            step_names = list(prep.steps.keys())
            if step_names:
                cols = st.columns(min(3, len(step_names)))
                for idx, step_name in enumerate(step_names):
                    step_arr = prep.steps[step_name]
                    col = cols[idx % len(cols)]
                    with col:
                        try:
                            step_bytes = _ndarray_to_png_bytes(step_arr)
                            st.image(
                                step_bytes,
                                caption=step_name.replace("_", " ").title(),
                                use_container_width=True,
                            )
                        except Exception as exc:
                            logger.warning("Could not display step %s: %s", step_name, exc)
                            st.caption(f"{step_name}: display error")

    # Metadata
    if prep.metadata:
        with st.expander("ℹ️ Image Metadata", expanded=False):
            for k, v in prep.metadata.items():
                st.text(f"{k}: {v}")

    # Preprocessing warnings
    if prep.warnings:
        for w in prep.warnings:
            st.warning(f"⚠️ Preprocessing: {w}")


def render_risk_markers_table(result: Any) -> None:
    """Render a colour-coded risk marker table from the analysis result.

    Displays all six OA risk markers with their detection status,
    confidence, and explanatory notes using emoji and colour cues.

    Args:
        result: An ``OAAnalysisResult`` instance.

    Returns:
        None. Side-effects: renders Streamlit elements.
    """
    st.subheader("🩻 Risk Markers")
    st.caption(
        "Model-derived markers — NOT radiologist-confirmed findings. "
        "For research and decision-support only."
    )

    try:
        rm_dict = result.risk_markers.to_dict()
    except Exception as exc:
        st.error(f"Could not load risk markers: {exc}")
        return

    # Build display rows
    rows_html = ""
    for key, label in _MARKER_LABELS.items():
        marker = rm_dict.get(key, {})
        status = marker.get("status", "uncertain") if isinstance(marker, dict) else "uncertain"
        confidence = marker.get("confidence", 0.0) if isinstance(marker, dict) else 0.0
        note = marker.get("note", "") if isinstance(marker, dict) else ""

        emoji = _STATUS_EMOJI.get(status, "⚪")
        color = _STATUS_COLOR.get(status, "#808080")
        conf_pct = f"{float(confidence) * 100:.1f}%"
        status_display = status.replace("_", " ").title()

        rows_html += (
            f"<tr>"
            f"<td style='padding:9px 12px;border-bottom:1px solid #eee'><strong>{label}</strong></td>"
            f"<td style='padding:9px 12px;border-bottom:1px solid #eee'>"
            f"  {emoji} <span style='color:{color};font-weight:700'>{status_display}</span>"
            f"</td>"
            f"<td style='padding:9px 12px;border-bottom:1px solid #eee'>{conf_pct}</td>"
            f"<td style='padding:9px 12px;border-bottom:1px solid #eee;font-size:.88em;color:#555'>{note}</td>"
            f"</tr>"
        )

    table_html = f"""
    <table style='width:100%;border-collapse:collapse;margin-top:8px'>
      <thead>
        <tr style='background:#2c3e50;color:#fff'>
          <th style='padding:10px 12px;text-align:left'>Marker</th>
          <th style='padding:10px 12px;text-align:left'>Status</th>
          <th style='padding:10px 12px;text-align:left'>Confidence</th>
          <th style='padding:10px 12px;text-align:left'>Note</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    <p style='font-size:.8em;color:#888;margin-top:6px'>
      🟠 Detected &nbsp; 🟡 Possible &nbsp; ⚪ Uncertain &nbsp; 🟢 Not Detected
    </p>
    """
    st.markdown(table_html, unsafe_allow_html=True)


def render_model_score(result: Any) -> None:
    """Render OA score, confidence badge, and KL grade (unavailable) section.

    Displays the model's OA-related score as a progress bar, labels the
    confidence level with a coloured badge, and clearly marks KL grade
    as unavailable pending task-specific fine-tuning.

    Args:
        result: An ``OAAnalysisResult`` instance.

    Returns:
        None. Side-effects: renders Streamlit elements.
    """
    st.subheader("📊 Model Output")

    pred = result.prediction
    oa_score = float(getattr(pred, "oa_score", 0.0))
    confidence = getattr(pred, "confidence_level", "low")

    # Score progress bar
    st.markdown("**OA-Related Score**")
    st.progress(oa_score, text=f"{oa_score:.3f} ({oa_score * 100:.1f}%)")

    # Confidence badge
    badge_colors = {
        "high": ("#721c24", "#f8d7da"),
        "medium": ("#856404", "#fff3cd"),
        "low": ("#155724", "#d4edda"),
    }
    fg, bg = badge_colors.get(confidence.lower(), ("#555", "#eee"))
    st.markdown(
        f"<span style='background:{bg};color:{fg};padding:5px 14px;"
        f"border-radius:14px;font-weight:700;font-size:.95em'>"
        f"{confidence.upper()} CONFIDENCE</span>",
        unsafe_allow_html=True,
    )

    if getattr(pred, "requires_finetuning", True):
        st.warning(
            "⚠️ **Requires fine-tuning**: This score is based on ImageNet "
            "pretrained weights only. Fine-tune on OA-labelled data before "
            "any research application."
        )

    # KL Grade section
    st.markdown("---")
    st.markdown("**Kellgren–Lawrence (KL) Grade**")
    st.markdown(
        """
        <div style='background:#f8f9fa;border-left:4px solid #bbb;
                    padding:12px 16px;border-radius:4px;color:#777;font-style:italic'>
        ⛔ Not available — KL grade prediction requires dedicated fine-tuning on
        labelled OA datasets (e.g., OAI, MOST). This feature is <strong>not
        supported</strong> in the current ImageNet-weights-only prototype.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_explainability(result: Any) -> None:
    """Render the Grad-CAM / explainability heat-map with labelling.

    Displays the overlaid heat-map image alongside a clear explanation
    of what the visualisation represents. Includes a warning that the
    heat-map is an algorithmic explanation and not a clinical finding.

    Args:
        result: An ``OAAnalysisResult`` instance.

    Returns:
        None. Side-effects: renders Streamlit elements.
    """
    st.subheader("🔥 Explainability — Grad-CAM")

    exp = result.explainability
    method = getattr(exp, "method", "gradcam").upper()

    st.caption(
        f"Method: **{method}** | Target layer: `{getattr(exp, 'target_layer', 'unknown')}`"
    )
    st.info(
        "ℹ️ The heat-map highlights regions that most strongly influenced the "
        "model prediction. **This is an algorithmic explanation only** — it "
        "does not represent anatomical ground truth or confirmed pathological findings."
    )

    if getattr(exp, "warning", ""):
        st.warning(f"⚠️ {exp.warning}")

    overlay = getattr(exp, "overlay", None)
    heatmap = getattr(exp, "heatmap", None)

    col_heat, col_over = st.columns(2)

    with col_heat:
        st.markdown("**Heat-map**")
        if heatmap is not None:
            try:
                hm_bytes = _ndarray_to_png_bytes(heatmap)
                st.image(hm_bytes, caption=f"{method} heat-map", use_container_width=True)
            except Exception as exc:
                logger.warning("Could not display heatmap: %s", exc)
                st.warning("Heat-map display unavailable.")
        else:
            st.info("Heat-map not available.")

    with col_over:
        st.markdown("**Overlay on Image**")
        if overlay is not None:
            try:
                ov_bytes = _ndarray_to_png_bytes(overlay)
                st.image(ov_bytes, caption="Overlay (heat-map on preprocessed)", use_container_width=True)
            except Exception as exc:
                logger.warning("Could not display overlay: %s", exc)
                st.warning("Overlay display unavailable.")
        else:
            st.info("Overlay not available.")


def render_download_buttons(result: Any, generator: "ReportGenerator") -> None:
    """Render JSON and HTML report download buttons.

    Generates both report formats on-demand and provides them as
    Streamlit ``download_button`` elements so users can save reports
    locally without touching the server filesystem.

    Args:
        result:    An ``OAAnalysisResult`` instance.
        generator: A ``ReportGenerator`` instance for creating reports.

    Returns:
        None. Side-effects: renders Streamlit elements.
    """
    st.subheader("📥 Download Reports")
    st.caption(
        "Download analysis reports for records or further review by a clinician."
    )

    col_json, col_html = st.columns(2)

    # ── JSON Report ─────────────────────────────────────────────────
    with col_json:
        try:
            json_data = generator.generate_json_report(result)
            json_bytes = json.dumps(json_data, indent=2, ensure_ascii=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download JSON Report",
                data=json_bytes,
                file_name="oa_analysis_report.json",
                mime="application/json",
                use_container_width=True,
            )
        except Exception as exc:
            logger.error("JSON report generation failed: %s", exc)
            st.error(f"JSON report error: {exc}")

    # ── HTML Report ─────────────────────────────────────────────────
    with col_html:
        try:
            html_data = generator.generate_html_report(result)
            html_bytes = html_data.encode("utf-8")
            st.download_button(
                label="⬇️ Download HTML Report",
                data=html_bytes,
                file_name="oa_analysis_report.html",
                mime="text/html",
                use_container_width=True,
            )
        except Exception as exc:
            logger.error("HTML report generation failed: %s", exc)
            st.error(f"HTML report error: {exc}")
