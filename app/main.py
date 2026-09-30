"""
app/main.py
===========
Osteoarthritis Imaging Analysis — Research Decision-Support Tool
================================================================

NOT A MEDICAL DIAGNOSIS TOOL.
All outputs require review by a qualified radiologist/clinician.

Run:
    streamlit run app/main.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration — must be the FIRST Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="OA Imaging Analysis",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": (
            "OA Imaging Analysis — Research Decision-Support Prototype v0.1.0\n\n"
            "NOT a medical diagnosis tool. "
            "All outputs require clinical review by a qualified radiologist."
        ),
    },
)

# Lazy imports to avoid slow startup when deps are not installed
try:
    from app.components.disclaimer import render_disclaimer_banner, render_footer_disclaimer
    from app.ui.sidebar import render_sidebar
    from app.ui.upload import render_upload_section
    from app.ui.results import (
        render_explainability,
        render_model_score,
        render_preprocessing_comparison,
        render_risk_markers_table,
        render_download_buttons,
    )
    from inference.predictor import OAPredictor
    from reports.report_generator import ReportGenerator
    _IMPORTS_OK = True
except ImportError as _e:
    _IMPORTS_OK = False
    _IMPORT_ERROR = str(_e)


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

_CSS = """
<style>
/* Disclaimer banner */
.disclaimer-banner {
    background: #fff3cd;
    border: 2px solid #ffc107;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 16px;
    font-size: 0.9em;
}
/* Status badges */
.badge-detected   { background:#ff7043;color:#fff;border-radius:4px;padding:2px 8px; }
.badge-possible   { background:#ffa726;color:#fff;border-radius:4px;padding:2px 8px; }
.badge-uncertain  { background:#78909c;color:#fff;border-radius:4px;padding:2px 8px; }
.badge-normal     { background:#66bb6a;color:#fff;border-radius:4px;padding:2px 8px; }
/* Footer */
footer { visibility: hidden; }
.footer-disclaimer {
    position: fixed; bottom: 0; left: 0; right: 0;
    background: #f8f9fa; border-top: 1px solid #dee2e6;
    padding: 6px 24px; font-size: 0.78em; color: #6c757d;
    z-index: 999;
}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------


def main() -> None:
    if not _IMPORTS_OK:
        st.error(
            f"⛔ Import error: {_IMPORT_ERROR}\n\n"
            "Run `pip install -r requirements.txt` and restart."
        )
        st.stop()

    # Sidebar
    settings = render_sidebar()

    # Main header
    st.title("🦴 Osteoarthritis Imaging Analysis")
    st.markdown(
        "**Research / Decision-Support Tool** &nbsp;|&nbsp; "
        "Knee X-ray OA Risk Marker Detection"
    )

    # Persistent disclaimer banner
    render_disclaimer_banner()

    # Model warnings if using ImageNet weights
    if not settings.get("checkpoint_path"):
        st.info(
            "ℹ️ **Demo mode**: Using ImageNet pretrained weights.\n\n"
            "The model backbone has **not** been fine-tuned on OA data. "
            "Risk marker scores reflect general image features, not validated OA measurements. "
            "Provide a fine-tuned checkpoint via the sidebar for improved accuracy.",
            icon="🔬",
        )

    st.divider()

    # ---------- File upload ----------
    image_path = render_upload_section()

    if image_path is None:
        st.markdown(
            """
            ### Getting started
            1. Upload a knee X-ray image (JPG, PNG, or DICOM).
            2. Click **Analyze**.
            3. Review OA-related risk markers and Grad-CAM explanation.
            4. Download the structured research report.

            > ⚠️ Always have results reviewed by a qualified radiologist.
            """
        )
        render_footer_disclaimer()
        return

    # ---------- Analysis button ----------
    col_btn, col_spacer = st.columns([1, 4])
    with col_btn:
        run_analysis = st.button("🔍 Analyze", type="primary", use_container_width=True)

    # Run or retrieve cached analysis
    if run_analysis or "oa_result" in st.session_state:
        if run_analysis or st.session_state.get("oa_image_path") != image_path:
            with st.spinner("Running OA analysis pipeline…"):
                try:
                    predictor = OAPredictor(
                        config_path=str(_PROJECT_ROOT / "configs" / "oa_model.yaml"),
                        checkpoint_path=settings.get("checkpoint_path"),
                    )
                    result = predictor.analyze(
                        image_path=image_path,
                        joint_type=settings["joint_type"],
                        run_gradcam=True,
                        gradcam_method=settings["explainability_method"],
                    )
                    st.session_state["oa_result"] = result
                    st.session_state["oa_image_path"] = image_path
                except Exception as exc:
                    st.error(f"❌ Analysis failed: {exc}")
                    logger.exception("Analysis pipeline error")
                    st.stop()

        result = st.session_state["oa_result"]
        generator = ReportGenerator(output_dir=str(_PROJECT_ROOT / "reports"))

        # --- Results ---
        st.divider()
        st.subheader("📊 Analysis Results")

        # Preprocessing comparison
        render_preprocessing_comparison(result, show_steps=settings["show_preprocessing_steps"])

        st.divider()

        # Score + confidence
        col_score, col_markers = st.columns([1, 2])
        with col_score:
            render_model_score(result)
        with col_markers:
            render_risk_markers_table(result)

        st.divider()

        # Explainability
        render_explainability(result)

        st.divider()

        # Downloads
        render_download_buttons(result, generator)

    render_footer_disclaimer()


if __name__ == "__main__":
    main()
