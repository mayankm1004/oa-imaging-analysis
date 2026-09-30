"""
app/ui/sidebar.py
==================
Streamlit sidebar for OA detection application settings.
"""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st


def render_sidebar() -> Dict[str, Any]:
    """Render the settings sidebar and return configuration selections.

    Returns
    -------
    dict with keys:
        ``model_name``, ``joint_type``, ``explainability_method``,
        ``show_preprocessing_steps``, ``checkpoint_path``.
    """
    with st.sidebar:
        st.title("⚙️ Settings")

        st.subheader("Model")
        model_name = st.selectbox(
            "Model backbone",
            options=["efficientnet_b4_oa"],
            help=(
                "EfficientNet-B4 adapted for OA-related classification. "
                "Uses ImageNet pretrained weights by default. "
                "Task-specific fine-tuning is required before clinical use."
            ),
        )

        checkpoint_path = st.text_input(
            "Fine-tuned checkpoint path",
            value="",
            placeholder="models/best_model.pth (optional)",
            help=(
                "Leave empty to use ImageNet pretrained weights. "
                "Provide a path to a fine-tuned checkpoint for improved accuracy."
            ),
        )
        checkpoint_path = checkpoint_path.strip() or None

        st.subheader("Joint Type")
        joint_type = st.selectbox(
            "Joint",
            options=["Knee", "Hip (future)", "Shoulder (future)", "Hand (future)"],
            index=0,
        )
        joint_type = joint_type.split(" ")[0].lower()

        st.subheader("Explainability")
        explainability_method = st.radio(
            "Grad-CAM method",
            options=["gradcam", "gradcam++"],
            help=(
                "Grad-CAM: gradient-weighted activation mapping. "
                "Grad-CAM++: improved second-order weighting."
            ),
        )

        show_preprocessing = st.checkbox(
            "Show preprocessing steps",
            value=False,
            help="Expand to view intermediate pipeline outputs.",
        )

        st.divider()
        st.subheader("ℹ️ About")
        st.markdown(
            """
            **OA Imaging Analysis**
            Research Decision-Support System

            - **Version**: 0.1.0
            - **Model**: EfficientNet-B4
            - **Framework**: PyTorch + OpenCV

            [GitHub Repository](#)
            """,
            unsafe_allow_html=False,
        )

        st.warning(
            "⚠️ **Not a medical device.**\n\n"
            "This tool is a research prototype. "
            "All outputs require clinical review.",
            icon="🩺",
        )

    return {
        "model_name": model_name,
        "joint_type": joint_type,
        "explainability_method": explainability_method,
        "show_preprocessing_steps": show_preprocessing,
        "checkpoint_path": checkpoint_path,
    }
