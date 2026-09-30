"""
app/components/disclaimer.py — Medical disclaimer UI components.

Provides persistent disclaimer banners and footer text for the OA
Imaging Analysis Streamlit application. Every screen must display at
least one form of the disclaimer so that users are never in doubt
about the research/non-clinical nature of the tool.
"""

from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------------------------
# Disclaimer text (single source of truth)
# ---------------------------------------------------------------------------

def get_disclaimer_text() -> str:
    """Return the full medical disclaimer text for use in reports and UI.

    Returns:
        Multi-sentence disclaimer string covering regulatory status,
        validation status, and appropriate use restrictions.
    """
    return (
        "This system is a research/educational decision-support prototype. "
        "It has NOT been validated for clinical use, approved by any "
        "regulatory body, or tested for diagnostic accuracy in any "
        "specific population. All model outputs are predictions that "
        "require review and interpretation by a qualified radiologist "
        "or clinician. The presence or absence of model-derived risk "
        "markers does NOT constitute a medical diagnosis of "
        "osteoarthritis or any other condition."
    )


# ---------------------------------------------------------------------------
# Streamlit components
# ---------------------------------------------------------------------------

def render_disclaimer_banner() -> None:
    """Render a persistent, attention-grabbing medical disclaimer banner.

    Uses a Streamlit ``warning`` widget styled with HTML to ensure the
    disclaimer is always visible at the top of each page. This function
    should be called at the start of every Streamlit page render.

    Returns:
        None. Side-effects: renders Streamlit elements.
    """
    st.markdown(
        """
        <div style="
            background: #fff3cd;
            border: 2px solid #ffc107;
            border-radius: 8px;
            padding: 14px 20px;
            margin-bottom: 16px;
            font-weight: 600;
            color: #856404;
            font-size: 1.0em;
        ">
        ⚠️ <strong>Research / Decision-Support Tool — NOT a medical diagnosis.</strong><br/>
        Clinical interpretation by a qualified radiologist is required before
        any action is taken based on these outputs.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer_disclaimer() -> None:
    """Render a footer section containing the full disclaimer text.

    Intended to be placed at the bottom of the Streamlit application page.
    Provides the complete disclaimer in a visually distinct red-bordered
    box, ensuring regulatory and ethical compliance for research tools.

    Returns:
        None. Side-effects: renders Streamlit elements.
    """
    st.markdown("---")
    st.markdown(
        f"""
        <div style="
            background: #fff0f0;
            border: 2px solid #e74c3c;
            border-radius: 8px;
            padding: 16px 20px;
            margin-top: 16px;
        ">
        <h4 style="color:#c0392b;margin-top:0">⚠️ Full Medical &amp; Regulatory Disclaimer</h4>
        <p style="color:#444;font-size:.9em;margin-bottom:8px">
            {get_disclaimer_text()}
        </p>
        <p style="color:#888;font-size:.8em;margin-bottom:0">
            <strong>Research prototype.</strong> Requires clinical validation
            before any diagnostic use. No PII is stored.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
