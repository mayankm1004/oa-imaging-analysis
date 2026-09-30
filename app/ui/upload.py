"""
app/ui/upload.py — File upload UI section for the OA Streamlit application.

Handles image upload (JPEG, PNG, DICOM), temporary file storage, and
basic file metadata display. Returns a path to the saved temporary file
so it can be passed to the preprocessing/inference pipeline.
"""

from __future__ import annotations

import os
import tempfile
import logging
from pathlib import Path
from typing import Optional

import streamlit as st

logger = logging.getLogger(__name__)

# Accepted MIME types mapped to extensions
_ACCEPTED_TYPES: list[str] = ["jpg", "jpeg", "png", "dcm"]
_MAX_FILE_SIZE_MB: int = 50


def render_upload_section() -> Optional[str]:
    """Render the file upload UI and persist the uploaded file to disk.

    Accepts JPEG, PNG, and DICOM (.dcm) images. The uploaded bytes are
    written to a temporary file (using Python's ``tempfile`` module) and
    the path is returned. The caller is responsible for deleting the file
    when it is no longer needed.

    Returns:
        Absolute path string to the saved temporary file if a file was
        uploaded and saved successfully, or ``None`` if no file has been
        uploaded yet.

    Side-effects:
        Renders Streamlit UI elements. Stores the temp-file path in
        ``st.session_state['uploaded_file_path']``.
    """
    st.subheader("📂 Upload Image")
    st.markdown(
        "Upload a knee X-ray image for OA analysis. "
        "Supported formats: **JPEG, PNG, DICOM (.dcm)**."
    )

    uploaded = st.file_uploader(
        label="Choose an image file",
        type=_ACCEPTED_TYPES,
        accept_multiple_files=False,
        help=(
            f"Accepted formats: {', '.join(f'.{t}' for t in _ACCEPTED_TYPES)}. "
            f"Maximum size: {_MAX_FILE_SIZE_MB} MB."
        ),
    )

    if uploaded is None:
        # Clear any previously stored path from session state
        st.session_state.pop("uploaded_file_path", None)
        st.info("👆 Upload an image to begin analysis.")
        return None

    # ── File metadata display ────────────────────────────────────────
    file_size_kb = len(uploaded.getvalue()) / 1024
    file_size_str = (
        f"{file_size_kb / 1024:.2f} MB"
        if file_size_kb >= 1024
        else f"{file_size_kb:.1f} KB"
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("File Name", uploaded.name)
    col2.metric("File Size", file_size_str)
    col3.metric("Format", Path(uploaded.name).suffix.upper() or "Unknown")

    # ── Size guard ───────────────────────────────────────────────────
    if file_size_kb / 1024 > _MAX_FILE_SIZE_MB:
        st.error(
            f"File size ({file_size_str}) exceeds the {_MAX_FILE_SIZE_MB} MB limit. "
            "Please upload a smaller image."
        )
        return None

    # ── Persist to temporary file ────────────────────────────────────
    suffix = Path(uploaded.name).suffix or ".png"
    try:
        tmp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
            prefix="oa_upload_",
        )
        tmp_file.write(uploaded.getvalue())
        tmp_file.flush()
        tmp_file.close()
        tmp_path = tmp_file.name
    except OSError as exc:
        logger.error("Failed to write uploaded file to disk: %s", exc)
        st.error(f"Could not save uploaded file: {exc}")
        return None

    # Cache in session state to avoid re-uploads on widget re-render
    st.session_state["uploaded_file_path"] = tmp_path
    logger.info("Uploaded file saved to: %s", tmp_path)

    # ── Preview for non-DICOM images ─────────────────────────────────
    if suffix.lower() in {".jpg", ".jpeg", ".png"}:
        with st.expander("🖼️ Uploaded Image Preview", expanded=False):
            st.image(uploaded, caption=uploaded.name, use_container_width=True)

    st.success(f"✅ File ready: **{uploaded.name}**")
    return tmp_path
