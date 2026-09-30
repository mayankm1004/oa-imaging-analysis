"""
evaluation/visualization.py
============================
Plotting utilities for OA detection evaluation results.

All plots are saved as PNG files and use a medical-report-friendly style.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

logger = logging.getLogger(__name__)

_STYLE = {
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
}


def _save(fig: plt.Figure, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Plot saved → %s", path)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Sequence[str],
    output_path: str,
) -> None:
    """Plot and save a confusion matrix.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Integer label arrays.
    classes : sequence of str
        Class labels in label order (e.g. ``['Normal', 'OA-Related']``).
    output_path : str
        File path to save the PNG.
    """
    with plt.rc_context(_STYLE):
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title("Confusion Matrix", fontsize=13, pad=10)
        fig.tight_layout()
        _save(fig, output_path)


# ---------------------------------------------------------------------------
# ROC curve
# ---------------------------------------------------------------------------


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: str,
) -> float:
    """Plot ROC curve and return AUC.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    output_path : str
        PNG output path.

    Returns
    -------
    float
        Area under the ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = float(np.trapz(tpr, fpr))

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr, tpr, lw=2, color="#2196F3", label=f"ROC (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax.set_xlabel("False Positive Rate (1 - Specificity)")
        ax.set_ylabel("True Positive Rate (Sensitivity)")
        ax.set_title("Receiver Operating Characteristic", fontsize=13)
        ax.legend(loc="lower right")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        _save(fig, output_path)

    return auc


# ---------------------------------------------------------------------------
# Precision-Recall curve
# ---------------------------------------------------------------------------


def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: str,
) -> float:
    """Plot Precision-Recall curve and return average precision.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    output_path : str
        PNG output path.

    Returns
    -------
    float
        Average precision score (PR-AUC).
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = float(np.trapz(precision[::-1], recall[::-1]))

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(recall, precision, lw=2, color="#4CAF50", label=f"PR (AP = {ap:.3f})")
        baseline = y_true.mean()
        ax.axhline(baseline, color="gray", linestyle="--", lw=1, label=f"Baseline ({baseline:.2f})")
        ax.set_xlabel("Recall (Sensitivity)")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve", fontsize=13)
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        _save(fig, output_path)

    return ap


# ---------------------------------------------------------------------------
# Calibration curve
# ---------------------------------------------------------------------------


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: str,
) -> None:
    """Plot a calibration (reliability) diagram.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_prob : np.ndarray
        Predicted probabilities.
    output_path : str
        PNG output path.
    """
    from sklearn.calibration import calibration_curve as sk_calib

    frac_pos, mean_pred = sk_calib(y_true, y_prob, n_bins=10, strategy="uniform")

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(mean_pred, frac_pos, "s-", color="#FF5722", label="Model")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title("Calibration Curve", fontsize=13)
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        _save(fig, output_path)


# ---------------------------------------------------------------------------
# Grad-CAM comparison
# ---------------------------------------------------------------------------


def plot_gradcam_comparison(
    original: np.ndarray,
    preprocessed: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    output_path: str,
) -> None:
    """Save a 4-panel comparison: original | preprocessed | heatmap | overlay.

    Parameters
    ----------
    original : np.ndarray
        Original loaded X-ray image (H, W) or (H, W, 3).
    preprocessed : np.ndarray
        After preprocessing (H, W).
    heatmap : np.ndarray
        Grad-CAM heatmap (H, W) float32.
    overlay : np.ndarray
        Blended overlay (H, W, 3) uint8.
    output_path : str
        PNG output path.
    """
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        panels = [
            (original, "Original X-Ray", "gray"),
            (preprocessed, "Preprocessed", "gray"),
            (heatmap, "Grad-CAM Heatmap\n(Model explanation — not anatomical proof)", "jet"),
            (overlay, "Overlay", None),
        ]

        for ax, (img, title, cmap) in zip(axes, panels):
            if cmap is None:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap=cmap)
            ax.set_title(title, fontsize=9, wrap=True)
            ax.axis("off")

        fig.suptitle(
            "⚠ Model heatmaps are explanations of neural network activations,\n"
            "NOT radiological proof of anatomical abnormalities.",
            fontsize=9,
            color="darkred",
            y=1.02,
        )
        fig.tight_layout()
        _save(fig, output_path)
