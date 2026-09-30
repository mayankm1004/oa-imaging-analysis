"""
evaluation/metrics.py
======================
Classification metric computation for OA detection evaluation.

All metrics are computed using scikit-learn conventions.
For medical screening applications, sensitivity (recall) and
specificity are especially important.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """Compute a comprehensive set of binary classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels (0 or 1).
    y_pred : np.ndarray
        Predicted integer labels (0 or 1).
    y_prob : np.ndarray
        Predicted probability for the positive class (OA-related), in [0, 1].

    Returns
    -------
    Dict[str, float]
        Dictionary containing:

        - ``accuracy``
        - ``precision``
        - ``recall`` (sensitivity / TPR)
        - ``specificity`` (TNR)
        - ``f1``
        - ``roc_auc``
        - ``pr_auc``
        - ``false_negative_rate`` (FNR = 1 - recall)
        - ``false_positive_rate`` (FPR = 1 - specificity)

    Notes
    -----
    For medical screening, pay close attention to sensitivity and
    false-negative rate.  A high false-negative rate means missed
    disease cases — clinically more dangerous than false positives.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))  # sensitivity
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Specificity from confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # ROC-AUC and PR-AUC
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = float("nan")
        logger.warning("ROC-AUC could not be computed (single class in y_true?).")

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        pr_auc = float("nan")
        logger.warning("PR-AUC could not be computed.")

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,           # sensitivity
        "specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "false_negative_rate": 1.0 - recall,
        "false_positive_rate": 1.0 - specificity,
        # Raw confusion matrix values for reference
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }

    logger.info("Metrics computed: %s", {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics.items()})
    return metrics


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compute expected calibration error (ECE) and calibration curve data.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels (0 or 1).
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    n_bins : int
        Number of bins for the calibration curve.

    Returns
    -------
    dict with keys:
        - ``'ece'`` : float — Expected Calibration Error
        - ``'fraction_of_positives'`` : list
        - ``'mean_predicted_value'`` : list
        - ``'n_bins'`` : int
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    frac_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    # ECE: weighted average of |confidence - accuracy| per bin
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(len(bin_edges) - 1):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = y_prob[mask].mean()
        bin_acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)

    return {
        "ece": float(ece),
        "fraction_of_positives": frac_pos.tolist(),
        "mean_predicted_value": mean_pred.tolist(),
        "n_bins": n_bins,
    }
