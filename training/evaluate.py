"""
training/evaluate.py
=====================
Standalone model evaluation script.

Usage
-----
.. code-block:: bash

    python training/evaluate.py \\
        --config configs/oa_model.yaml \\
        --checkpoint models/best_model.pth \\
        --output-dir reports/

Outputs saved to ``reports/``:
    - confusion_matrix.png
    - roc_curve.png
    - precision_recall_curve.png
    - calibration_curve.png
    - evaluation_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import compute_calibration_metrics, compute_classification_metrics
from evaluation.visualization import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)
from models.classifier import EfficientNetOAClassifier
from training.dataset import IDX_TO_CLASS, create_data_loaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: str = "reports/",
) -> Dict[str, Any]:
    """Run full evaluation and save plots and metrics.

    Parameters
    ----------
    model : nn.Module
        Loaded and eval-mode model.
    data_loader : DataLoader
        DataLoader for the test split.
    device : torch.device
        Evaluation device.
    output_dir : str
        Directory to save plots and the JSON report.

    Returns
    -------
    Dict[str, Any]
        Full evaluation metrics dictionary.
    """
    model.eval()
    all_labels: List[int] = []
    all_preds: List[int] = []
    all_probs: List[float] = []

    for batch_tensors, labels, _ in tqdm(data_loader, desc="Evaluating"):
        images = batch_tensors.to(device, non_blocking=True)
        logits = model(images) if hasattr(model, "forward") else model.backbone(images)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Classification metrics ---
    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    calib = compute_calibration_metrics(y_true, y_prob)

    # --- Plots ---
    class_names = [IDX_TO_CLASS.get(i, str(i)) for i in range(2)]
    plot_confusion_matrix(y_true, y_pred, class_names, str(out / "confusion_matrix.png"))
    metrics["roc_auc"] = plot_roc_curve(y_true, y_prob, str(out / "roc_curve.png"))
    metrics["pr_auc"] = plot_pr_curve(y_true, y_prob, str(out / "precision_recall_curve.png"))
    plot_calibration_curve(y_true, y_prob, str(out / "calibration_curve.png"))

    # --- JSON report ---
    report: Dict[str, Any] = {
        "classification_metrics": {k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()},
        "calibration": {"ece": round(calib["ece"], 6)},
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "n_negative": int((y_true == 0).sum()),
        "disclaimer": (
            "These metrics were computed on a held-out test set. "
            "They do not constitute clinical validation. "
            "Prospective clinical evaluation is required before any medical use."
        ),
    }
    report_path = out / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Evaluation report saved to: %s", report_path)
    logger.info(
        "Key metrics: Acc=%.3f | Sens=%.3f | Spec=%.3f | F1=%.3f | AUC=%.3f",
        metrics["accuracy"],
        metrics["recall"],
        metrics["specificity"],
        metrics["f1"],
        metrics["roc_auc"],
    )
    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OA detection model.")
    parser.add_argument("--config", type=str, default="configs/oa_model.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="reports/")
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info("Device: %s", device)

    model = EfficientNetOAClassifier(config=config)
    model.load(checkpoint_path=args.checkpoint)
    model = model.to(device)

    dataset_root = config.get("dataset", {}).get("root", "dataset/")
    _, val_loader, test_loader = create_data_loaders(dataset_root, config)
    loader = test_loader if args.split == "test" else val_loader

    evaluate_model(model, loader, device, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
