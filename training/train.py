"""
training/train.py
==================
OA Detection model training script.

Usage
-----
.. code-block:: bash

    # Train from scratch (ImageNet pretrained backbone)
    python training/train.py --config configs/oa_model.yaml

    # Resume from checkpoint
    python training/train.py --config configs/oa_model.yaml \\
        --resume models/checkpoint_last.pth

    # Dry run (validate config + dataset, no training)
    python training/train.py --config configs/oa_model.yaml --dry-run

Training phases
---------------
Phase 1 (warmup)
    Backbone frozen; only the classification head is trained for
    ``warmup_epochs`` epochs.
Phase 2 (fine-tune)
    Last two backbone blocks unfrozen; full model trained with a lower
    learning rate and cosine annealing.

Outputs
-------
``models/best_model.pth``     – checkpoint with lowest validation loss
``models/checkpoint_last.pth`` – checkpoint after final epoch

Notes
-----
This script uses AMP (Automatic Mixed Precision) when a CUDA device is
available.  All random seeds are set for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from models.classifier import EfficientNetOAClassifier
from training.dataset import create_data_loaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance in binary classification.

    Parameters
    ----------
    gamma : float
        Focusing parameter.  ``gamma=0`` reduces to cross-entropy.
    alpha : Optional[torch.Tensor]
        Per-class weighting tensor of shape ``(num_classes,)``.

    References
    ----------
    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seeds(seed: int) -> None:
    """Set random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Random seed set to %d.", seed)


# ---------------------------------------------------------------------------
# Epoch helpers
# ---------------------------------------------------------------------------


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler,
) -> Dict[str, float]:
    """Run one training epoch.

    Returns
    -------
    dict with keys ``'loss'`` and ``'accuracy'``.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_tensors, labels, _ in tqdm(loader, desc="Train", leave=False):
        images = batch_tensors.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=device.type == "cuda"):
            logits = model.backbone(images)  # type: ignore[attr-defined]
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return {"loss": total_loss / total, "accuracy": correct / total}


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Run one validation epoch.

    Returns
    -------
    dict with keys ``'loss'`` and ``'accuracy'``.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_tensors, labels, _ in tqdm(loader, desc="Val  ", leave=False):
        images = batch_tensors.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=device.type == "cuda"):
            logits = model.backbone(images)  # type: ignore[attr-defined]
            loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return {"loss": total_loss / total, "accuracy": correct / total}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_checkpoint(
    state: Dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    logger.info("Checkpoint saved → %s", path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> int:
    """Load checkpoint and return the starting epoch."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    start_epoch: int = ckpt.get("epoch", 0) + 1
    logger.info("Resumed from %s (epoch %d).", path, ckpt.get("epoch", 0))
    return start_epoch


# ---------------------------------------------------------------------------
# Backbone freeze/unfreeze helpers
# ---------------------------------------------------------------------------


def freeze_backbone(model: EfficientNetOAClassifier) -> None:
    """Freeze all backbone parameters except the classification head."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Backbone frozen. Trainable params: %s", f"{trainable:,}")


def unfreeze_last_blocks(
    model: EfficientNetOAClassifier,
    num_blocks: int = 2,
) -> None:
    """Unfreeze the last ``num_blocks`` blocks of the EfficientNet features."""
    # EfficientNet-B4 features has indices 0–8
    features = model.backbone.features  # type: ignore[attr-defined]
    n = len(features)
    for i, block in enumerate(features):
        if i >= n - num_blocks:
            for param in block.parameters():
                param.requires_grad_(True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Unfrozen last %d backbone blocks. Trainable params: %s",
        num_blocks,
        f"{trainable:,}",
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train OA detection model (EfficientNet-B4)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/oa_model.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and dataset without training.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/",
        help="Directory to save checkpoints.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})

    seed: int = train_cfg.get("random_seed", 42)
    set_seeds(seed)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    # Dataset
    dataset_root = dataset_cfg.get("root", "dataset/")
    logger.info("Loading dataset from: %s", dataset_root)
    try:
        train_loader, val_loader, _ = create_data_loaders(dataset_root, config)
    except FileNotFoundError as exc:
        logger.error("Dataset not found: %s", exc)
        logger.info(
            "Create the dataset directory structure:\n"
            "  dataset/train/normal/*.png\n"
            "  dataset/train/oa_related/*.png\n"
            "  dataset/validation/normal/*.png\n"
            "  dataset/validation/oa_related/*.png"
        )
        sys.exit(1)

    if args.dry_run:
        logger.info("Dry-run complete. Config and dataset look valid.")
        logger.info(
            "  Train batches: %d | Val batches: %d", len(train_loader), len(val_loader)
        )
        return

    # Model
    model = EfficientNetOAClassifier(config=config)
    model.load(checkpoint_path=args.resume)
    model = model.to(device)

    # Class weights
    class_weights = train_loader.dataset.get_class_weights().to(device)  # type: ignore

    # Loss
    use_focal: bool = train_cfg.get("use_focal_loss", True)
    if use_focal:
        criterion: nn.Module = FocalLoss(
            gamma=float(train_cfg.get("focal_loss_gamma", 2.0)),
            alpha=class_weights,
        )
        logger.info("Using FocalLoss (gamma=%.1f).", train_cfg.get("focal_loss_gamma", 2.0))
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logger.info("Using CrossEntropyLoss.")

    lr: float = float(train_cfg.get("learning_rate", 1e-4))
    wd: float = float(train_cfg.get("weight_decay", 1e-4))
    n_epochs: int = int(train_cfg.get("epochs", 50))
    warmup_epochs: int = int(train_cfg.get("warmup_epochs", 5))
    patience: int = int(train_cfg.get("early_stopping_patience", 10))
    output_dir = Path(args.output_dir)

    scaler = GradScaler(enabled=(device.type == "cuda"))

    # ----------------------------------------------------------------
    # Phase 1: Train head only (backbone frozen)
    # ----------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Phase 1: Training classification head (%d epochs).", warmup_epochs)
    logger.info("=" * 60)
    freeze_backbone(model)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr * 10,
        weight_decay=wd,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=warmup_epochs, eta_min=lr * 0.1)

    best_val_loss = float("inf")
    no_improve = 0
    history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_acc": []}

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(Path(args.resume), model, optimizer)

    for epoch in range(start_epoch, warmup_epochs):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        logger.info(
            "Epoch %d/%d | train_loss=%.4f acc=%.3f | val_loss=%.4f acc=%.3f | %.1fs",
            epoch + 1, warmup_epochs,
            train_metrics["loss"], train_metrics["accuracy"],
            val_metrics["loss"], val_metrics["accuracy"],
            time.time() - t0,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "optimizer_state": optimizer.state_dict(), "val_loss": best_val_loss},
                output_dir / "best_model.pth",
            )

    # ----------------------------------------------------------------
    # Phase 2: Fine-tune last backbone blocks
    # ----------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Phase 2: Fine-tuning last 2 backbone blocks (%d epochs).", n_epochs - warmup_epochs)
    logger.info("=" * 60)
    unfreeze_last_blocks(model, num_blocks=2)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=wd,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs - warmup_epochs, eta_min=lr * 0.01)
    no_improve = 0

    for epoch in range(warmup_epochs, n_epochs):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        logger.info(
            "Epoch %d/%d | train_loss=%.4f acc=%.3f | val_loss=%.4f acc=%.3f | %.1fs",
            epoch + 1, n_epochs,
            train_metrics["loss"], train_metrics["accuracy"],
            val_metrics["loss"], val_metrics["accuracy"],
            time.time() - t0,
        )

        # Save last checkpoint
        save_checkpoint(
            {"epoch": epoch, "model_state": model.state_dict(),
             "optimizer_state": optimizer.state_dict(), "val_loss": val_metrics["loss"]},
            output_dir / "checkpoint_last.pth",
        )

        # Early stopping
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            no_improve = 0
            save_checkpoint(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "optimizer_state": optimizer.state_dict(), "val_loss": best_val_loss},
                output_dir / "best_model.pth",
            )
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping triggered after %d epochs without improvement.", patience)
                break

    # Save training history
    history_path = output_dir / "training_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    logger.info("Training complete. Best val_loss: %.4f", best_val_loss)
    logger.info("Best model saved to: %s", output_dir / "best_model.pth")
    logger.info(
        "\n⚠  IMPORTANT: This model requires clinical validation before use in "
        "any medical decision-making context. Fine-tuning on ImageNet weights is "
        "a starting point only."
    )


if __name__ == "__main__":
    main()
