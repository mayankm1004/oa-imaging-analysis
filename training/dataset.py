"""
training/dataset.py
====================
PyTorch Dataset implementation for OA detection training.

Supports the standard directory layout::

    dataset/
        train/
            normal/
                patient01_left.png
                ...
            oa_related/
                patient42_right.png
                ...
        validation/
            normal/   ...
            oa_related/  ...
        test/
            normal/   ...
            oa_related/  ...

Patient-level integrity
-----------------------
To prevent data leakage, images from the same patient must not span
multiple splits.  Patient IDs are extracted from the filename prefix
(before the first underscore).  Use :func:`patient_aware_split` when
you need to create splits programmatically.

Reproducibility
---------------
All random operations accept a ``seed`` parameter so results are
deterministic.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# Class name → integer label mapping
CLASS_TO_IDX: Dict[str, int] = {"normal": 0, "oa_related": 1}
IDX_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_TO_IDX.items()}
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Patient-aware split helper
# ---------------------------------------------------------------------------


def extract_patient_id(filename: str) -> str:
    """Extract patient ID from a filename.

    Assumes the format ``{patient_id}_{...}.ext`` or ``{patient_id}.ext``.
    Returns the part before the first underscore as the patient identifier.
    If no underscore exists, the whole stem is used.

    Parameters
    ----------
    filename : str
        Basename of the image file.

    Returns
    -------
    str
        Patient ID string.
    """
    stem = Path(filename).stem
    return stem.split("_")[0]


def patient_aware_split(
    samples: List[Tuple[str, int]],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Split samples into train/val/test without patient-level leakage.

    Parameters
    ----------
    samples : List[Tuple[str, int]]
        List of ``(image_path, label)`` tuples.
    val_frac : float
        Fraction of patients for validation.
    test_frac : float
        Fraction of patients for test.
    seed : int
        Random seed.

    Returns
    -------
    Tuple of (train_samples, val_samples, test_samples).
    """
    rng = np.random.default_rng(seed)

    # Group sample indices by patient ID
    patient_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, (path, _) in enumerate(samples):
        pid = extract_patient_id(Path(path).name)
        patient_to_indices[pid].append(idx)

    patient_ids = np.array(sorted(patient_to_indices.keys()))
    rng.shuffle(patient_ids)

    n = len(patient_ids)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))

    test_ids = set(patient_ids[:n_test])
    val_ids = set(patient_ids[n_test: n_test + n_val])

    train_samples, val_samples, test_samples = [], [], []
    for pid, indices in patient_to_indices.items():
        for i in indices:
            if pid in test_ids:
                test_samples.append(samples[i])
            elif pid in val_ids:
                val_samples.append(samples[i])
            else:
                train_samples.append(samples[i])

    logger.info(
        "Patient-aware split: train=%d, val=%d, test=%d samples "
        "(%d/%d/%d unique patients).",
        len(train_samples), len(val_samples), len(test_samples),
        n - n_test - n_val, n_val, n_test,
    )
    return train_samples, val_samples, test_samples


# ---------------------------------------------------------------------------
# Albumentations transforms
# ---------------------------------------------------------------------------


def get_augmentation_transforms(config: Dict[str, Any], split: str) -> A.Compose:
    """Build albumentations transform pipeline for a given split.

    Parameters
    ----------
    config : dict
        Full config dict (loaded from ``oa_model.yaml``).
    split : str
        One of ``'train'``, ``'validation'``, ``'test'``.

    Returns
    -------
    A.Compose
        Albumentations transform pipeline.
    """
    aug_cfg = config.get("augmentation", {})
    prep_cfg = config.get("preprocessing", {})
    h, w = prep_cfg.get("target_size", [384, 384])

    if split == "train":
        transforms = [
            A.Resize(h, w),
            A.HorizontalFlip(p=0.5 if aug_cfg.get("horizontal_flip", True) else 0.0),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=aug_cfg.get("rotation_limit", 10),
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=aug_cfg.get("brightness_limit", 0.2),
                contrast_limit=aug_cfg.get("contrast_limit", 0.2),
                p=0.5,
            ),
        ]
        if aug_cfg.get("elastic_transform", True):
            transforms.append(
                A.ElasticTransform(alpha=1.0, sigma=50, p=0.3)
            )
        transforms += [
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    else:
        transforms = [
            A.Resize(h, w),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]

    return A.Compose(transforms)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class OADataset(Dataset):
    """PyTorch Dataset for OA-related abnormality classification.

    Parameters
    ----------
    root_dir : str | Path
        Root directory containing ``train/``, ``validation/``, ``test/``
        subdirectories.
    split : str
        Which split to load: ``'train'``, ``'validation'``, or ``'test'``.
    config : dict
        Full config dictionary loaded from ``oa_model.yaml``.
    transform : A.Compose, optional
        Albumentations transform pipeline.  If ``None``, a default
        pipeline is built from ``config``.

    Attributes
    ----------
    samples : List[Tuple[str, int]]
        List of ``(image_path, label)`` tuples.
    class_weights : torch.Tensor
        Per-class weights for :class:`torch.nn.CrossEntropyLoss`, computed
        from class frequencies in this split.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str,
        config: Dict[str, Any],
        transform: Optional[A.Compose] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.config = config
        self.transform = transform or get_augmentation_transforms(config, split)

        self.samples = self._load_samples()
        if not self.samples:
            raise FileNotFoundError(
                f"No samples found in '{self.root_dir / split}'. "
                "Check directory structure: root/split/class/*.png"
            )
        logger.info("OADataset [%s]: %d samples loaded.", split, len(self.samples))

    # ------------------------------------------------------------------

    def _load_samples(self) -> List[Tuple[str, int]]:
        """Scan class subdirectories and collect (path, label) pairs."""
        split_dir = self.root_dir / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        samples: List[Tuple[str, int]] = []
        for class_name, label in CLASS_TO_IDX.items():
            class_dir = split_dir / class_name
            if not class_dir.exists():
                logger.warning(
                    "Class directory not found, skipping: %s", class_dir
                )
                continue
            for path in sorted(class_dir.iterdir()):
                if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    samples.append((str(path), label))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """Return (image_tensor, label, image_path) for index ``idx``."""
        path, label = self.samples[idx]

        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise OSError(f"Failed to load image: {path}")

        # Replicate to 3 channels for RGB model input
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        augmented = self.transform(image=image_rgb)
        tensor: torch.Tensor = augmented["image"]  # (3, H, W) float32

        return tensor, label, path

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for loss weighting.

        Returns
        -------
        torch.Tensor
            Shape ``(num_classes,)``.  Higher weight for minority class.
        """
        counts = defaultdict(int)
        for _, label in self.samples:
            counts[label] += 1

        total = len(self.samples)
        n_classes = len(CLASS_TO_IDX)
        weights = torch.zeros(n_classes, dtype=torch.float32)
        for label, count in counts.items():
            weights[label] = total / (n_classes * count) if count > 0 else 1.0

        logger.info("Class weights: %s", dict(zip(IDX_TO_CLASS.values(), weights.tolist())))
        return weights


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def create_data_loaders(
    root_dir: str | Path,
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation, and test DataLoaders.

    Parameters
    ----------
    root_dir : str | Path
        Dataset root (contains train/, validation/, test/).
    config : dict
        Full config dict.

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        ``(train_loader, val_loader, test_loader)``.
    """
    train_cfg = config.get("training", {})
    batch_size: int = train_cfg.get("batch_size", 16)
    num_workers: int = train_cfg.get("num_workers", 4)

    train_ds = OADataset(root_dir, "train", config)
    val_ds = OADataset(root_dir, "validation", config)
    test_ds = OADataset(root_dir, "test", config)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
