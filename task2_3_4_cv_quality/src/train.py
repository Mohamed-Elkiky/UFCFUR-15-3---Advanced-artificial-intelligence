"""Training-side data plumbing for the CV quality task.

This module builds the reproducible train / val / test ``DataLoader``
foundation the rest of the task will consume. It applies the transforms
defined in :mod:`preprocess` and handles class imbalance on the
training split via a :class:`~torch.utils.data.WeightedRandomSampler`.

Typical usage::

    from task2_3_4_cv_quality.src.train import create_dataloaders

    bundle = create_dataloaders()
    train_loader = bundle.train_loader
    val_loader = bundle.val_loader
    test_loader = bundle.test_loader
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from task2_3_4_cv_quality.src.preprocess import (
    ClassBalanceReport,
    check_class_balance,
    get_transforms,
    load_config,
)


@dataclass
class DataLoaderBundle:
    """Container for the objects produced by :func:`create_dataloaders`."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    classes: List[str]
    train_indices: List[int]
    val_indices: List[int]
    train_class_counts: Dict[str, int]
    balance_report: ClassBalanceReport


def _resolve_path(path_str: str) -> Path:
    """Resolve a config path relative to the repo root when not absolute."""
    path = Path(path_str)
    if not path.is_absolute():
        repo_root = Path(__file__).resolve().parents[2]
        path = repo_root / path
    return path


def create_data_splits(
    train_dir: Path,
    val_split: float,
    seed: int,
    train_transform: transforms.Compose,
    val_transform: transforms.Compose,
) -> Tuple[Subset, Subset, List[str], List[int], List[int]]:
    """Build train / val ``Subset`` datasets over a shared index split.

    Two independent :class:`~torchvision.datasets.ImageFolder` instances
    are created over ``train_dir`` so the training and validation
    subsets can apply different transforms despite sharing samples.

    Parameters
    ----------
    train_dir:
        Directory passed to :class:`ImageFolder` (the ``Train`` folder).
    val_split:
        Fraction of samples held out for validation (``0 < val_split < 1``).
    seed:
        Seed used for the reproducible random permutation of indices.
    train_transform, val_transform:
        Transform pipelines applied by the train and validation subsets
        respectively.

    Returns
    -------
    (train_subset, val_subset, classes, train_indices, val_indices)
    """
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"`val_split` must be in (0, 1), got {val_split}.")

    base_dataset = ImageFolder(str(train_dir))
    n_total = len(base_dataset)
    if n_total == 0:
        raise RuntimeError(f"No images found under {train_dir}.")

    generator = torch.Generator().manual_seed(int(seed))
    permuted = torch.randperm(n_total, generator=generator).tolist()
    n_val = int(round(n_total * val_split))
    val_indices = permuted[:n_val]
    train_indices = permuted[n_val:]

    train_dataset = ImageFolder(str(train_dir), transform=train_transform)
    val_dataset = ImageFolder(str(train_dir), transform=val_transform)

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    return train_subset, val_subset, list(base_dataset.classes), train_indices, val_indices


def get_training_class_counts(
    targets: Sequence[int],
    indices: Sequence[int],
    classes: Sequence[str],
) -> Dict[str, int]:
    """Return per-class counts restricted to ``indices``.

    Parameters
    ----------
    targets:
        Full list of integer labels (e.g. ``ImageFolder.targets``).
    indices:
        Indices of the training subset within ``targets``.
    classes:
        Ordered class names so the dict preserves ``ImageFolder`` order.
    """
    counter = Counter(int(targets[i]) for i in indices)
    return {name: int(counter.get(idx, 0)) for idx, name in enumerate(classes)}


def create_weighted_sampler(
    targets: Sequence[int],
    indices: Sequence[int],
    num_classes: int,
) -> WeightedRandomSampler:
    """Build a :class:`WeightedRandomSampler` that oversamples minorities.

    Each training sample receives a weight of ``1 / count[class]`` so
    rare classes are drawn more often. Sampling is performed with
    replacement and for one full training-subset-worth of samples per
    epoch.
    """
    class_counts = [0] * num_classes
    for i in indices:
        class_counts[int(targets[i])] += 1

    class_weights = [
        1.0 / c if c > 0 else 0.0 for c in class_counts
    ]
    sample_weights = torch.as_tensor(
        [class_weights[int(targets[i])] for i in indices],
        dtype=torch.double,
    )
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(indices),
        replacement=True,
    )


def create_dataloaders(
    config: Optional[dict] = None,
    verbose: bool = True,
) -> DataLoaderBundle:
    """Build train / val / test ``DataLoader`` objects for the task.

    The training loader uses a :class:`WeightedRandomSampler` (no
    ``shuffle=True``) to oversample minority classes; the validation
    and test loaders are plain, unshuffled loaders with deterministic
    transforms.
    """
    cfg = config if config is not None else load_config()

    train_dir = _resolve_path(cfg["train_dir"])
    test_dir = _resolve_path(cfg["test_dir"])
    val_split = float(cfg["val_split"])
    batch_size = int(cfg["batch_size"])
    num_workers = int(cfg["num_workers"])
    seed = int(cfg["random_seed"])

    train_tf = get_transforms("train")
    val_tf = get_transforms("val")
    test_tf = get_transforms("test")

    train_subset, val_subset, classes, train_indices, val_indices = create_data_splits(
        train_dir=train_dir,
        val_split=val_split,
        seed=seed,
        train_transform=train_tf,
        val_transform=val_tf,
    )

    base_targets = train_subset.dataset.targets

    train_class_counts = get_training_class_counts(
        targets=base_targets, indices=train_indices, classes=classes
    )
    balance_report = check_class_balance(
        train_class_counts, verbose=verbose
    )

    sampler = create_weighted_sampler(
        targets=base_targets,
        indices=train_indices,
        num_classes=len(classes),
    )

    test_dataset = ImageFolder(str(test_dir), transform=test_tf)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    if verbose:
        print()
        print(f"Classes ({len(classes)}): {classes}")
        print(f"Train samples : {len(train_subset)}")
        print(f"Val   samples : {len(val_subset)}")
        print(f"Test  samples : {len(test_dataset)}")

    return DataLoaderBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        classes=classes,
        train_indices=train_indices,
        val_indices=val_indices,
        train_class_counts=train_class_counts,
        balance_report=balance_report,
    )


if __name__ == "__main__":
    create_dataloaders()