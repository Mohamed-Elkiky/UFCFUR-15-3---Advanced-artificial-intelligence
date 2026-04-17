"""Image preprocessing transforms for the CV quality task.

Provides :func:`get_transforms` which builds a ``torchvision`` transform
pipeline for the ``train``, ``val`` or ``test`` split, driven by values
loaded from the repository-level ``config.yaml``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import yaml
from torchvision import transforms

_TASK_KEY = "task2_3_4_cv_quality"
_VALID_SPLITS = {"train", "val", "test"}


def _find_config_path() -> Path:
    """Locate ``config.yaml`` at the repository root.

    The file lives two directories above this module
    (``<repo>/task2_3_4_cv_quality/src/preprocess.py``).
    """
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "config.yaml"


def load_config(config_path: Path | str | None = None) -> Dict[str, Any]:
    """Load the task-specific section from ``config.yaml``.

    Parameters
    ----------
    config_path:
        Optional override for the YAML file location. Defaults to the
        repository-level ``config.yaml``.

    Returns
    -------
    dict
        The mapping stored under the ``task2_3_4_cv_quality`` key.
    """
    path = Path(config_path) if config_path is not None else _find_config_path()
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    if _TASK_KEY not in cfg:
        raise KeyError(
            f"Missing '{_TASK_KEY}' section in config file: {path}"
        )
    return cfg[_TASK_KEY]


def get_transforms(split: str) -> transforms.Compose:
    """Build a ``torchvision`` transform pipeline for the given split.

    Parameters
    ----------
    split:
        One of ``"train"``, ``"val"`` or ``"test"``.

    Returns
    -------
    torchvision.transforms.Compose
        ``train`` returns an augmentation pipeline; ``val`` and ``test``
        return a deterministic resize + normalise pipeline.

    Raises
    ------
    ValueError
        If ``split`` is not one of the supported values.
    """
    if split not in _VALID_SPLITS:
        raise ValueError(
            f"Invalid split '{split}'. Expected one of {sorted(_VALID_SPLITS)}."
        )

    cfg = load_config()
    image_size = int(cfg["image_size"])
    mean = list(cfg["mean"])
    std = list(cfg["std"])
    size = (image_size, image_size)

    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize(size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.02,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


@dataclass
class ClassBalanceReport:
    """Structured summary returned by :func:`check_class_balance`."""

    class_counts: Dict[str, int]
    largest_class_count: int
    flagged_classes: List[str]
    threshold: float


def _counts_from_iterable(
    targets: Iterable[int], classes: List[str]
) -> Dict[str, int]:
    """Convert an iterable of integer labels into a ``name -> count`` dict."""
    counter = Counter(int(t) for t in targets)
    return {name: int(counter.get(idx, 0)) for idx, name in enumerate(classes)}


def check_class_balance(
    class_counts: Mapping[str, int] | Iterable[int],
    threshold: Optional[float] = None,
    classes: Optional[List[str]] = None,
    verbose: bool = True,
) -> ClassBalanceReport:
    """Report per-class counts and flag under-represented classes.

    Parameters
    ----------
    class_counts:
        Either a mapping of ``class_name -> count`` or an iterable of
        integer labels (in which case ``classes`` must be provided so
        indices can be mapped back to names).
    threshold:
        Fraction of the largest class count below which a class is
        flagged as imbalanced. If ``None``, ``imbalance_threshold`` is
        read from ``config.yaml``.
    classes:
        Ordered list of class names. Required when ``class_counts`` is
        an iterable of integer labels; ignored otherwise.
    verbose:
        If ``True`` (default) print the counts and flagged classes.

    Returns
    -------
    ClassBalanceReport
        Structured result with ``class_counts``, ``largest_class_count``,
        ``flagged_classes`` and the effective ``threshold``.
    """
    if isinstance(class_counts, Mapping):
        counts: Dict[str, int] = {str(k): int(v) for k, v in class_counts.items()}
    else:
        if classes is None:
            raise ValueError(
                "`classes` must be provided when `class_counts` is not a mapping."
            )
        counts = _counts_from_iterable(class_counts, classes)

    if not counts:
        raise ValueError("`class_counts` is empty; nothing to check.")

    if threshold is None:
        threshold = float(load_config()["imbalance_threshold"])

    largest = max(counts.values())
    cutoff = threshold * largest
    flagged = [name for name, n in counts.items() if n < cutoff]

    if verbose:
        name_width = max(len(name) for name in counts)
        print(f"Class balance (threshold = {threshold:.2f} of largest class):")
        for name, n in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
            ratio = n / largest if largest else 0.0
            marker = "  FLAGGED" if name in flagged else ""
            print(f"  {name:<{name_width}}  {n:6d}  ({ratio:6.1%}){marker}")
        print(f"Largest class count: {largest}")
        if flagged:
            print(f"Flagged classes ({len(flagged)}): {', '.join(flagged)}")
        else:
            print("No classes flagged below threshold.")

    return ClassBalanceReport(
        class_counts=counts,
        largest_class_count=largest,
        flagged_classes=flagged,
        threshold=threshold,
    )
