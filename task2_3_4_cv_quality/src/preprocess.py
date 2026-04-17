"""Image preprocessing transforms for the CV quality task.

Provides :func:`get_transforms` which builds a ``torchvision`` transform
pipeline for the ``train``, ``val`` or ``test`` split, driven by values
loaded from the repository-level ``config.yaml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

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