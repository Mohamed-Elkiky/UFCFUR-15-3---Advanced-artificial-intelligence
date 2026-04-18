"""
Quality grading for Task 2-3-4 Computer Vision pipeline.

Maps a produce item's Color / Size / Ripeness scores to an overall letter
grade (A, B or C) using the thresholds defined in the case study.

The thresholds are externalised into ``config.yaml`` so that operators can
adjust grading policy without code changes, and so the same cutoffs are
used consistently across training, inference, and reporting.

Grade rules (inclusive):
    Grade A: color >= 75, size >= 80, ripeness >= 70
    Grade B: color >= 65, size >= 70, ripeness >= 60  (any Grade A miss)
    Grade C: anything below the Grade B thresholds
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


REQUIRED_SCORE_KEYS = ("color", "size", "ripeness")
REQUIRED_GRADE_KEYS = ("grade_a", "grade_b")


def _default_config_path() -> Path:
    """
    Return the default ``config.yaml`` location (repository root).

    The grading module lives at
    ``task2_3_4_cv_quality/src/grading.py``, so the project root is
    two levels up.
    """
    return Path(__file__).resolve().parents[2] / "config.yaml"


def _validate_threshold_section(thresholds: Dict[str, Dict[str, float]]) -> None:
    """Raise a ValueError if the grading_thresholds section is malformed."""
    for grade_key in REQUIRED_GRADE_KEYS:
        if grade_key not in thresholds:
            raise ValueError(
                f"config.yaml 'grading_thresholds' section is missing "
                f"required key: {grade_key!r}"
            )
        grade_config = thresholds[grade_key]
        if not isinstance(grade_config, dict):
            raise ValueError(
                f"'grading_thresholds.{grade_key}' must be a mapping of "
                f"score -> threshold value"
            )
        for score_key in REQUIRED_SCORE_KEYS:
            if score_key not in grade_config:
                raise ValueError(
                    f"'grading_thresholds.{grade_key}' is missing "
                    f"required score: {score_key!r}"
                )


def load_thresholds(
    config_path: str | Path | None = None,
) -> Dict[str, Dict[str, float]]:
    """
    Load grading thresholds from ``config.yaml``.

    Parameters
    ----------
    config_path : str | Path, optional
        Path to the YAML config file. Defaults to the project-root
        ``config.yaml`` when not supplied.

    Returns
    -------
    Dict[str, Dict[str, float]]
        A dict of the form
        ``{"grade_a": {"color": ..., "size": ..., "ripeness": ...},
        "grade_b": {...}}``.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If the file is empty, does not contain a ``grading_thresholds``
        section, or the section is malformed.
    """
    if config_path is None:
        config_path = _default_config_path()
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config: Any = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config file is empty or malformed: {config_path}")

    if "grading_thresholds" not in config:
        raise ValueError(
            f"Config file {config_path} is missing the "
            f"'grading_thresholds' section"
        )

    thresholds = config["grading_thresholds"]
    _validate_threshold_section(thresholds)

    return thresholds


def _validate_scores(scores: Dict[str, float]) -> None:
    """Raise a ValueError if the scores dict is missing keys or out of range."""
    if not isinstance(scores, dict):
        raise ValueError(
            f"scores must be a dict with keys {REQUIRED_SCORE_KEYS}, "
            f"got {type(scores).__name__}"
        )
    for key in REQUIRED_SCORE_KEYS:
        if key not in scores:
            raise ValueError(f"scores is missing required key: {key!r}")
        value = scores[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(
                f"scores[{key!r}] must be numeric, got {type(value).__name__}"
            )
        if value < 0 or value > 100:
            raise ValueError(
                f"scores[{key!r}] must be between 0 and 100, got {value}"
            )


def _meets_thresholds(
    scores: Dict[str, float],
    thresholds: Dict[str, float],
) -> bool:
    """Return True if every score is >= its matching threshold."""
    return all(scores[key] >= thresholds[key] for key in REQUIRED_SCORE_KEYS)


def assign_grade(
    scores: Dict[str, float],
    thresholds: Dict[str, Dict[str, float]] | None = None,
) -> str:
    """
    Assign an overall quality grade based on Color / Size / Ripeness scores.

    Grade A is checked first: the item must meet every Grade A threshold.
    If any score falls below its Grade A threshold, Grade B is tried using
    the lower thresholds. If that also fails, the item is assigned Grade C.

    Parameters
    ----------
    scores : Dict[str, float]
        Dict with keys ``color``, ``size`` and ``ripeness``, each a
        percentage in the range 0-100.
    thresholds : Dict[str, Dict[str, float]], optional
        Pre-loaded thresholds in the shape returned by
        :func:`load_thresholds`. If omitted, thresholds are loaded from
        the default config file.

    Returns
    -------
    str
        ``"A"``, ``"B"`` or ``"C"``.

    Raises
    ------
    ValueError
        If ``scores`` is missing a required key, contains a non-numeric
        value, or contains a value outside the 0-100 range.

    Examples
    --------
    >>> assign_grade({"color": 85, "size": 90, "ripeness": 80})
    'A'
    >>> assign_grade({"color": 70, "size": 75, "ripeness": 65})
    'B'
    >>> assign_grade({"color": 50, "size": 40, "ripeness": 30})
    'C'
    """
    _validate_scores(scores)

    if thresholds is None:
        thresholds = load_thresholds()

    if _meets_thresholds(scores, thresholds["grade_a"]):
        return "A"
    if _meets_thresholds(scores, thresholds["grade_b"]):
        return "B"
    return "C"