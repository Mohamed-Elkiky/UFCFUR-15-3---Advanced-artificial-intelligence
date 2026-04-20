"""Quality scoring and grading for produce classification.

This module converts model predictions (fresh/rotten classification) into
actionable quality scores, letter grades, and business recommendations.

Pipeline (AA-29):
    Model Output -> Quality Scores (Color/Size/Ripeness %) -> Grade (A/B/C) -> Recommendation

Grade thresholds (AA-30) are loaded from ``config.yaml`` so operators can
adjust grading policy without code changes, and so the same cutoffs are
used consistently across training, inference, and reporting.

Grade rules (inclusive, per the case study):
    Grade A: color >= 75, size >= 80, ripeness >= 70
    Grade B: color >= 65, size >= 70, ripeness >= 60
    Grade C: anything below the Grade B thresholds

Example:
    >>> model_output = {"predicted_class": "Apple__Healthy", "confidence": 0.95}
    >>> result = grade_produce(model_output)
    >>> print(result['grade'])  # 'A'
    >>> print(result['recommendation'])  # 'Premium quality - sell at full price...'
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

REQUIRED_SCORE_KEYS = ("color_score", "size_score", "ripeness_score")
REQUIRED_GRADE_KEYS = ("grade_a", "grade_b")

# Mapping from our internal score key to the matching threshold key in
# ``config.yaml``. Keeps the two naming conventions isolated: the pipeline
# uses ``*_score`` to match compute_quality_scores' output, while the
# config keeps the short names for readability.
_SCORE_TO_THRESHOLD_KEY = {
    "color_score": "color",
    "size_score": "size",
    "ripeness_score": "ripeness",
}

# Business recommendations by grade (AA-29 addition)
RECOMMENDATIONS: Dict[str, str] = {
    "A": "Premium quality - sell at full price in premium display",
    "B": "Standard quality - sell at regular price",
    "C": "Low quality - heavy discount or remove from sale",
}


# ---------------------------------------------------------------------------
# Threshold loading (AA-30)
# ---------------------------------------------------------------------------

def _default_config_path() -> Path:
    """
    Return the default ``config.yaml`` location (repository root).

    The grading module lives at
    ``task2_3_4_cv_quality/src/grading.py``, so the project root is
    two levels up.
    """
    return Path(__file__).resolve().parents[2] / "config.yaml"


def _validate_threshold_section(thresholds: Mapping[str, Any]) -> None:
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
        for short_key in ("color", "size", "ripeness"):
            if short_key not in grade_config:
                raise ValueError(
                    f"'grading_thresholds.{grade_key}' is missing "
                    f"required score: {short_key!r}"
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


# ---------------------------------------------------------------------------
# Model-output parsing (AA-29)
# ---------------------------------------------------------------------------

def _parse_class_name(predicted_class: str) -> tuple[str, str]:
    """Parse class name into produce type and state (Healthy/Rotten).

    Parameters
    ----------
    predicted_class : str
        Class name in format ``"ProduceType__State"`` (e.g., ``"Apple__Healthy"``)

    Returns
    -------
    tuple[str, str]
        ``(produce_type, state)`` where state is ``"Healthy"`` or ``"Rotten"``.

    Examples
    --------
    >>> _parse_class_name("Apple__Healthy")
    ('Apple', 'Healthy')
    >>> _parse_class_name("Banana__Rotten")
    ('Banana', 'Rotten')
    """
    if "__" not in predicted_class:
        raise ValueError(
            f"Invalid class name format: '{predicted_class}'. "
            "Expected format: 'ProduceType__State' (e.g., 'Apple__Healthy')"
        )

    parts = predicted_class.split("__")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid class name format: '{predicted_class}'. "
            "Expected exactly one '__' separator."
        )

    produce_type, state = parts

    if state not in ["Healthy", "Rotten"]:
        raise ValueError(
            f"Invalid state: '{state}'. Expected 'Healthy' or 'Rotten'."
        )

    return produce_type, state


def compute_quality_scores(model_output: Mapping[str, Any]) -> Dict[str, Any]:
    """Compute quality scores (Color %, Size %, Ripeness %) from model output.

    Scoring Logic:
    --------------
    **Healthy Produce (high confidence = high quality):**
    - Color Score:    confidence * 100
    - Ripeness Score: confidence * 100
    - Size Score:     70 + confidence * 30  (less sensitive to freshness)

    **Rotten Produce (high confidence = low quality):**
    - Color Score:    (1 - confidence) * 40  (severe penalty)
    - Ripeness Score: (1 - confidence) * 35  (severe penalty)
    - Size Score:     50 + (1 - confidence) * 20  (moderate penalty)

    Rationale:
    - Healthy items: Higher confidence -> higher scores
    - Rotten items: Higher confidence (more certain it's rotten) -> lower scores
    - Size is less affected by freshness than color/ripeness

    Parameters
    ----------
    model_output : Mapping[str, Any]
        Dictionary with keys:
        - ``predicted_class``: str (e.g., ``"Apple__Healthy"``)
        - ``confidence``: float in ``[0, 1]``

    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - ``predicted_class``: str
        - ``produce_type``: str
        - ``state``: str (``"Healthy"`` or ``"Rotten"``)
        - ``confidence``: float
        - ``color_score``: int (0-100)
        - ``size_score``: int (0-100)
        - ``ripeness_score``: int (0-100)

    Raises
    ------
    KeyError
        If model_output is missing required keys.
    ValueError
        If confidence is not in ``[0, 1]`` or the class name is invalid.
    """
    if "predicted_class" not in model_output:
        raise KeyError("model_output missing 'predicted_class'")
    if "confidence" not in model_output:
        raise KeyError("model_output missing 'confidence'")

    predicted_class = model_output["predicted_class"]
    confidence = float(model_output["confidence"])

    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"Confidence must be in [0, 1], got {confidence}")

    produce_type, state = _parse_class_name(predicted_class)

    if state == "Healthy":
        color_score = round(confidence * 100)
        ripeness_score = round(confidence * 100)
        size_score = round(70 + confidence * 30)
    else:  # Rotten
        color_score = round((1 - confidence) * 40)
        ripeness_score = round((1 - confidence) * 35)
        size_score = round(50 + (1 - confidence) * 20)

    # Clamp to [0, 100]
    color_score = max(0, min(100, color_score))
    size_score = max(0, min(100, size_score))
    ripeness_score = max(0, min(100, ripeness_score))

    return {
        "predicted_class": predicted_class,
        "produce_type": produce_type,
        "state": state,
        "confidence": round(confidence, 4),
        "color_score": color_score,
        "size_score": size_score,
        "ripeness_score": ripeness_score,
    }


# ---------------------------------------------------------------------------
# Grade assignment (AA-30 logic, AA-29 interface)
# ---------------------------------------------------------------------------

def _validate_scores(scores: Mapping[str, Any]) -> None:
    """Raise a ValueError if the scores mapping is missing keys or out of range."""
    if not isinstance(scores, Mapping):
        raise ValueError(
            f"scores must be a mapping with keys {REQUIRED_SCORE_KEYS}, "
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
    scores: Mapping[str, float],
    grade_thresholds: Mapping[str, float],
) -> bool:
    """Return True when every score meets or exceeds its matching threshold."""
    for score_key, threshold_key in _SCORE_TO_THRESHOLD_KEY.items():
        if scores[score_key] < grade_thresholds[threshold_key]:
            return False
    return True


def _get_weakest_dimension(scores: Mapping[str, Any]) -> tuple[str, int]:
    """Return the weakest quality dimension and its score."""
    weakest_key = min(REQUIRED_SCORE_KEYS, key=lambda key: scores[key])
    weakest_dimension = weakest_key.replace("_score", "")
    weakest_value = int(scores[weakest_key])
    return weakest_dimension, weakest_value


def assign_grade(
    scores: Mapping[str, Any],
    thresholds: Mapping[str, Mapping[str, float]] | None = None,
) -> str:
    """Assign letter grade (A/B/C) based on quality scores.

    Grade A is checked first: the item must meet every Grade A threshold.
    If any score falls below its Grade A threshold, Grade B is tried using
    the lower thresholds. If that also fails, the item is assigned Grade C.

    This matches the case-study specification of per-dimension thresholds
    (A: 75/80/70, B: 65/70/60). Thresholds are loaded from
    ``config.yaml`` so grading policy can change without code changes.

    Parameters
    ----------
    scores : Mapping[str, Any]
        Dict with keys ``color_score``, ``size_score`` and ``ripeness_score``,
        each a percentage in the range 0-100.
    thresholds : Mapping[str, Mapping[str, float]], optional
        Pre-loaded thresholds in the shape returned by
        :func:`load_thresholds`. If omitted, thresholds are loaded from
        the default config file.

    Returns
    -------
    str
        ``"A"``, ``"B"``, or ``"C"``.
    """
    _validate_scores(scores)

    if thresholds is None:
        thresholds = load_thresholds()

    if _meets_thresholds(scores, thresholds["grade_a"]):
        return "A"
    if _meets_thresholds(scores, thresholds["grade_b"]):
        return "B"
    return "C"


# ---------------------------------------------------------------------------
# End-to-end pipeline (AA-29)
# ---------------------------------------------------------------------------

def get_recommendation(
    grade: str,
    scores: Mapping[str, Any] | None = None,
) -> str:
    """Get producer-facing recommendation text for a grade.

    Parameters
    ----------
    grade : str
        ``"A"``, ``"B"`` or ``"C"``.
    scores : Mapping[str, Any], optional
        Mapping containing ``color_score``, ``size_score`` and
        ``ripeness_score``. When provided, Grade B and C messages
        include the weakest score dimension.

    Returns
    -------
    str
        Business recommendation text tailored to the grade.
    """
    grade = grade.strip().upper()

    if grade == "A":
        return "Premium quality - sell at full price in premium display"

    if grade not in {"B", "C"}:
        raise ValueError(
            f"Unknown grade: '{grade}'. Expected 'A', 'B', or 'C'."
        )

    if scores is None:
        if grade == "B":
            return "Standard quality - recommend 20% discount."
        return "Low quality - recommend 35% discount or removal from sale."

    _validate_scores(scores)
    weakest_dimension, weakest_value = _get_weakest_dimension(scores)

    if grade == "B":
        return (
            f"Standard quality - {weakest_dimension} is the weakest area "
            f"({weakest_value}%). Recommend 20% discount."
        )

    return (
        f"Low quality - {weakest_dimension} is the weakest area "
        f"({weakest_value}%). Recommend 35% discount or removal from sale."
    )


def update_inventory_action(
    grade: str,
    producer_id: str,
    product: str,
    quantity: int,
) -> Dict[str, Any]:
    """Return the inventory action to take after a quality prediction.

    Parameters
    ----------
    grade : str
        Produce quality grade: ``"A"``, ``"B"`` or ``"C"``.
    producer_id : str
        Identifier of the producer whose stock is being updated.
    product : str
        Product name.
    quantity : int
        Current stock quantity.

    Returns
    -------
    Dict[str, Any]
        Inventory action payload containing:
        - ``action``: one of ``"list_full_price"``, ``"apply_discount"``,
          ``"remove_listing"``
        - ``discount_pct``: one of ``0``, ``20``, ``35``
        - ``producer_id``
    """
    grade = grade.strip().upper()

    if quantity < 0:
        raise ValueError(f"quantity must be >= 0, got {quantity}")

    if grade == "A":
        action = "list_full_price"
        discount_pct = 0
    elif grade == "B":
        action = "apply_discount"
        discount_pct = 20
    elif grade == "C":
        if quantity <= 10:
            action = "remove_listing"
            discount_pct = 0
        else:
            action = "apply_discount"
            discount_pct = 35
    else:
        raise ValueError(
            f"Unknown grade: '{grade}'. Expected 'A', 'B', or 'C'."
        )

    return {
        "producer_id": producer_id,
        "product": product,
        "quantity": quantity,
        "grade": grade,
        "action": action,
        "discount_pct": discount_pct,
    }


def grade_produce(
    model_output: Mapping[str, Any],
    thresholds: Mapping[str, Mapping[str, float]] | None = None,
    producer_id: str | None = None,
    product: str | None = None,
    quantity: int | None = None,
) -> Dict[str, Any]:
    """End-to-end: Model output -> Quality scores -> Grade -> Recommendation."""
    scores = compute_quality_scores(model_output)
    grade = assign_grade(scores, thresholds=thresholds)
    recommendation = get_recommendation(grade, scores)

    result = {
        **scores,
        "grade": grade,
        "recommendation": recommendation,
    }

    if producer_id is not None and product is not None and quantity is not None:
        result["inventory_action"] = update_inventory_action(
            grade=grade,
            producer_id=producer_id,
            product=product,
            quantity=quantity,
        )

    return result


if __name__ == "__main__":
    print("=" * 80)
    print("QUALITY GRADING DEMO")
    print("=" * 80)

    test_cases = [
        {"predicted_class": "Apple__Healthy", "confidence": 0.95},
        {"predicted_class": "Banana__Healthy", "confidence": 0.75},
        {"predicted_class": "Tomato__Rotten", "confidence": 0.90},
        {"predicted_class": "Orange__Rotten", "confidence": 0.60},
        {"predicted_class": "Strawberry__Healthy", "confidence": 0.50},
    ]

    for i, test_input in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"  Input: {test_input}")

        result = grade_produce(test_input)

        print(f"  Produce: {result['produce_type']} ({result['state']})")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(
            f"  Scores: Color={result['color_score']}, "
            f"Size={result['size_score']}, "
            f"Ripeness={result['ripeness_score']}"
        )
        print(f"  Grade: {result['grade']}")
        print(f"  Recommendation: {result['recommendation']}")

    print("\n" + "=" * 80)
    print("Demo complete.")