"""
Tests for Task 2-3-4 Computer Vision quality pipeline.

Covers the merged grading module from AA-29 + AA-30:

AA-29 (Mohamed):
    - ``compute_quality_scores``: model output -> Color/Size/Ripeness scores
    - ``get_recommendation``: grade letter -> business recommendation
    - ``grade_produce``: end-to-end pipeline

AA-30 (KD):
    - ``assign_grade``: per-dimension grade thresholds from config
    - ``load_thresholds``: YAML threshold loader
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from task2_3_4_cv_quality.src.grading import (
    assign_grade,
    compute_quality_scores,
    get_recommendation,
    grade_produce,
    load_thresholds,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Thresholds from the case study specification.
CASE_STUDY_THRESHOLDS = {
    "grade_a": {"color": 75, "size": 80, "ripeness": 70},
    "grade_b": {"color": 65, "size": 70, "ripeness": 60},
}


@pytest.fixture
def thresholds() -> dict:
    """Return a copy of the case-study thresholds for each test."""
    return {
        "grade_a": dict(CASE_STUDY_THRESHOLDS["grade_a"]),
        "grade_b": dict(CASE_STUDY_THRESHOLDS["grade_b"]),
    }


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """Write a well-formed grading config.yaml to a temp path."""
    config = {"grading_thresholds": CASE_STUDY_THRESHOLDS}
    path = tmp_path / "config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    return path


# ---------------------------------------------------------------------------
# assign_grade - happy path (AA-30)
# ---------------------------------------------------------------------------

def test_assign_grade_example_from_case_study_is_A(thresholds):
    """Case study example: Color 85, Size 90, Ripeness 80 -> Grade A."""
    result = assign_grade(
        {"color_score": 85, "size_score": 90, "ripeness_score": 80},
        thresholds=thresholds,
    )
    assert result == "A"


def test_assign_grade_typical_B_case(thresholds):
    """Scores just below the A bar but well above B -> Grade B."""
    result = assign_grade(
        {"color_score": 70, "size_score": 75, "ripeness_score": 65},
        thresholds=thresholds,
    )
    assert result == "B"


def test_assign_grade_typical_C_case(thresholds):
    """Scores well below the B bar -> Grade C."""
    result = assign_grade(
        {"color_score": 50, "size_score": 40, "ripeness_score": 30},
        thresholds=thresholds,
    )
    assert result == "C"


# ---------------------------------------------------------------------------
# assign_grade - boundary values (AA-30)
# ---------------------------------------------------------------------------

def test_assign_grade_exact_A_boundary_is_A(thresholds):
    """Exactly meeting every Grade A threshold qualifies for A (inclusive)."""
    result = assign_grade(
        {"color_score": 75, "size_score": 80, "ripeness_score": 70},
        thresholds=thresholds,
    )
    assert result == "A"


def test_assign_grade_one_below_A_drops_to_B(thresholds):
    """A single score below the A bar drops the overall grade to B."""
    result = assign_grade(
        {"color_score": 74, "size_score": 80, "ripeness_score": 70},
        thresholds=thresholds,
    )
    assert result == "B"


def test_assign_grade_exact_B_boundary_is_B(thresholds):
    """Exactly meeting every Grade B threshold qualifies for B."""
    result = assign_grade(
        {"color_score": 65, "size_score": 70, "ripeness_score": 60},
        thresholds=thresholds,
    )
    assert result == "B"


def test_assign_grade_one_below_B_drops_to_C(thresholds):
    """A single score below the B bar drops the overall grade to C."""
    result = assign_grade(
        {"color_score": 64, "size_score": 70, "ripeness_score": 60},
        thresholds=thresholds,
    )
    assert result == "C"


def test_assign_grade_zero_scores_is_C(thresholds):
    """Zero across the board is Grade C (not an error)."""
    result = assign_grade(
        {"color_score": 0, "size_score": 0, "ripeness_score": 0},
        thresholds=thresholds,
    )
    assert result == "C"


def test_assign_grade_perfect_scores_is_A(thresholds):
    """100/100/100 is Grade A."""
    result = assign_grade(
        {"color_score": 100, "size_score": 100, "ripeness_score": 100},
        thresholds=thresholds,
    )
    assert result == "A"


# ---------------------------------------------------------------------------
# assign_grade - invalid input (AA-30)
# ---------------------------------------------------------------------------

def test_assign_grade_rejects_missing_key(thresholds):
    """A missing score key must raise a ValueError."""
    with pytest.raises(ValueError, match="ripeness_score"):
        assign_grade(
            {"color_score": 80, "size_score": 85},
            thresholds=thresholds,
        )


def test_assign_grade_rejects_non_numeric_score(thresholds):
    """A non-numeric score must raise a ValueError."""
    with pytest.raises(ValueError, match="numeric"):
        assign_grade(
            {"color_score": "high", "size_score": 85, "ripeness_score": 70},
            thresholds=thresholds,
        )


def test_assign_grade_rejects_out_of_range_score(thresholds):
    """Scores outside 0-100 must raise a ValueError."""
    with pytest.raises(ValueError, match="between 0 and 100"):
        assign_grade(
            {"color_score": 150, "size_score": 80, "ripeness_score": 70},
            thresholds=thresholds,
        )

    with pytest.raises(ValueError, match="between 0 and 100"):
        assign_grade(
            {"color_score": -1, "size_score": 80, "ripeness_score": 70},
            thresholds=thresholds,
        )


# ---------------------------------------------------------------------------
# load_thresholds (AA-30)
# ---------------------------------------------------------------------------

def test_load_thresholds_reads_well_formed_yaml(config_file: Path):
    """A valid config file must load into the expected shape."""
    thresholds = load_thresholds(config_file)

    assert thresholds["grade_a"]["color"] == 75
    assert thresholds["grade_a"]["size"] == 80
    assert thresholds["grade_a"]["ripeness"] == 70
    assert thresholds["grade_b"]["color"] == 65


def test_load_thresholds_raises_on_missing_file(tmp_path: Path):
    """A non-existent config path must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_thresholds(tmp_path / "does_not_exist.yaml")


def test_load_thresholds_raises_on_missing_section(tmp_path: Path):
    """A config file without grading_thresholds must raise ValueError."""
    path = tmp_path / "config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"some_other_section": {"foo": 1}}, f)

    with pytest.raises(ValueError, match="grading_thresholds"):
        load_thresholds(path)


def test_load_thresholds_raises_on_missing_grade_key(tmp_path: Path):
    """A config file missing grade_b must raise ValueError."""
    path = tmp_path / "config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "grading_thresholds": {
                    "grade_a": {"color": 75, "size": 80, "ripeness": 70}
                }
            },
            f,
        )

    with pytest.raises(ValueError, match="grade_b"):
        load_thresholds(path)


def test_assign_grade_loads_default_config_when_thresholds_absent(
    monkeypatch, config_file: Path
):
    """When thresholds is None, assign_grade falls back to load_thresholds()."""
    import task2_3_4_cv_quality.src.grading as grading_module

    monkeypatch.setattr(
        grading_module, "_default_config_path", lambda: config_file
    )

    result = assign_grade(
        {"color_score": 85, "size_score": 90, "ripeness_score": 80}
    )
    assert result == "A"


# ---------------------------------------------------------------------------
# compute_quality_scores (AA-29)
# ---------------------------------------------------------------------------

def test_compute_quality_scores_healthy_high_confidence():
    """High-confidence healthy produce should produce high scores."""
    result = compute_quality_scores(
        {"predicted_class": "Apple__Healthy", "confidence": 0.95}
    )

    assert result["produce_type"] == "Apple"
    assert result["state"] == "Healthy"
    assert result["color_score"] == 95
    assert result["ripeness_score"] == 95
    # size_score = 70 + 0.95 * 30 = 98.5 -> 98
    assert result["size_score"] == 98


def test_compute_quality_scores_rotten_high_confidence():
    """High-confidence rotten produce should produce low scores."""
    result = compute_quality_scores(
        {"predicted_class": "Banana__Rotten", "confidence": 0.90}
    )

    assert result["produce_type"] == "Banana"
    assert result["state"] == "Rotten"
    # color_score = (1 - 0.90) * 40 = 4
    assert result["color_score"] == 4
    # ripeness_score = (1 - 0.90) * 35 = 3.5 -> 3 (banker's rounding)
    assert result["ripeness_score"] == 3
    # size_score = 50 + (1 - 0.90) * 20 = 52
    assert result["size_score"] == 52


def test_compute_quality_scores_rejects_invalid_class_name():
    """A class name without the '__' separator must raise ValueError."""
    with pytest.raises(ValueError, match="Invalid class name"):
        compute_quality_scores(
            {"predicted_class": "AppleHealthy", "confidence": 0.9}
        )


def test_compute_quality_scores_rejects_out_of_range_confidence():
    """A confidence outside [0, 1] must raise ValueError."""
    with pytest.raises(ValueError, match="Confidence"):
        compute_quality_scores(
            {"predicted_class": "Apple__Healthy", "confidence": 1.5}
        )


def test_compute_quality_scores_rejects_missing_keys():
    """Missing predicted_class or confidence must raise KeyError."""
    with pytest.raises(KeyError, match="predicted_class"):
        compute_quality_scores({"confidence": 0.9})

    with pytest.raises(KeyError, match="confidence"):
        compute_quality_scores({"predicted_class": "Apple__Healthy"})


# ---------------------------------------------------------------------------
# get_recommendation (AA-29)
# ---------------------------------------------------------------------------

def test_get_recommendation_returns_text_for_each_grade():
    """Each valid grade must map to non-empty recommendation text."""
    for grade in ("A", "B", "C"):
        text = get_recommendation(grade)
        assert isinstance(text, str)
        assert len(text) > 0


def test_get_recommendation_normalises_case_and_whitespace():
    """Grade lookup should be case- and whitespace-insensitive."""
    assert get_recommendation("a") == get_recommendation("A")
    assert get_recommendation("  B  ") == get_recommendation("B")


def test_get_recommendation_rejects_unknown_grade():
    """Unknown grades must raise ValueError."""
    with pytest.raises(ValueError, match="Unknown grade"):
        get_recommendation("D")


# ---------------------------------------------------------------------------
# grade_produce (AA-29)
# ---------------------------------------------------------------------------

def test_grade_produce_healthy_apple_is_A(thresholds):
    """A highly-confident healthy apple should come out as Grade A."""
    result = grade_produce(
        {"predicted_class": "Apple__Healthy", "confidence": 0.95},
        thresholds=thresholds,
    )

    assert result["grade"] == "A"
    assert result["state"] == "Healthy"
    assert "Premium" in result["recommendation"]


def test_grade_produce_confident_rotten_is_C(thresholds):
    """A confidently-rotten item should come out as Grade C."""
    result = grade_produce(
        {"predicted_class": "Banana__Rotten", "confidence": 0.85},
        thresholds=thresholds,
    )

    assert result["grade"] == "C"
    assert result["state"] == "Rotten"


def test_grade_produce_includes_all_fields(thresholds):
    """The pipeline output must include scores, grade, and recommendation."""
    result = grade_produce(
        {"predicted_class": "Tomato__Healthy", "confidence": 0.80},
        thresholds=thresholds,
    )

    expected_keys = {
        "predicted_class",
        "produce_type",
        "state",
        "confidence",
        "color_score",
        "size_score",
        "ripeness_score",
        "grade",
        "recommendation",
    }
    assert set(result.keys()) == expected_keys