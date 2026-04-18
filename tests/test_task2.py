"""
Tests for Task 2-3-4 Computer Vision quality pipeline.

Currently covers AA-30: the rule-based grading function
``assign_grade`` and its YAML threshold loader ``load_thresholds``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from task2_3_4_cv_quality.src.grading import assign_grade, load_thresholds


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
# assign_grade - happy path
# ---------------------------------------------------------------------------

def test_assign_grade_example_from_case_study_is_A(thresholds):
    """Case study example: Color 85, Size 90, Ripeness 80 -> Grade A."""
    result = assign_grade(
        {"color": 85, "size": 90, "ripeness": 80},
        thresholds=thresholds,
    )
    assert result == "A"


def test_assign_grade_typical_B_case(thresholds):
    """Scores just below the A bar but well above B -> Grade B."""
    result = assign_grade(
        {"color": 70, "size": 75, "ripeness": 65},
        thresholds=thresholds,
    )
    assert result == "B"


def test_assign_grade_typical_C_case(thresholds):
    """Scores well below the B bar -> Grade C."""
    result = assign_grade(
        {"color": 50, "size": 40, "ripeness": 30},
        thresholds=thresholds,
    )
    assert result == "C"


# ---------------------------------------------------------------------------
# assign_grade - boundary values
# ---------------------------------------------------------------------------

def test_assign_grade_exact_A_boundary_is_A(thresholds):
    """Exactly meeting every Grade A threshold qualifies for A (inclusive)."""
    result = assign_grade(
        {"color": 75, "size": 80, "ripeness": 70},
        thresholds=thresholds,
    )
    assert result == "A"


def test_assign_grade_one_below_A_drops_to_B(thresholds):
    """A single score below the A bar drops the overall grade to B."""
    # Color is one below 75 but everything else still qualifies for A
    result = assign_grade(
        {"color": 74, "size": 80, "ripeness": 70},
        thresholds=thresholds,
    )
    assert result == "B"


def test_assign_grade_exact_B_boundary_is_B(thresholds):
    """Exactly meeting every Grade B threshold qualifies for B."""
    result = assign_grade(
        {"color": 65, "size": 70, "ripeness": 60},
        thresholds=thresholds,
    )
    assert result == "B"


def test_assign_grade_one_below_B_drops_to_C(thresholds):
    """A single score below the B bar drops the overall grade to C."""
    result = assign_grade(
        {"color": 64, "size": 70, "ripeness": 60},
        thresholds=thresholds,
    )
    assert result == "C"


def test_assign_grade_zero_scores_is_C(thresholds):
    """Zero across the board is Grade C (not an error)."""
    result = assign_grade(
        {"color": 0, "size": 0, "ripeness": 0},
        thresholds=thresholds,
    )
    assert result == "C"


def test_assign_grade_perfect_scores_is_A(thresholds):
    """100/100/100 is Grade A."""
    result = assign_grade(
        {"color": 100, "size": 100, "ripeness": 100},
        thresholds=thresholds,
    )
    assert result == "A"


# ---------------------------------------------------------------------------
# assign_grade - invalid input
# ---------------------------------------------------------------------------

def test_assign_grade_rejects_missing_key(thresholds):
    """A missing score key must raise a ValueError."""
    with pytest.raises(ValueError, match="ripeness"):
        assign_grade(
            {"color": 80, "size": 85},
            thresholds=thresholds,
        )


def test_assign_grade_rejects_non_numeric_score(thresholds):
    """A non-numeric score must raise a ValueError."""
    with pytest.raises(ValueError, match="numeric"):
        assign_grade(
            {"color": "high", "size": 85, "ripeness": 70},
            thresholds=thresholds,
        )


def test_assign_grade_rejects_out_of_range_score(thresholds):
    """Scores outside 0-100 must raise a ValueError."""
    with pytest.raises(ValueError, match="between 0 and 100"):
        assign_grade(
            {"color": 150, "size": 80, "ripeness": 70},
            thresholds=thresholds,
        )

    with pytest.raises(ValueError, match="between 0 and 100"):
        assign_grade(
            {"color": -1, "size": 80, "ripeness": 70},
            thresholds=thresholds,
        )


# ---------------------------------------------------------------------------
# load_thresholds
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
            {"grading_thresholds": {"grade_a": {"color": 75, "size": 80, "ripeness": 70}}},
            f,
        )

    with pytest.raises(ValueError, match="grade_b"):
        load_thresholds(path)


def test_assign_grade_loads_default_config_when_thresholds_absent(
    monkeypatch, config_file: Path
):
    """When thresholds is None, assign_grade falls back to load_thresholds()."""
    # Point the module's default config resolver at our temp file
    import task2_3_4_cv_quality.src.grading as grading_module

    monkeypatch.setattr(
        grading_module, "_default_config_path", lambda: config_file
    )

    # Should load thresholds from the temp config file without an explicit arg
    result = assign_grade({"color": 85, "size": 90, "ripeness": 80})
    assert result == "A"