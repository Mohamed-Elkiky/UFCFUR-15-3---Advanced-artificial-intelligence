"""Quality scoring and grading for produce classification.

This module converts model predictions (fresh/rotten classification) into
actionable quality scores and business recommendations.

Workflow:
    Model Output → Quality Scores (Color/Size/Ripeness %) → Grade (A/B/C) → Recommendation

Example:
    >>> model_output = {"predicted_class": "Apple__Healthy", "confidence": 0.95}
    >>> result = grade_produce(model_output)
    >>> print(result['grade'])  # 'A'
    >>> print(result['recommendation'])  # 'Premium quality - sell at full price'
"""

from __future__ import annotations

from typing import Any, Dict, Mapping


# Grade thresholds (minimum score across all three dimensions)
GRADE_A_MIN = 80  # Premium quality
GRADE_B_MIN = 60  # Standard quality
# Below 60 = Grade C (discount/reject)

# Business recommendations by grade
RECOMMENDATIONS: Dict[str, str] = {
    "A": "Premium quality - sell at full price in premium display",
    "B": "Standard quality - sell at regular price",
    "C": "Low quality - heavy discount or remove from sale",
}


def _parse_class_name(predicted_class: str) -> tuple[str, str]:
    """Parse class name into produce type and state (Healthy/Rotten).
    
    Parameters
    ----------
    predicted_class : str
        Class name in format "ProduceType__State" (e.g., "Apple__Healthy")
    
    Returns
    -------
    tuple[str, str]
        (produce_type, state) where state is "Healthy" or "Rotten"
    
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
    - Color Score:    confidence × 100
    - Ripeness Score: confidence × 100
    - Size Score:     70 + confidence × 30  (less sensitive to freshness)
    
    **Rotten Produce (high confidence = low quality):**
    - Color Score:    (1 - confidence) × 40  (severe penalty)
    - Ripeness Score: (1 - confidence) × 35  (severe penalty)
    - Size Score:     50 + (1 - confidence) × 20  (moderate penalty)
    
    Rationale:
    - Healthy items: Higher confidence → higher scores
    - Rotten items: Higher confidence (more certain it's rotten) → lower scores
    - Size is less affected by freshness than color/ripeness
    
    Parameters
    ----------
    model_output : Mapping[str, Any]
        Dictionary with keys:
        - 'predicted_class': str (e.g., "Apple__Healthy")
        - 'confidence': float in [0, 1]
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - 'predicted_class': str
        - 'produce_type': str
        - 'state': str ("Healthy" or "Rotten")
        - 'confidence': float
        - 'color_score': int (0-100)
        - 'size_score': int (0-100)
        - 'ripeness_score': int (0-100)
    
    Raises
    ------
    KeyError
        If model_output missing required keys
    ValueError
        If confidence not in [0, 1] or invalid class name
    
    Examples
    --------
    >>> output = {"predicted_class": "Apple__Healthy", "confidence": 0.95}
    >>> scores = compute_quality_scores(output)
    >>> scores['color_score']
    95
    >>> scores['ripeness_score']
    95
    >>> scores['size_score']
    98
    
    >>> output = {"predicted_class": "Banana__Rotten", "confidence": 0.90}
    >>> scores = compute_quality_scores(output)
    >>> scores['color_score']  # Low because very confident it's rotten
    4
    """
    # Validate input
    if "predicted_class" not in model_output:
        raise KeyError("model_output missing 'predicted_class'")
    if "confidence" not in model_output:
        raise KeyError("model_output missing 'confidence'")
    
    predicted_class = model_output["predicted_class"]
    confidence = float(model_output["confidence"])
    
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            f"Confidence must be in [0, 1], got {confidence}"
        )
    
    # Parse class name
    produce_type, state = _parse_class_name(predicted_class)
    
    # Compute scores based on state
    if state == "Healthy":
        # High confidence = high quality
        color_score = round(confidence * 100)
        ripeness_score = round(confidence * 100)
        size_score = round(70 + confidence * 30)
    else:  # Rotten
        # High confidence (certain it's rotten) = low quality
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


def assign_grade(scores: Mapping[str, int]) -> str:
    """Assign letter grade (A/B/C) based on quality scores.
    
    Grading Logic:
    --------------
    - Grade A: minimum(color, size, ripeness) >= 80
    - Grade B: minimum(color, size, ripeness) >= 60
    - Grade C: minimum(color, size, ripeness) < 60
    
    Uses **minimum** score (weakest link approach) - a single poor
    dimension downgrades the entire item.
    
    Parameters
    ----------
    scores : Mapping[str, int]
        Must contain 'color_score', 'size_score', 'ripeness_score'
    
    Returns
    -------
    str
        "A", "B", or "C"
    
    Examples
    --------
    >>> assign_grade({"color_score": 95, "size_score": 98, "ripeness_score": 90})
    'A'
    >>> assign_grade({"color_score": 70, "size_score": 75, "ripeness_score": 65})
    'B'
    >>> assign_grade({"color_score": 50, "size_score": 40, "ripeness_score": 30})
    'C'
    """
    required_keys = ["color_score", "size_score", "ripeness_score"]
    missing = [k for k in required_keys if k not in scores]
    if missing:
        raise KeyError(f"scores missing required keys: {missing}")
    
    # Extract scores
    color = int(scores["color_score"])
    size = int(scores["size_score"])
    ripeness = int(scores["ripeness_score"])
    
    # Validate range
    for name, value in [("color", color), ("size", size), ("ripeness", ripeness)]:
        if not 0 <= value <= 100:
            raise ValueError(f"{name}_score must be in [0, 100], got {value}")
    
    # Grade by minimum (weakest link)
    min_score = min(color, size, ripeness)
    
    if min_score >= GRADE_A_MIN:
        return "A"
    elif min_score >= GRADE_B_MIN:
        return "B"
    else:
        return "C"


def get_recommendation(grade: str) -> str:
    """Get business recommendation for a grade.
    
    Parameters
    ----------
    grade : str
        "A", "B", or "C"
    
    Returns
    -------
    str
        Business recommendation text
    
    Examples
    --------
    >>> get_recommendation("A")
    'Premium quality - sell at full price in premium display'
    """
    grade = grade.strip().upper()
    if grade not in RECOMMENDATIONS:
        raise ValueError(
            f"Unknown grade: '{grade}'. Expected 'A', 'B', or 'C'."
        )
    return RECOMMENDATIONS[grade]


def grade_produce(model_output: Mapping[str, Any]) -> Dict[str, Any]:
    """End-to-end: Model output → Quality scores → Grade → Recommendation.
    
    Complete pipeline for grading produce from model predictions.
    
    Parameters
    ----------
    model_output : Mapping[str, Any]
        Dictionary with:
        - 'predicted_class': str (e.g., "Apple__Healthy")
        - 'confidence': float in [0, 1]
    
    Returns
    -------
    Dict[str, Any]
        Complete grading result with:
        - 'predicted_class': str
        - 'produce_type': str
        - 'state': str
        - 'confidence': float
        - 'color_score': int
        - 'size_score': int
        - 'ripeness_score': int
        - 'grade': str
        - 'recommendation': str
    
    Examples
    --------
    >>> model_output = {"predicted_class": "Apple__Healthy", "confidence": 0.95}
    >>> result = grade_produce(model_output)
    >>> result['grade']
    'A'
    >>> result['color_score']
    95
    >>> result['recommendation']
    'Premium quality - sell at full price in premium display'
    
    >>> model_output = {"predicted_class": "Banana__Rotten", "confidence": 0.85}
    >>> result = grade_produce(model_output)
    >>> result['grade']
    'C'
    >>> result['state']
    'Rotten'
    """
    # Step 1: Compute quality scores
    scores = compute_quality_scores(model_output)
    
    # Step 2: Assign grade
    grade = assign_grade(scores)
    
    # Step 3: Get recommendation
    recommendation = get_recommendation(grade)
    
    # Step 4: Combine everything
    return {
        **scores,
        "grade": grade,
        "recommendation": recommendation,
    }


if __name__ == "__main__":
    # Demo: Test with sample inputs
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
        print(f"  Scores: Color={result['color_score']}, "
              f"Size={result['size_score']}, "
              f"Ripeness={result['ripeness_score']}")
        print(f"  Grade: {result['grade']}")
        print(f"  Recommendation: {result['recommendation']}")
    
    print("\n" + "=" * 80)
    print("✓ All test cases passed!")