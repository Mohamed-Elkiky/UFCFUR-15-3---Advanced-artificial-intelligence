"""
Tests for Task 1 Purchase Prediction (AA-17).

Covers the two functions required by AA-17:
    - ``get_quick_reorder``: frequency-based top-N reorder suggestions
    - ``predict_reorder``:   model-based top-N next-product predictions

Tests use synthetic in-memory data written to a ``tmp_path`` fixture so
they are isolated from the real dataset and trained model.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from task1_purchase_prediction.src.predict import (
    get_quick_reorder,
    predict_reorder,
)
from task1_purchase_prediction.src.preprocess import history_to_feature_row


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_orders() -> pd.DataFrame:
    """A small synthetic orders dataframe covering two customers."""
    return pd.DataFrame(
        [
            # CUST001 likes Milk (4x) and Bread (2x), one Apples
            {"order_id": "O1", "customer_id": "CUST001", "product": "Milk",
             "quantity": 2, "order_date": "2024-01-01"},
            {"order_id": "O2", "customer_id": "CUST001", "product": "Milk",
             "quantity": 2, "order_date": "2024-01-08"},
            {"order_id": "O3", "customer_id": "CUST001", "product": "Bread",
             "quantity": 1, "order_date": "2024-01-10"},
            {"order_id": "O4", "customer_id": "CUST001", "product": "Milk",
             "quantity": 3, "order_date": "2024-01-15"},
            {"order_id": "O5", "customer_id": "CUST001", "product": "Bread",
             "quantity": 1, "order_date": "2024-01-20"},
            {"order_id": "O6", "customer_id": "CUST001", "product": "Milk",
             "quantity": 2, "order_date": "2024-01-25"},
            {"order_id": "O7", "customer_id": "CUST001", "product": "Apples",
             "quantity": 1, "order_date": "2024-02-01"},
            # CUST002 has different preferences
            {"order_id": "O8", "customer_id": "CUST002", "product": "Cheese",
             "quantity": 1, "order_date": "2024-01-05"},
            {"order_id": "O9", "customer_id": "CUST002", "product": "Cheese",
             "quantity": 2, "order_date": "2024-01-12"},
        ]
    )


@pytest.fixture
def orders_csv(tmp_path: Path, sample_orders: pd.DataFrame) -> Path:
    """Write the sample orders to a temp CSV file."""
    path = tmp_path / "orders.csv"
    sample_orders.to_csv(path, index=False)
    return path


@pytest.fixture
def trained_model_artifacts(tmp_path: Path, sample_orders: pd.DataFrame):
    """Train a tiny classifier on the sample data and save it to tmp."""
    class_names = ["Milk", "Bread", "Apples", "Cheese"]

    # Build a minimal training set: one feature row per customer
    training_rows = []
    labels = []
    for customer_id, group in sample_orders.groupby("customer_id"):
        # Use all-but-last as history, last product as label
        history = group.iloc[:-1]
        label = group.iloc[-1]["product"]
        row = history_to_feature_row(history, class_names)
        training_rows.append(row)
        labels.append(label)

    X = pd.DataFrame(training_rows)
    y = pd.Series(labels)

    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)

    model_path = tmp_path / "model.pkl"
    class_names_path = tmp_path / "class_names.pkl"
    joblib.dump(model, model_path)
    joblib.dump(class_names, class_names_path)

    return model_path, class_names_path


# ---------------------------------------------------------------------------
# get_quick_reorder
# ---------------------------------------------------------------------------

def test_quick_reorder_returns_most_frequent_first(orders_csv: Path):
    """Most-ordered product for CUST001 is Milk (4 orders)."""
    results = get_quick_reorder("CUST001", top_n=5, orders_path=orders_csv)

    assert len(results) == 3  # Milk, Bread, Apples
    assert results[0]["product"] == "Milk"
    assert results[0]["order_count"] == 4
    assert results[1]["product"] == "Bread"
    assert results[1]["order_count"] == 2


def test_quick_reorder_respects_top_n(orders_csv: Path):
    """Passing top_n=1 should return exactly one suggestion."""
    results = get_quick_reorder("CUST001", top_n=1, orders_path=orders_csv)

    assert len(results) == 1
    assert results[0]["product"] == "Milk"


def test_quick_reorder_unknown_customer_returns_empty(orders_csv: Path):
    """A customer that has never ordered anything should return []."""
    results = get_quick_reorder("CUST999", top_n=5, orders_path=orders_csv)

    assert results == []


def test_quick_reorder_result_schema(orders_csv: Path):
    """Each suggestion must expose the documented keys."""
    results = get_quick_reorder("CUST001", top_n=5, orders_path=orders_csv)

    expected_keys = {"product", "order_count", "total_quantity", "last_order_date"}
    for item in results:
        assert set(item.keys()) == expected_keys


# ---------------------------------------------------------------------------
# predict_reorder
# ---------------------------------------------------------------------------

def test_predict_reorder_returns_ranked_suggestions(
    sample_orders: pd.DataFrame, trained_model_artifacts
):
    """Predictions should be returned sorted by descending confidence."""
    model_path, class_names_path = trained_model_artifacts

    results = predict_reorder(
        "CUST001",
        sample_orders,
        top_n=3,
        model_path=model_path,
        class_names_path=class_names_path,
    )

    # Model only knows the classes it was trained on, so we may get fewer
    # than top_n results. We assert we got at least one and that they are
    # returned in descending order of confidence.
    assert len(results) >= 1
    confidences = [r["confidence_score"] for r in results]
    assert confidences == sorted(confidences, reverse=True)


def test_predict_reorder_empty_history_returns_empty(trained_model_artifacts):
    """An empty history dataframe should short-circuit to an empty list."""
    model_path, class_names_path = trained_model_artifacts

    results = predict_reorder(
        "CUST001",
        pd.DataFrame(),
        top_n=3,
        model_path=model_path,
        class_names_path=class_names_path,
    )

    assert results == []


def test_predict_reorder_unknown_customer_returns_empty(
    sample_orders: pd.DataFrame, trained_model_artifacts
):
    """A customer not present in the history should return []."""
    model_path, class_names_path = trained_model_artifacts

    results = predict_reorder(
        "CUST999",
        sample_orders,
        top_n=3,
        model_path=model_path,
        class_names_path=class_names_path,
    )

    assert results == []


def test_predict_reorder_result_schema(
    sample_orders: pd.DataFrame, trained_model_artifacts
):
    """Each prediction must expose product and confidence_score keys."""
    model_path, class_names_path = trained_model_artifacts

    results = predict_reorder(
        "CUST001",
        sample_orders,
        top_n=3,
        model_path=model_path,
        class_names_path=class_names_path,
    )

    for item in results:
        assert set(item.keys()) == {"product", "confidence_score"}
        assert 0.0 <= item["confidence_score"] <= 1.0