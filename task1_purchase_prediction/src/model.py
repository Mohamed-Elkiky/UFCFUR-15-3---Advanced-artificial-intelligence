"""
Training pipeline for Task 1 Purchase Prediction.

Trains a Random Forest classifier that, given a customer's purchase history,
predicts the next product they are most likely to order.

The feature-engineering logic is imported from ``preprocess.py`` so that
training and inference always build features in the same way.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from task1_purchase_prediction.src.preprocess import (
    history_to_feature_row,
    load_class_names,
    load_orders,
)


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ORDERS_PATH = BASE_DIR / "data" / "raw" / "orders.csv"
DEFAULT_PRODUCTS_PATH = BASE_DIR / "data" / "raw" / "products.csv"
MODELS_DIR = BASE_DIR / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "reorder_model.pkl"
DEFAULT_CLASS_NAMES_PATH = MODELS_DIR / "class_names.pkl"


def build_training_dataset(
    orders_df: pd.DataFrame,
    class_names: Sequence[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build a supervised training set of (history, next_product) pairs.

    For each customer, orders are sorted chronologically. For each order
    at position ``idx`` (with ``idx >= 1``) the feature vector summarises
    orders ``[0, idx)`` and the label is the product purchased at ``idx``.

    Parameters
    ----------
    orders_df : pd.DataFrame
        All orders across all customers.
    class_names : Sequence[str]
        The product catalogue used as the feature/label space.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        ``(X, y)`` where ``X`` is the feature matrix and ``y`` contains
        the next-product labels.

    Raises
    ------
    ValueError
        If no training samples could be constructed (e.g. every customer
        has fewer than two orders).
    """
    X_rows: List[dict] = []
    y_rows: List[str] = []

    for _, customer_df in orders_df.groupby("customer_id"):
        sort_cols = ["order_date"]
        if "order_id" in customer_df.columns:
            sort_cols.append("order_id")
        customer_df = customer_df.sort_values(sort_cols).reset_index(drop=True)

        # Need at least 2 orders to predict a next item
        if len(customer_df) < 2:
            continue

        for idx in range(1, len(customer_df)):
            history = customer_df.iloc[:idx]
            next_product = str(customer_df.iloc[idx]["product"])

            feature_row = history_to_feature_row(history, class_names)
            X_rows.append(feature_row)
            y_rows.append(next_product)

    if not X_rows:
        raise ValueError("No training samples could be generated from orders.csv")

    X = pd.DataFrame(X_rows)
    y = pd.Series(y_rows, name="target_product")

    return X, y


def train_reorder_model(
    orders_path: str | Path = DEFAULT_ORDERS_PATH,
    products_path: str | Path = DEFAULT_PRODUCTS_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    class_names_path: str | Path = DEFAULT_CLASS_NAMES_PATH,
) -> RandomForestClassifier:
    """
    Train the purchase-prediction model and persist it to disk.

    Parameters
    ----------
    orders_path : str | Path
        Path to the orders CSV file.
    products_path : str | Path
        Path to the products CSV file (defines the label space).
    model_path : str | Path
        Destination path for the trained model (``.pkl``).
    class_names_path : str | Path
        Destination path for the serialised class-name list (``.pkl``).

    Returns
    -------
    RandomForestClassifier
        The trained classifier.
    """
    orders_df = load_orders(orders_path)
    class_names = load_class_names(products_path, fallback_orders_path=orders_path)

    X, y = build_training_dataset(orders_df, class_names)

    # Keep only labels that exist in the product catalogue
    valid_mask = y.isin(class_names)
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)

    if X.empty or y.empty:
        raise ValueError("Training dataset is empty after filtering valid labels.")

    # Stratification fails if any class appears fewer than 2 times
    label_counts = y.value_counts()
    can_stratify = (label_counts.min() >= 2) and (y.nunique() > 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if can_stratify else None,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\nTraining complete.")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape:  {X_test.shape}")
    print(f"Accuracy:    {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(list(class_names), class_names_path)

    print(f"\nSaved model to: {Path(model_path)}")
    print(f"Saved class names to: {Path(class_names_path)}")

    return model


if __name__ == "__main__":
    train_reorder_model()