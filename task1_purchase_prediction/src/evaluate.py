"""
Evaluation module for Task 1 Purchase Prediction model.

This module provides comprehensive evaluation metrics for the reorder
prediction model including accuracy, precision, recall, F1-score, and ROC-AUC.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ORDERS_PATH = BASE_DIR / "data" / "raw" / "orders.csv"
DEFAULT_PRODUCTS_PATH = BASE_DIR / "data" / "raw" / "products.csv"
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "reorder_model.pkl"
DEFAULT_CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.pkl"

REQUIRED_COLUMNS = {"customer_id", "product", "quantity", "order_date"}


def _load_orders(orders_path: str | Path = DEFAULT_ORDERS_PATH) -> pd.DataFrame:
    """Load and validate orders data."""
    orders_path = Path(orders_path)
    if not orders_path.exists():
        raise FileNotFoundError(f"Orders file not found: {orders_path}")

    df = pd.read_csv(orders_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["customer_id"] = df["customer_id"].astype(str)
    df["product"] = df["product"].astype(str)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df.dropna(subset=["order_date"])

    sort_cols = ["customer_id", "order_date"]
    if "order_id" in df.columns:
        sort_cols.append("order_id")

    return df.sort_values(sort_cols).reset_index(drop=True)


def _load_class_names(products_path: str | Path = DEFAULT_PRODUCTS_PATH) -> List[str]:
    """Load product class names."""
    products_path = Path(products_path)

    if products_path.exists():
        products_df = pd.read_csv(products_path)
        if "product" not in products_df.columns:
            raise ValueError("products.csv must contain a 'product' column")
        return products_df["product"].astype(str).tolist()

    orders_df = _load_orders()
    return sorted(orders_df["product"].astype(str).unique().tolist())


def _empty_feature_dict(class_names: List[str]) -> Dict[str, float]:
    """Create empty feature dictionary."""
    feature_row: Dict[str, float] = {}
    for product in class_names:
        feature_row[f"freq_{product}"] = 0.0
        feature_row[f"qty_{product}"] = 0.0
    return feature_row


def _history_to_feature_row(
    history_df: pd.DataFrame,
    class_names: List[str],
) -> Dict[str, float]:
    """Convert purchase history to feature row."""
    feature_row = _empty_feature_dict(class_names)

    if history_df.empty:
        return feature_row

    grouped = (
        history_df.groupby("product", as_index=False)
        .agg(
            order_count=("product", "count"),
            total_quantity=("quantity", "sum"),
        )
    )

    for _, row in grouped.iterrows():
        product = str(row["product"])
        if product in class_names:
            feature_row[f"freq_{product}"] = float(row["order_count"])
            feature_row[f"qty_{product}"] = float(row["total_quantity"])

    return feature_row


def build_training_dataset(
    orders_df: pd.DataFrame,
    class_names: List[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build supervised training dataset.
    
    Parameters
    ----------
    orders_df : pd.DataFrame
        Orders data
    class_names : List[str]
        List of product names
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Features (X) and labels (y)
    """
    X_rows: List[Dict[str, float]] = []
    y_rows: List[str] = []

    for customer_id, customer_df in orders_df.groupby("customer_id"):
        customer_df = customer_df.sort_values(
            ["order_date"] + (["order_id"] if "order_id" in customer_df.columns else [])
        )
        customer_df = customer_df.reset_index(drop=True)

        if len(customer_df) < 2:
            continue

        for idx in range(1, len(customer_df)):
            history = customer_df.iloc[:idx]
            next_product = str(customer_df.iloc[idx]["product"])

            feature_row = _history_to_feature_row(history, class_names)
            X_rows.append(feature_row)
            y_rows.append(next_product)

    if not X_rows:
        raise ValueError("No training samples generated")

    X = pd.DataFrame(X_rows)
    y = pd.Series(y_rows, name="target_product")

    return X, y


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_proba: Optional[np.ndarray] = None,
    average: str = "weighted",
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_proba : np.ndarray, optional
        Prediction probabilities for ROC-AUC calculation
    average : str, default='weighted'
        Averaging method for multi-class metrics
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - accuracy: Overall accuracy
        - precision: Precision score
        - recall: Recall score
        - f1: F1 score
        - roc_auc: ROC-AUC score (if y_proba provided)
    """
    metrics = {}
    
    # Basic metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(
        y_true, y_pred, average=average, zero_division=0
    )
    metrics["recall"] = recall_score(
        y_true, y_pred, average=average, zero_division=0
    )
    metrics["f1"] = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    # ROC-AUC for multi-class (one-vs-rest)
    if y_proba is not None:
        try:
            # Get unique classes
            classes = np.unique(y_true)
            
            # Binarize labels for multi-class ROC-AUC
            y_true_bin = label_binarize(y_true, classes=classes)
            
            # Handle binary classification case
            if len(classes) == 2:
                metrics["roc_auc"] = roc_auc_score(
                    y_true, y_proba[:, 1], average=average
                )
            else:
                # Multi-class case
                metrics["roc_auc"] = roc_auc_score(
                    y_true_bin, y_proba, average=average, multi_class="ovr"
                )
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not compute ROC-AUC: {e}")
            metrics["roc_auc"] = np.nan
    else:
        metrics["roc_auc"] = np.nan
    
    return metrics


def print_full_report(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    target_names: Optional[List[str]] = None,
    digits: int = 4,
) -> None:
    """
    Print detailed classification report.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    target_names : List[str], optional
        Display names for classes
    digits : int, default=4
        Number of decimal places for metrics
    """
    print("\n" + "=" * 80)
    print("CLASSIFICATION REPORT")
    print("=" * 80)
    
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=digits,
        zero_division=0,
    )
    print(report)
    
    print("=" * 80 + "\n")


def get_confusion_matrix(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    labels: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    labels : List[str], optional
        List of label names in order
        
    Returns
    -------
    np.ndarray
        Confusion matrix
    """
    return confusion_matrix(y_true, y_pred, labels=labels)


def evaluate_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    orders_path: str | Path = DEFAULT_ORDERS_PATH,
    products_path: str | Path = DEFAULT_PRODUCTS_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Full model evaluation pipeline.
    
    Parameters
    ----------
    model_path : str | Path
        Path to saved model
    orders_path : str | Path
        Path to orders data
    products_path : str | Path
        Path to products data
    test_size : float, default=0.2
        Test set proportion
    random_state : int, default=42
        Random seed
    verbose : bool, default=True
        Print detailed output
        
    Returns
    -------
    Tuple containing:
        - metrics: Dict of evaluation metrics
        - y_test: True labels
        - y_pred: Predicted labels
        - y_proba: Prediction probabilities
    """
    # Load model
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    
    # Load data
    orders_df = _load_orders(orders_path)
    class_names = _load_class_names(products_path)
    
    # Build dataset
    X, y = build_training_dataset(orders_df, class_names)
    
    # Filter valid labels
    valid_mask = y.isin(class_names)
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)
    
    if X.empty or y.empty:
        raise ValueError("Dataset is empty after filtering")
    
    # Split data (same as training)
    label_counts = y.value_counts()
    can_stratify = (label_counts.min() >= 2) and (y.nunique() > 1)
    
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if can_stratify else None,
    )
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    
    # Compute metrics
    metrics = compute_metrics(y_test, y_pred, y_proba)
    
    if verbose:
        print("\n" + "=" * 80)
        print("MODEL EVALUATION RESULTS")
        print("=" * 80)
        print(f"Test samples: {len(y_test)}")
        print(f"Number of classes: {y_test.nunique()}")
        print("\nMetrics Summary:")
        print("-" * 80)
        for metric_name, value in metrics.items():
            if not np.isnan(value):
                print(f"{metric_name.upper():.<30} {value:.4f}")
            else:
                print(f"{metric_name.upper():.<30} N/A")
        print("=" * 80)
        
        # Full classification report
        print_full_report(y_test, y_pred, target_names=None)
        
        # Confusion matrix info
        cm = get_confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix Shape: {cm.shape}")
        print(f"Confusion Matrix Sample (top-left 5x5):")
        print(cm[:5, :5])
        print("=" * 80 + "\n")
    
    return metrics, y_test, y_pred, y_proba


if __name__ == "__main__":
    """Run evaluation on the trained model."""
    try:
        metrics, y_test, y_pred, y_proba = evaluate_model(verbose=True)
        print("\nEvaluation completed successfully!")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please train the model first using: python -m task1_purchase_prediction.src.model")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        raise