"""
Evaluation module for Task 1 Purchase Prediction model.

This module provides comprehensive evaluation metrics for the reorder
prediction model including accuracy, precision, recall, F1-score, and ROC-AUC.

It also provides a fairness / bias audit that groups model predictions by
producer and flags any producer whose share of recommendations is more
than 20% above the mean. This directly addresses the Fairness,
Accountability and Trust requirement of the case study.
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

# Threshold above the mean recommendation rate at which a producer is
# flagged as potentially over-represented in the model's predictions.
BIAS_FLAG_MULTIPLIER = 1.20

# AA-50: standard-deviation multiplier for the updated bias flagging logic.
# Producers whose recommendation rate is more than STD_FLAG_THRESHOLD standard
# deviations above the cross-producer mean are flagged.
STD_FLAG_THRESHOLD = 1.5


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

    for _, customer_df in orders_df.groupby("customer_id"):
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


def build_training_dataset_with_context(
    orders_df: pd.DataFrame,
    class_names: List[str],
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Build a supervised training dataset that also preserves order context.

    This is an extended version of :func:`build_training_dataset` that
    additionally returns the ``producer_id`` and ``customer_id`` associated
    with each training sample's target product. Those columns are required
    for fairness / bias analysis, where we need to know which producer
    actually supplied the product the model is trying to predict.

    Parameters
    ----------
    orders_df : pd.DataFrame
        Orders data. Must include a ``producer_id`` column.
    class_names : List[str]
        List of product names.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame]
        ``(X, y, context)`` where ``context`` has columns
        ``producer_id`` and ``customer_id`` aligned row-wise with ``X`` and ``y``.

    Raises
    ------
    ValueError
        If ``producer_id`` is missing or no samples can be generated.
    """
    if "producer_id" not in orders_df.columns:
        raise ValueError("orders_df must contain a 'producer_id' column for bias audit")

    X_rows: List[Dict[str, float]] = []
    y_rows: List[str] = []
    producer_ids: List[str] = []
    customer_ids: List[str] = []

    for customer_id, customer_df in orders_df.groupby("customer_id"):
        customer_df = customer_df.sort_values(
            ["order_date"] + (["order_id"] if "order_id" in customer_df.columns else [])
        )
        customer_df = customer_df.reset_index(drop=True)

        if len(customer_df) < 2:
            continue

        for idx in range(1, len(customer_df)):
            history = customer_df.iloc[:idx]
            target_row = customer_df.iloc[idx]
            next_product = str(target_row["product"])

            feature_row = _history_to_feature_row(history, class_names)
            X_rows.append(feature_row)
            y_rows.append(next_product)
            producer_ids.append(str(target_row["producer_id"]))
            customer_ids.append(str(customer_id))

    if not X_rows:
        raise ValueError("No training samples generated")

    X = pd.DataFrame(X_rows)
    y = pd.Series(y_rows, name="target_product")
    context = pd.DataFrame(
        {
            "producer_id": producer_ids,
            "customer_id": customer_ids,
        }
    )

    return X, y, context


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

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(
        y_true, y_pred, average=average, zero_division=0
    )
    metrics["recall"] = recall_score(
        y_true, y_pred, average=average, zero_division=0
    )
    metrics["f1"] = f1_score(y_true, y_pred, average=average, zero_division=0)

    if y_proba is not None:
        try:
            classes = np.unique(y_true)
            y_true_bin = label_binarize(y_true, classes=classes)

            if len(classes) == 2:
                metrics["roc_auc"] = roc_auc_score(
                    y_true, y_proba[:, 1], average=average
                )
            else:
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
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)

    orders_df = _load_orders(orders_path)
    class_names = _load_class_names(products_path)

    X, y = build_training_dataset(orders_df, class_names)

    valid_mask = y.isin(class_names)
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)

    if X.empty or y.empty:
        raise ValueError("Dataset is empty after filtering")

    label_counts = y.value_counts()
    can_stratify = (label_counts.min() >= 2) and (y.nunique() > 1)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if can_stratify else None,
    )

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

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

        print_full_report(y_test, y_pred, target_names=None)

        cm = get_confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix Shape: {cm.shape}")
        print(f"Confusion Matrix Sample (top-left 5x5):")
        print(cm[:5, :5])
        print("=" * 80 + "\n")

    return metrics, y_test, y_pred, y_proba


# ---------------------------------------------------------------------------
# AA-21: Bias / Fairness Audit
# ---------------------------------------------------------------------------


def build_predictions_df(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    orders_path: str | Path = DEFAULT_ORDERS_PATH,
    products_path: str | Path = DEFAULT_PRODUCTS_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Build a predictions dataframe suitable for per-producer fairness auditing.

    Re-runs the same train/test split used by :func:`evaluate_model`, then
    emits one row per test sample with the ground-truth product, the model's
    prediction, and the ``producer_id`` of the true order. The producer_id
    reflects who actually supplied the target product, which is the correct
    attribution for asking "does the model predict equally accurately
    across producers?".

    Parameters
    ----------
    model_path : str | Path
        Path to the saved model.
    orders_path : str | Path
        Path to orders data (must include ``producer_id``).
    products_path : str | Path
        Path to products data.
    test_size : float, default=0.2
        Test set proportion. Must match training for a coherent audit.
    random_state : int, default=42
        Random seed. Must match training for a coherent audit.

    Returns
    -------
    pd.DataFrame
        Columns: ``customer_id``, ``producer_id``, ``true_product``,
        ``predicted_product``, ``is_correct``.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)

    orders_df = _load_orders(orders_path)
    class_names = _load_class_names(products_path)

    X, y, context = build_training_dataset_with_context(orders_df, class_names)

    valid_mask = y.isin(class_names)
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)
    context = context.loc[valid_mask].reset_index(drop=True)

    if X.empty:
        raise ValueError("Dataset is empty after filtering")

    label_counts = y.value_counts()
    can_stratify = (label_counts.min() >= 2) and (y.nunique() > 1)

    # Keep context aligned with the test set by splitting indices, not rows
    indices = np.arange(len(X))
    _, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y if can_stratify else None,
    )

    X_test = X.iloc[test_indices].reset_index(drop=True)
    y_test = y.iloc[test_indices].reset_index(drop=True)
    context_test = context.iloc[test_indices].reset_index(drop=True)

    y_pred = model.predict(X_test)

    predictions_df = pd.DataFrame(
        {
            "customer_id": context_test["customer_id"].astype(str),
            "producer_id": context_test["producer_id"].astype(str),
            "true_product": y_test.astype(str).values,
            "predicted_product": pd.Series(y_pred).astype(str).values,
        }
    )
    predictions_df["is_correct"] = (
        predictions_df["true_product"] == predictions_df["predicted_product"]
    )

    return predictions_df


def bias_audit(
    predictions_df: pd.DataFrame,
    flag_multiplier: float = BIAS_FLAG_MULTIPLIER,
    std_threshold: float = STD_FLAG_THRESHOLD,
) -> Dict:
    """
    Group predictions by producer and flag potentially biased outcomes (AA-50).

    For each producer this computes:
        - ``precision``: fraction of test samples where the model was correct.
        - ``recommendation_rate``: that producer's share of all test samples.
        - ``avg_confidence``: mean model confidence for that producer's samples
          (requires a ``confidence`` column; set to NaN if absent).
        - ``deviation``: how many standard deviations the producer's
          recommendation rate is above the cross-producer mean.
        - ``sample_count``: how many test samples belong to that producer.

    A producer is flagged when its recommendation rate is more than
    ``std_threshold`` standard deviations above the cross-producer mean
    (AA-50 requirement). The legacy ``flag_multiplier`` threshold is retained
    as a secondary check and stored separately in the returned dict.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Must contain ``producer_id``, ``true_product``, ``predicted_product``,
        and ``is_correct`` columns (as produced by :func:`build_predictions_df`).
        An optional ``confidence`` column is used for avg_confidence when present.
    flag_multiplier : float, default=1.20
        Legacy multiplier threshold (kept for backwards compatibility).
    std_threshold : float, default=1.5
        Number of standard deviations above the mean at which a producer is
        flagged (AA-50 primary criterion).

    Returns
    -------
    Dict
        ``{
            "per_producer": DataFrame,
            "mean_recommendation_rate": float,
            "std_recommendation_rate": float,
            "flag_threshold": float,
            "flag_threshold_std": float,
            "flag_threshold_multiplier": float,
            "flagged_producers": List[str],
        }``

    Raises
    ------
    ValueError
        If required columns are missing from ``predictions_df``.
    """
    required = {"producer_id", "true_product", "predicted_product", "is_correct"}
    missing = required - set(predictions_df.columns)
    if missing:
        raise ValueError(
            f"predictions_df is missing required columns: {sorted(missing)}"
        )

    if predictions_df.empty:
        return {
            "per_producer": pd.DataFrame(
                columns=[
                    "producer_id",
                    "sample_count",
                    "precision",
                    "recommendation_rate",
                    "avg_confidence",
                    "deviation",
                    "flagged",
                ]
            ),
            "mean_recommendation_rate": 0.0,
            "std_recommendation_rate": 0.0,
            "flag_threshold": 0.0,
            "flag_threshold_std": 0.0,
            "flag_threshold_multiplier": 0.0,
            "flagged_producers": [],
        }

    total_samples = len(predictions_df)
    has_confidence = "confidence" in predictions_df.columns

    agg_dict: Dict = {
        "sample_count": ("producer_id", "count"),
        "correct_count": ("is_correct", "sum"),
    }
    if has_confidence:
        agg_dict["avg_confidence"] = ("confidence", "mean")

    per_producer = predictions_df.groupby("producer_id", as_index=False).agg(
        **agg_dict
    )

    per_producer["precision"] = (
        per_producer["correct_count"] / per_producer["sample_count"]
    )
    per_producer["recommendation_rate"] = (
        per_producer["sample_count"] / total_samples
    )

    if not has_confidence:
        per_producer["avg_confidence"] = np.nan

    mean_rec_rate = float(per_producer["recommendation_rate"].mean())
    std_rec_rate = float(per_producer["recommendation_rate"].std(ddof=0))

    # AA-50: flag producers > std_threshold standard deviations above mean
    flag_threshold_std = mean_rec_rate + std_threshold * std_rec_rate
    flag_threshold_mult = mean_rec_rate * flag_multiplier

    per_producer["deviation"] = (
        (per_producer["recommendation_rate"] - mean_rec_rate)
        / std_rec_rate
        if std_rec_rate > 0
        else 0.0
    )
    per_producer["flagged"] = per_producer["recommendation_rate"] > flag_threshold_std

    flagged_producers = per_producer.loc[
        per_producer["flagged"], "producer_id"
    ].tolist()

    per_producer = per_producer.sort_values(
        "recommendation_rate", ascending=False
    ).reset_index(drop=True)

    display_cols = [
        "producer_id",
        "sample_count",
        "precision",
        "recommendation_rate",
        "avg_confidence",
        "deviation",
        "flagged",
    ]

    return {
        "per_producer": per_producer[display_cols],
        "mean_recommendation_rate": mean_rec_rate,
        "std_recommendation_rate": std_rec_rate,
        "flag_threshold": float(flag_threshold_mult),
        "flag_threshold_std": float(flag_threshold_std),
        "flag_threshold_multiplier": float(flag_threshold_mult),
        "flagged_producers": flagged_producers,
    }


def print_bias_audit(audit_result: Dict) -> None:
    """
    Print a human-readable summary of a bias audit result.

    Parameters
    ----------
    audit_result : Dict
        The dictionary returned by :func:`bias_audit`.
    """
    print("\n" + "=" * 80)
    print("BIAS / FAIRNESS AUDIT (per producer)")
    print("=" * 80)

    per_producer = audit_result["per_producer"]
    if per_producer.empty:
        print("No predictions to audit.")
        print("=" * 80 + "\n")
        return

    print(
        f"Mean recommendation rate across producers: "
        f"{audit_result['mean_recommendation_rate']:.4f}"
    )
    print(
        f"Std recommendation rate: "
        f"{audit_result.get('std_recommendation_rate', 0.0):.4f}"
    )
    print(
        f"Flag threshold (mean + {STD_FLAG_THRESHOLD} std): "
        f"{audit_result.get('flag_threshold_std', audit_result.get('flag_threshold', 0.0)):.4f}"
    )
    print("-" * 80)

    for _, row in per_producer.iterrows():
        marker = "  FLAGGED" if row["flagged"] else ""
        deviation = row.get("deviation", float("nan"))
        dev_str = f"{deviation:+.2f}σ" if not (isinstance(deviation, float) and deviation != deviation) else "N/A"
        print(
            f"{row['producer_id']:<10} "
            f"samples={int(row['sample_count']):<5} "
            f"precision={row['precision']:.4f}   "
            f"rec_rate={row['recommendation_rate']:.4f}  "
            f"dev={dev_str}"
            f"{marker}"
        )

    print("-" * 80)
    threshold_key = "flag_threshold_std" if "flag_threshold_std" in audit_result else "flag_threshold"
    if audit_result["flagged_producers"]:
        print(
            f"Flagged producers (rec_rate > {STD_FLAG_THRESHOLD}σ above mean, "
            f"threshold={audit_result[threshold_key]:.4f}): "
            f"{audit_result['flagged_producers']}"
        )
    else:
        print("No producers exceeded the flag threshold.")
    print("=" * 80 + "\n")

# ---------------------------------------------------------------------------
# AA-22: Override Analysis & Monitoring Strategy
# ---------------------------------------------------------------------------


def generate_synthetic_interactions(
    predictions_df: pd.DataFrame,
    n_weeks: int = 26,
    base_override_rate: float = 0.12,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate user interactions where customers accept or override suggestions."""
    rng = np.random.default_rng(seed)

    customer_types = {
        "CUST001": "household", "CUST002": "household",
        "CUST003": "household", "CUST004": "household",
        "CUST005": "household", "CUST006": "restaurant",
        "CUST007": "restaurant", "CUST008": "retailer",
        "CUST009": "retailer", "CUST010": "cafe",
        "CUST011": "cafe", "CUST012": "hotel",
        "CUST013": "household", "CUST014": "restaurant",
        "CUST015": "retailer",
    }

    products = predictions_df["true_product"].unique().tolist()
    start_date = pd.Timestamp("2025-09-01")
    rows: List[Dict] = []

    for week_num in range(n_weeks):
        week_start = start_date + pd.Timedelta(weeks=week_num)
        staleness_factor = 1.0 + 0.015 * week_num
        n_interactions = rng.integers(25, 50)
        sampled = predictions_df.sample(
            n=min(n_interactions, len(predictions_df)),
            replace=True,
            random_state=int(rng.integers(0, 2**31)),
        )

        for _, row in sampled.iterrows():
            if row["is_correct"]:
                p_override = base_override_rate * 0.4 * staleness_factor
            else:
                p_override = min(0.95, base_override_rate * 3.5 * staleness_factor)

            was_overridden = bool(rng.random() < p_override)

            if was_overridden:
                alternatives = [p for p in products if p != row["predicted_product"]]
                actual = rng.choice(alternatives) if alternatives else row["predicted_product"]
            else:
                actual = row["predicted_product"]

            day_offset = int(rng.integers(0, 7))
            rows.append({
                "interaction_id": f"INT-{len(rows)+1:05d}",
                "customer_id": str(row["customer_id"]),
                "customer_type": customer_types.get(str(row["customer_id"]), "unknown"),
                "predicted_product": str(row["predicted_product"]),
                "actual_product": str(actual),
                "was_overridden": was_overridden,
                "date": week_start + pd.Timedelta(days=day_offset),
                "week": week_num + 1,
            })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def override_analysis(interactions_df: pd.DataFrame) -> Dict:
    """Compute override rates by product, customer type and week."""
    required = {"was_overridden", "predicted_product", "customer_type", "week"}
    missing = required - set(interactions_df.columns)
    if missing:
        raise ValueError(f"interactions_df is missing columns: {sorted(missing)}")

    n_total = len(interactions_df)
    n_overrides = int(interactions_df["was_overridden"].sum())
    overall_rate = n_overrides / n_total if n_total > 0 else 0.0

    by_product = (
        interactions_df.groupby("predicted_product", as_index=False)
        .agg(interactions=("was_overridden", "count"), overrides=("was_overridden", "sum"))
    )
    by_product["override_rate"] = by_product["overrides"] / by_product["interactions"]
    by_product = by_product.sort_values("override_rate", ascending=False).reset_index(drop=True)

    by_customer = (
        interactions_df.groupby("customer_type", as_index=False)
        .agg(interactions=("was_overridden", "count"), overrides=("was_overridden", "sum"))
    )
    by_customer["override_rate"] = by_customer["overrides"] / by_customer["interactions"]
    by_customer = by_customer.sort_values("override_rate", ascending=False).reset_index(drop=True)

    weekly = (
        interactions_df.groupby("week", as_index=False)
        .agg(interactions=("was_overridden", "count"), overrides=("was_overridden", "sum"))
    )
    weekly["override_rate"] = weekly["overrides"] / weekly["interactions"]
    weekly["rolling_avg"] = weekly["override_rate"].rolling(window=4, min_periods=1).mean()

    retrain_threshold = 0.20
    weekly["alert"] = weekly["rolling_avg"] > retrain_threshold
    alert_weeks = weekly.loc[weekly["alert"], "week"].tolist()

    return {
        "overall_override_rate": float(overall_rate),
        "by_product": by_product,
        "by_customer_type": by_customer,
        "weekly_trend": weekly,
        "n_interactions": n_total,
        "n_overrides": n_overrides,
        "alert_triggered": len(alert_weeks) > 0,
        "alert_weeks": alert_weeks,
        "retrain_threshold": retrain_threshold,
    }


def print_override_analysis(result: Dict) -> None:
    """Print override analysis summary."""
    print("\n" + "=" * 80)
    print("OVERRIDE & MONITORING ANALYSIS (AA-22)")
    print("=" * 80)
    print(f"Total interactions : {result['n_interactions']}")
    print(f"Total overrides    : {result['n_overrides']}")
    print(f"Overall override % : {result['overall_override_rate']:.2%}")
    print(f"Retrain threshold  : {result['retrain_threshold']:.0%} (rolling 4-week avg)")

    print("\n--- Override Rate by Product ---")
    print(result["by_product"].to_string(index=False))

    print("\n--- Override Rate by Customer Type ---")
    print(result["by_customer_type"].to_string(index=False))

    if result["alert_triggered"]:
        print(f"\n⚠  RETRAIN ALERT: rolling override rate exceeded "
              f"{result['retrain_threshold']:.0%} in weeks: {result['alert_weeks']}")
    else:
        print("\n✓  No retrain alert — override rate within acceptable limits.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    """Run evaluation on the trained model."""
    try:
        metrics, y_test, y_pred, y_proba = evaluate_model(verbose=True)
        print("\nEvaluation completed successfully!")

        # Bias audit
        predictions_df = build_predictions_df()
        audit_result = bias_audit(predictions_df)
        print_bias_audit(audit_result)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please train the model first using: python -m task1_purchase_prediction.src.model")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        raise