"""
Shared preprocessing utilities for Task 1 Purchase Prediction.

This module centralises the feature-engineering logic used by both the
training pipeline (``model.py``) and the inference pipeline (``predict.py``),
ensuring that features are constructed identically at train and predict time.

The feature representation for a single sample is:
    - ``freq_<product>``: number of times the customer has ordered <product>
    - ``qty_<product>``:  total quantity the customer has ordered of <product>

for every product in the known product catalogue.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd


REQUIRED_ORDER_COLUMNS = {"customer_id", "product", "quantity", "order_date"}


def load_orders(orders_path: str | Path) -> pd.DataFrame:
    """
    Load and validate an ``orders.csv`` file.

    Parameters
    ----------
    orders_path : str | Path
        Path to the orders CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned orders dataframe sorted by customer and order date.

    Raises
    ------
    FileNotFoundError
        If the orders file does not exist.
    ValueError
        If required columns are missing from the file.
    """
    orders_path = Path(orders_path)
    if not orders_path.exists():
        raise FileNotFoundError(f"Orders file not found: {orders_path}")

    df = pd.read_csv(orders_path)

    missing = REQUIRED_ORDER_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in orders.csv: {sorted(missing)}")

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


def load_class_names(
    products_path: str | Path,
    fallback_orders_path: str | Path | None = None,
) -> List[str]:
    """
    Load the list of product class names from ``products.csv``.

    Parameters
    ----------
    products_path : str | Path
        Path to the products CSV file.
    fallback_orders_path : str | Path, optional
        If ``products.csv`` is missing, fall back to deriving class names
        from the unique products in this orders file.

    Returns
    -------
    List[str]
        Ordered list of product names used as the feature and label space.

    Raises
    ------
    ValueError
        If ``products.csv`` exists but is missing a ``product`` column.
    FileNotFoundError
        If neither the products file nor the fallback orders file is available.
    """
    products_path = Path(products_path)

    if products_path.exists():
        products_df = pd.read_csv(products_path)
        if "product" not in products_df.columns:
            raise ValueError("products.csv must contain a 'product' column")
        return products_df["product"].astype(str).tolist()

    if fallback_orders_path is not None:
        orders_df = load_orders(fallback_orders_path)
        return sorted(orders_df["product"].astype(str).unique().tolist())

    raise FileNotFoundError(
        f"Products file not found and no fallback provided: {products_path}"
    )


def empty_feature_dict(class_names: Sequence[str]) -> Dict[str, float]:
    """
    Build a feature dictionary with all product features initialised to zero.

    Parameters
    ----------
    class_names : Sequence[str]
        The product catalogue used as the feature space.

    Returns
    -------
    Dict[str, float]
        A dict containing ``freq_<product>`` and ``qty_<product>`` keys
        for every product, all set to ``0.0``.
    """
    feature_row: Dict[str, float] = {}
    for product in class_names:
        feature_row[f"freq_{product}"] = 0.0
        feature_row[f"qty_{product}"] = 0.0
    return feature_row


def history_to_feature_row(
    history_df: pd.DataFrame,
    class_names: Sequence[str],
) -> Dict[str, float]:
    """
    Convert a customer's purchase history into a single feature row.

    For every product in ``class_names`` the function records:
        - how many orders in the history contained that product (``freq_*``)
        - the total quantity of that product across the history (``qty_*``)

    Products that do not appear in ``class_names`` are ignored so that
    feature vectors remain consistent with the trained model.

    Parameters
    ----------
    history_df : pd.DataFrame
        Orders belonging to a single customer. Must contain ``product``
        and ``quantity`` columns.
    class_names : Sequence[str]
        The product catalogue used as the feature space.

    Returns
    -------
    Dict[str, float]
        Feature dict with ``freq_<product>`` and ``qty_<product>`` keys
        for every product in ``class_names``.
    """
    feature_row = empty_feature_dict(class_names)

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


def history_to_feature_frame(
    history_df: pd.DataFrame,
    class_names: Sequence[str],
) -> pd.DataFrame:
    """
    Convert a customer's purchase history into a single-row feature DataFrame.

    This is a thin wrapper around :func:`history_to_feature_row` that
    returns a DataFrame suitable for passing directly to scikit-learn's
    ``predict`` / ``predict_proba``.

    Parameters
    ----------
    history_df : pd.DataFrame
        Orders belonging to a single customer.
    class_names : Sequence[str]
        The product catalogue used as the feature space.

    Returns
    -------
    pd.DataFrame
        A DataFrame with exactly one row and ``2 * len(class_names)`` columns.
    """
    feature_row = history_to_feature_row(history_df, class_names)
    return pd.DataFrame([feature_row])