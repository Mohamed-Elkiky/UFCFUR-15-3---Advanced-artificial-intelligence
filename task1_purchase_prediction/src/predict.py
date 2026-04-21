"""
Inference pipeline for Task 1 Purchase Prediction.

Exposes two entry points used by the API and by offline demos:

    - :func:`get_quick_reorder`: a pure frequency-based lookup that returns
      the products a customer has ordered most often. Does not require the
      trained model and is therefore resilient to model-file absence.

    - :func:`predict_reorder`: runs the trained classifier against a
      customer's history and returns the top-N next-product suggestions
      with their confidence scores.

Feature engineering is delegated to ``preprocess.py`` so that inference
features are guaranteed to match the representation used at training time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import joblib
import numpy as np
import pandas as pd

from task1_purchase_prediction.src.preprocess import (
    history_to_feature_frame,
    load_class_names,
    load_orders,
)


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ORDERS_PATH = BASE_DIR / "data" / "raw" / "orders.csv"
DEFAULT_PRODUCTS_PATH = BASE_DIR / "data" / "raw" / "products.csv"
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "reorder_model.pkl"
DEFAULT_CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.pkl"


def _load_class_names_cached(
    class_names_path: str | Path = DEFAULT_CLASS_NAMES_PATH,
    products_path: str | Path = DEFAULT_PRODUCTS_PATH,
) -> List[str]:
    """Load class names, preferring the serialised list saved at training time."""
    class_names_path = Path(class_names_path)
    if class_names_path.exists():
        return list(joblib.load(class_names_path))
    return load_class_names(products_path)


def _load_model(model_path: str | Path = DEFAULT_MODEL_PATH):
    """Load the trained classifier from disk."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def get_quick_reorder(
    customer_id: str,
    top_n: int = 5,
    orders_path: str | Path = DEFAULT_ORDERS_PATH,
) -> List[Dict]:
    """
    Return the ``top_n`` products a customer has ordered most often.

    This is a non-ML, frequency-based suggestion used to give customers
    a fast "reorder your usuals" experience. Ordering tiebreakers fall
    back to total quantity and then recency of last order.

    Parameters
    ----------
    customer_id : str
        The customer whose history should be analysed.
    top_n : int, default=5
        Maximum number of products to return.
    orders_path : str | Path, default=DEFAULT_ORDERS_PATH
        Path to the orders CSV file.

    Returns
    -------
    List[Dict]
        A list of suggestion dicts, each containing ``product``,
        ``order_count``, ``total_quantity`` and ``last_order_date``.
        Returns an empty list if the customer has no recorded orders.
    """
    orders_df = load_orders(orders_path)
    customer_history = orders_df[orders_df["customer_id"] == customer_id].copy()

    if customer_history.empty:
        return []

    grouped = (
        customer_history.groupby("product", as_index=False)
        .agg(
            order_count=("product", "count"),
            total_quantity=("quantity", "sum"),
            last_order_date=("order_date", "max"),
        )
        .sort_values(
            by=["order_count", "total_quantity", "last_order_date"],
            ascending=[False, False, False],
        )
        .head(top_n)
    )

    results: List[Dict] = []
    for _, row in grouped.iterrows():
        results.append(
            {
                "product": row["product"],
                "order_count": int(row["order_count"]),
                "total_quantity": float(row["total_quantity"]),
                "last_order_date": row["last_order_date"].strftime("%Y-%m-%d"),
            }
        )

    return results


def predict_reorder(
    customer_id: str,
    history: Union[pd.DataFrame, List[Dict]],
    top_n: int = 5,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    class_names_path: str | Path = DEFAULT_CLASS_NAMES_PATH,
    products_path: str | Path = DEFAULT_PRODUCTS_PATH,
) -> List[Dict]:
    """
    Predict a customer's next likely purchases using the trained model.

    The customer's history is converted to the model's feature
    representation, the classifier's class probabilities are computed,
    and the top-N most probable products are returned.

    Parameters
    ----------
    customer_id : str
        The customer whose next purchases should be predicted.
    history : pd.DataFrame | List[Dict]
        Purchase history containing at minimum ``customer_id``, ``product``
        and ``quantity`` fields. Only rows matching ``customer_id`` are used.
    top_n : int, default=5
        Maximum number of product suggestions to return.
    model_path : str | Path
        Path to the serialised trained model.
    class_names_path : str | Path
        Path to the serialised class-name list.
    products_path : str | Path
        Fallback path used to recover class names if the serialised list
        cannot be found.

    Returns
    -------
    List[Dict]
        Suggestions sorted by descending confidence, each containing
        ``product`` and ``confidence_score``. Returns an empty list if
        the customer has no history to predict from.
    """
    if isinstance(history, list):
        history_df = pd.DataFrame(history)
    else:
        history_df = history.copy()

    if history_df.empty:
        return []

    history_df["customer_id"] = history_df["customer_id"].astype(str)
    history_df["product"] = history_df["product"].astype(str)
    history_df["quantity"] = pd.to_numeric(history_df["quantity"], errors="coerce").fillna(0)

    customer_history = history_df[history_df["customer_id"] == customer_id].copy()
    if customer_history.empty:
        return []

    class_names = _load_class_names_cached(
        class_names_path=class_names_path,
        products_path=products_path,
    )
    model = _load_model(model_path)

    X = history_to_feature_frame(customer_history, class_names)

    # Align columns with the model's expected feature names if available
    if hasattr(model, "feature_names_in_"):
        X = X.reindex(columns=model.feature_names_in_, fill_value=0)

    results: List[Dict] = []

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        model_classes = [str(c) for c in model.classes_]
        top_indices = np.argsort(probabilities)[::-1][:top_n]

        for idx in top_indices:
            results.append(
                {
                    "product": model_classes[int(idx)],
                    "confidence_score": round(float(probabilities[int(idx)]), 4),
                }
            )
    else:
        predicted = model.predict(X)
        results.append(
            {
                "product": str(predicted[0]),
                "confidence_score": 1.0,
            }
        )

    return results


def forecast_demand(
    product: str,
    orders_path: str | Path = DEFAULT_ORDERS_PATH,
    n_months: int = 12,
) -> List[Dict]:
    """Forecast monthly demand and return structured data for charting (AA-52).

    Computes both the current-year predicted demand and the previous-year
    actual demand for each calendar month, along with a trend label. The
    forecast uses a simple seasonal model: the historical monthly average
    plus a linear trend term estimated from the last 24 months of data.

    Parameters
    ----------
    product : str
        Product name to forecast (must match values in orders.csv).
    orders_path : str | Path
        Path to the orders CSV file.
    n_months : int, default=12
        Number of months to forecast forward from the most recent data month.

    Returns
    -------
    List[Dict]
        One dict per month with keys:
        ``month_name``, ``predicted_demand``, ``last_year_demand``,
        ``trend_label``.
        ``trend_label`` is one of ``"rising"``, ``"stable"``, ``"falling"``.
    """
    import calendar

    orders_df = load_orders(orders_path)
    product_orders = orders_df[orders_df["product"] == product].copy()

    if product_orders.empty:
        return []

    product_orders["year_month"] = product_orders["order_date"].dt.to_period("M")
    monthly = (
        product_orders.groupby("year_month", as_index=False)
        .agg(demand=("quantity", "sum"))
    )
    monthly = monthly.sort_values("year_month").reset_index(drop=True)

    # Build a month-of-year seasonal baseline (average demand per calendar month)
    monthly["month"] = monthly["year_month"].dt.month
    seasonal_avg = monthly.groupby("month")["demand"].mean().to_dict()

    # Estimate a linear trend from the last 24 data points
    recent = monthly.tail(24)
    if len(recent) >= 2:
        x = np.arange(len(recent))
        slope, intercept = np.polyfit(x, recent["demand"].values, 1)
    else:
        slope = 0.0
        intercept = float(recent["demand"].mean()) if not recent.empty else 0.0

    if slope > 0.5:
        base_trend = "rising"
    elif slope < -0.5:
        base_trend = "falling"
    else:
        base_trend = "stable"

    # Determine the starting month for forecast
    if not monthly.empty:
        last_period = monthly["year_month"].iloc[-1]
        start_year = last_period.year
        start_month = last_period.month + 1
        if start_month > 12:
            start_month = 1
            start_year += 1
    else:
        import datetime
        now = datetime.date.today()
        start_year, start_month = now.year, now.month

    results: List[Dict] = []
    for i in range(n_months):
        month = (start_month - 1 + i) % 12 + 1
        year = start_year + (start_month - 1 + i) // 12

        seasonal_base = seasonal_avg.get(month, float(np.mean(list(seasonal_avg.values()))))
        trend_offset = slope * (len(monthly) + i)
        predicted_demand = max(0.0, round(seasonal_base + trend_offset, 1))

        # Last year's actual demand for the same month
        last_year_period = f"{year - 1}-{month:02d}"
        ly_rows = monthly[monthly["year_month"].astype(str) == last_year_period]
        last_year_demand = (
            float(ly_rows["demand"].iloc[0]) if not ly_rows.empty else None
        )

        # Refine trend label per month
        if last_year_demand is not None and last_year_demand > 0:
            ratio = predicted_demand / last_year_demand
            if ratio > 1.10:
                trend_label = "rising"
            elif ratio < 0.90:
                trend_label = "falling"
            else:
                trend_label = "stable"
        else:
            trend_label = base_trend

        results.append(
            {
                "month_name": f"{calendar.month_name[month]} {year}",
                "predicted_demand": predicted_demand,
                "last_year_demand": last_year_demand,
                "trend_label": trend_label,
            }
        )

    return results


if __name__ == "__main__":
    customer_id = "CUST001"

    print(f"\nQuick reorder suggestions for {customer_id}:")
    quick_reorders = get_quick_reorder(customer_id, top_n=5)
    for i, item in enumerate(quick_reorders, start=1):
        print(
            f"{i}. {item['product']} | "
            f"orders={item['order_count']} | "
            f"qty={item['total_quantity']} | "
            f"last={item['last_order_date']}"
        )

    try:
        orders_df = load_orders(DEFAULT_ORDERS_PATH)
        predictions = predict_reorder(customer_id, orders_df, top_n=5)

        print(f"\nModel predictions for {customer_id}:")
        for i, item in enumerate(predictions, start=1):
            print(
                f"{i}. {item['product']} | "
                f"confidence={item['confidence_score']}"
            )
    except FileNotFoundError as exc:
        print(f"\nModel prediction skipped: {exc}")