from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd


TARGET_COLUMN = "is_reorder"

MODEL_FEATURE_COLUMNS = [
    "customer_type",
    "producer_id",
    "product",
    "quantity",
    "unit_price",
    "total_price",
    "month",
    "season",
    "purchase_frequency",
    "days_since_last_order",
    "avg_quantity",
]

ENCODER_COLUMNS = ["product", "customer_type", "season", "producer_id"]

NUMERIC_COLUMNS_TO_SCALE = [
    "quantity",
    "unit_price",
    "total_price",
    "month",
    "purchase_frequency",
    "days_since_last_order",
    "avg_quantity",
]


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_raw_orders_path() -> Path:
    return get_project_root() / "data" / "raw" / "orders.csv"


def get_models_dir() -> Path:
    return get_project_root() / "models"


def get_artifacts_dir() -> Path:
    return get_project_root() / "artifacts"


def load_raw_orders() -> pd.DataFrame:
    path = get_raw_orders_path()
    if not path.exists():
        raise FileNotFoundError(f"orders.csv not found at: {path}")

    df = pd.read_csv(path)
    required_columns = [
        "order_id",
        "customer_id",
        "customer_type",
        "producer_id",
        "product",
        "quantity",
        "unit_price",
        "total_price",
        "order_date",
        "month",
        "season",
        "is_reorder",
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"orders.csv is missing required columns: {missing}")

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    df["total_price"] = pd.to_numeric(df["total_price"], errors="coerce")
    df["month"] = pd.to_numeric(df["month"], errors="coerce")
    df["is_reorder"] = pd.to_numeric(df["is_reorder"], errors="coerce")

    return df


def load_model():
    model_path = get_models_dir() / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found at: {model_path}")
    return joblib.load(model_path)


def load_encoders() -> Dict[str, Any]:
    artifacts_dir = get_artifacts_dir()
    encoders: Dict[str, Any] = {}

    for col in ENCODER_COLUMNS:
        enc_path = artifacts_dir / f"{col}_label_encoder.joblib"
        if not enc_path.exists():
            raise FileNotFoundError(f"Encoder not found for '{col}' at: {enc_path}")
        encoders[col] = joblib.load(enc_path)

    return encoders


def load_scaler():
    scaler_path = get_artifacts_dir() / "numeric_scaler.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at: {scaler_path}")
    return joblib.load(scaler_path)


def month_to_season(month: int) -> str:
    month = int(month)
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def get_quick_reorder(customer_id: str, top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Group orders by customer_id + product, count frequency,
    and return the top N most ordered items for that customer.
    """
    if top_n <= 0:
        raise ValueError("top_n must be greater than 0.")

    orders_df = load_raw_orders()

    customer_orders = orders_df[orders_df["customer_id"].astype(str) == str(customer_id)].copy()
    customer_orders = customer_orders.dropna(subset=["product", "quantity", "order_date"])

    if customer_orders.empty:
        return []

    summary = (
        customer_orders.groupby(["customer_id", "product"], as_index=False)
        .agg(
            order_count=("product", "size"),
            total_quantity=("quantity", "sum"),
            avg_quantity=("quantity", "mean"),
            latest_order_date=("order_date", "max"),
        )
        .sort_values(
            ["order_count", "total_quantity", "avg_quantity", "latest_order_date"],
            ascending=[False, False, False, False],
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    summary["avg_quantity"] = summary["avg_quantity"].round(2)
    summary["latest_order_date"] = summary["latest_order_date"].dt.strftime("%Y-%m-%d")

    return summary.to_dict(orient="records")


def _safe_transform(encoder, value: str) -> int:
    value = str(value)
    known_classes = set(map(str, encoder.classes_))

    if value in known_classes:
        return int(encoder.transform([value])[0])

    fallback = str(encoder.classes_[0])
    return int(encoder.transform([fallback])[0])


def _prepare_history_dataframe(customer_id: str, history: List[Dict[str, Any]]) -> pd.DataFrame:
    if not history:
        raise ValueError("history is empty.")

    df = pd.DataFrame(history).copy()

    required_minimum = ["product", "quantity", "unit_price", "order_date"]
    missing = [col for col in required_minimum if col not in df.columns]
    if missing:
        raise ValueError(f"history is missing required fields: {missing}")

    df["customer_id"] = str(customer_id)

    if "customer_type" not in df.columns:
        df["customer_type"] = "household"
    else:
        df["customer_type"] = df["customer_type"].fillna("household")

    if "producer_id" not in df.columns:
        df["producer_id"] = "PROD001"
    else:
        df["producer_id"] = df["producer_id"].fillna("PROD001")

    df["product"] = df["product"].astype(str)
    df["customer_type"] = df["customer_type"].astype(str)
    df["producer_id"] = df["producer_id"].astype(str)

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")

    if "total_price" not in df.columns:
        df["total_price"] = df["quantity"] * df["unit_price"]
    else:
        df["total_price"] = pd.to_numeric(df["total_price"], errors="coerce")
        df["total_price"] = df["total_price"].fillna(df["quantity"] * df["unit_price"])

    df = df.dropna(subset=["product", "quantity", "unit_price", "order_date"]).copy()
    if df.empty:
        raise ValueError("history became empty after cleaning invalid rows.")

    df["quantity"] = df["quantity"].round().astype(int)
    df["unit_price"] = df["unit_price"].round(2)
    df["total_price"] = (df["quantity"] * df["unit_price"]).round(2)

    df["month"] = df["order_date"].dt.month

    if "season" not in df.columns:
        df["season"] = df["month"].apply(month_to_season)
    else:
        df["season"] = df["season"].where(df["season"].notna(), df["month"].apply(month_to_season))
        df["season"] = df["season"].astype(str)

    return df.sort_values(["product", "order_date"]).reset_index(drop=True)


def _build_candidate_rows(customer_id: str, history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one candidate row per previously ordered product.

    Important:
    The trained model predicts binary reorder probability (`is_reorder`), not product classes.
    So each candidate row represents one known product from the customer's history,
    and we rank those products by reorder probability.
    """
    rows: List[Dict[str, Any]] = []
    last_global_date = history_df["order_date"].max()

    for product_name, product_history in history_df.groupby("product"):
        product_history = product_history.sort_values("order_date").reset_index(drop=True)

        purchase_frequency = int(len(product_history))
        avg_quantity = float(product_history["quantity"].mean())

        last_row = product_history.iloc[-1]
        last_order_date = last_row["order_date"]
        days_since_last_order = float((last_global_date - last_order_date).days)

        representative_quantity = max(1, int(round(avg_quantity)))
        representative_unit_price = float(last_row["unit_price"])

        rows.append(
            {
                "customer_id": str(customer_id),
                "product_name": str(product_name),
                "customer_type": str(last_row["customer_type"]),
                "producer_id": str(last_row["producer_id"]),
                "product": str(product_name),
                "quantity": representative_quantity,
                "unit_price": representative_unit_price,
                "total_price": round(representative_quantity * representative_unit_price, 2),
                "month": int(last_global_date.month),
                "season": str(month_to_season(int(last_global_date.month))),
                "purchase_frequency": float(purchase_frequency),
                "days_since_last_order": float(days_since_last_order),
                "avg_quantity": float(avg_quantity),
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def _encode_and_scale_features(candidate_df: pd.DataFrame) -> pd.DataFrame:
    encoders = load_encoders()
    scaler = load_scaler()

    df = candidate_df.copy()

    for col in ENCODER_COLUMNS:
        df[col] = df[col].apply(lambda x: _safe_transform(encoders[col], x))

    df[NUMERIC_COLUMNS_TO_SCALE] = scaler.transform(df[NUMERIC_COLUMNS_TO_SCALE])

    return df[MODEL_FEATURE_COLUMNS].copy()


def predict_reorder(customer_id: str, history: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Preprocess the history list into model features, run model prediction,
    and return reorder suggestions with confidence scores.

    Correction to the task wording:
    the current pipeline predicts binary reorder likelihood, not product class indices.
    So we score candidate products and return the highest-confidence suggestions.
    """
    if top_n <= 0:
        raise ValueError("top_n must be greater than 0.")

    history_df = _prepare_history_dataframe(customer_id, history)
    candidate_df = _build_candidate_rows(customer_id, history_df)

    if candidate_df.empty:
        return []

    X_candidates = _encode_and_scale_features(candidate_df)
    model = load_model()

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_candidates)
        classes = list(model.classes_)
        if 1 in classes:
            positive_class_index = classes.index(1)
            reorder_scores = proba[:, positive_class_index]
        else:
            reorder_scores = proba.max(axis=1)
    else:
        reorder_scores = model.predict(X_candidates).astype(float)

    candidate_df["confidence_score"] = reorder_scores
    candidate_df["predicted_reorder"] = (candidate_df["confidence_score"] >= 0.5).astype(int)

    suggestions_df = (
        candidate_df.sort_values(
            ["confidence_score", "purchase_frequency", "avg_quantity", "days_since_last_order"],
            ascending=[False, False, False, True],
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    suggestions: List[Dict[str, Any]] = []
    for _, row in suggestions_df.iterrows():
        suggestions.append(
            {
                "customer_id": str(customer_id),
                "product": row["product_name"],
                "predicted_reorder": int(row["predicted_reorder"]),
                "confidence_score": round(float(row["confidence_score"]), 4),
                "purchase_frequency": int(row["purchase_frequency"]),
                "avg_quantity": round(float(row["avg_quantity"]), 2),
                "days_since_last_order": round(float(row["days_since_last_order"]), 2),
            }
        )

    return suggestions


def forecast_demand(product: str, months_ahead: int = 4) -> List[Dict[str, Any]]:
    """
    Group historical orders by product + month, compute average demand per calendar month,
    and project forward using a simple seasonal pattern plus linear trend.

    Returns:
        [
            {"month": "2025-04", "expected_demand": 123.4, "trend": "up"},
            ...
        ]
    """
    if months_ahead <= 0:
        raise ValueError("months_ahead must be greater than 0.")

    orders_df = load_raw_orders()
    product_df = orders_df[orders_df["product"].astype(str) == str(product)].copy()

    if product_df.empty:
        return []

    product_df = product_df.dropna(subset=["order_date", "quantity"]).copy()
    if product_df.empty:
        return []

    product_df["year_month"] = product_df["order_date"].dt.to_period("M").astype(str)
    product_df["calendar_month"] = product_df["order_date"].dt.month.astype(int)

    monthly_totals = (
        product_df.groupby("year_month", as_index=False)["quantity"]
        .sum()
        .rename(columns={"quantity": "monthly_demand"})
    )
    monthly_totals["year_month_dt"] = pd.to_datetime(monthly_totals["year_month"] + "-01")
    monthly_totals = monthly_totals.sort_values("year_month_dt").reset_index(drop=True)

    monthly_totals["calendar_month"] = monthly_totals["year_month_dt"].dt.month.astype(int)

    seasonal_profile = (
        monthly_totals.groupby("calendar_month", as_index=False)["monthly_demand"]
        .mean()
        .rename(columns={"monthly_demand": "avg_monthly_demand"})
        .sort_values("calendar_month")
        .reset_index(drop=True)
    )

    seasonal_lookup = {
        int(row["calendar_month"]): float(row["avg_monthly_demand"])
        for _, row in seasonal_profile.iterrows()
    }

    overall_mean = float(monthly_totals["monthly_demand"].mean())

    if len(monthly_totals) >= 2:
        x = np.arange(len(monthly_totals), dtype=float)
        y = monthly_totals["monthly_demand"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
    else:
        slope = 0.0
        intercept = overall_mean

    last_month_dt = monthly_totals["year_month_dt"].max()
    last_observed_demand = float(monthly_totals.iloc[-1]["monthly_demand"])

    forecasts: List[Dict[str, Any]] = []

    for step in range(1, months_ahead + 1):
        future_month_dt = last_month_dt + pd.DateOffset(months=step)
        future_calendar_month = int(future_month_dt.month)

        seasonal_base = seasonal_lookup.get(future_calendar_month, overall_mean)
        expected_demand = max(0.0, float(seasonal_base + slope * step))

        if expected_demand > last_observed_demand * 1.05:
            trend = "up"
        elif expected_demand < last_observed_demand * 0.95:
            trend = "down"
        else:
            trend = "stable"

        forecasts.append(
            {
                "month": future_month_dt.strftime("%Y-%m"),
                "expected_demand": round(expected_demand, 2),
                "trend": trend,
            }
        )

    return forecasts


if __name__ == "__main__":
    demo_customer_id = "CUST001"

    print("Quick reorder based on raw purchase counts:")
    print(get_quick_reorder(demo_customer_id, top_n=5))

    raw_orders = load_raw_orders()
    demo_history_df = (
        raw_orders[raw_orders["customer_id"].astype(str) == demo_customer_id]
        .sort_values("order_date")
        .tail(20)
        .copy()
    )

    demo_history = demo_history_df[
        [
            "product",
            "quantity",
            "unit_price",
            "order_date",
            "customer_type",
            "producer_id",
            "season",
            "total_price",
        ]
    ].to_dict(orient="records")

    print("\nPredicted reorder suggestions:")
    print(predict_reorder(demo_customer_id, demo_history, top_n=5))

    print("\nForecast demand for Tomatoes:")
    print(forecast_demand("Tomatoes", months_ahead=4))

    print("\nForecast demand for Milk:")
    print(forecast_demand("Milk", months_ahead=4))