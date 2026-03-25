from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ORDERS_PATH = BASE_DIR / "data" / "raw" / "orders.csv"
DEFAULT_PRODUCTS_PATH = BASE_DIR / "data" / "raw" / "products.csv"
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "reorder_model.pkl"
DEFAULT_CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.pkl"


REQUIRED_COLUMNS = {"customer_id", "product", "quantity", "order_date"}


def _load_orders(orders_path: str | Path = DEFAULT_ORDERS_PATH) -> pd.DataFrame:
    orders_path = Path(orders_path)
    if not orders_path.exists():
        raise FileNotFoundError(f"Orders file not found: {orders_path}")

    df = pd.read_csv(orders_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in orders.csv: {sorted(missing)}")

    df = df.copy()
    df["customer_id"] = df["customer_id"].astype(str)
    df["product"] = df["product"].astype(str)
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df.dropna(subset=["order_date"])

    return df.sort_values("order_date").reset_index(drop=True)


def _load_class_names(
    class_names_path: str | Path = DEFAULT_CLASS_NAMES_PATH,
    products_path: str | Path = DEFAULT_PRODUCTS_PATH,
) -> List[str]:
    class_names_path = Path(class_names_path)
    if class_names_path.exists():
        class_names = joblib.load(class_names_path)
        return list(class_names)

    products_path = Path(products_path)
    if products_path.exists():
        products_df = pd.read_csv(products_path)
        if "product" not in products_df.columns:
            raise ValueError("products.csv must contain a 'product' column")
        return products_df["product"].astype(str).tolist()

    raise FileNotFoundError(
        f"Neither class names file nor products file could be found:\n"
        f"- {class_names_path}\n"
        f"- {products_path}"
    )


def _load_model(model_path: str | Path = DEFAULT_MODEL_PATH):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def _build_feature_vector(
    customer_history: pd.DataFrame,
    class_names: List[str],
) -> pd.DataFrame:
    """
    Convert one customer's grouped purchase history into a single-row feature vector.
    Each feature is the frequency count and total quantity for one product.
    """
    grouped = (
        customer_history.groupby("product", as_index=False)
        .agg(
            order_count=("product", "count"),
            total_quantity=("quantity", "sum"),
        )
    )

    frequency_lookup = dict(zip(grouped["product"], grouped["order_count"]))
    quantity_lookup = dict(zip(grouped["product"], grouped["total_quantity"]))

    feature_row: Dict[str, float] = {}

    for product in class_names:
        feature_row[f"freq_{product}"] = float(frequency_lookup.get(product, 0))
        feature_row[f"qty_{product}"] = float(quantity_lookup.get(product, 0))

    return pd.DataFrame([feature_row])


def get_quick_reorder(
    customer_id: str,
    top_n: int = 5,
    orders_path: str | Path = DEFAULT_ORDERS_PATH,
) -> List[Dict]:
    """
    Group orders by customer_id + product, count frequency,
    and return the top N most ordered items.
    """
    orders_df = _load_orders(orders_path)
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
    history: pd.DataFrame | List[Dict],
    top_n: int = 5,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    class_names_path: str | Path = DEFAULT_CLASS_NAMES_PATH,
    products_path: str | Path = DEFAULT_PRODUCTS_PATH,
) -> List[Dict]:
    """
    Preprocess the history into model features, run model.predict()/predict_proba(),
    map predictions back to product names, and return suggestions with confidence scores.
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

    class_names = _load_class_names(class_names_path=class_names_path, products_path=products_path)
    model = _load_model(model_path)

    X = _build_feature_vector(customer_history, class_names)

    if hasattr(model, "feature_names_in_"):
        X = X.reindex(columns=model.feature_names_in_, fill_value=0)

    predicted = model.predict(X)

    if isinstance(predicted[0], (str, np.str_)):
        top_prediction = str(predicted[0])
    else:
        top_prediction = class_names[int(predicted[0])]

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
        results.append(
            {
                "product": top_prediction,
                "confidence_score": 1.0,
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
        orders_df = _load_orders()
        predictions = predict_reorder(customer_id, orders_df, top_n=5)

        print(f"\nModel predictions for {customer_id}:")
        for i, item in enumerate(predictions, start=1):
            print(
                f"{i}. {item['product']} | "
                f"confidence={item['confidence_score']}"
            )
    except FileNotFoundError as exc:
        print(f"\nModel prediction skipped: {exc}")