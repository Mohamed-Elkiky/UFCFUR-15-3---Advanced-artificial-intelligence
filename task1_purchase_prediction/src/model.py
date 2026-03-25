from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ORDERS_PATH = BASE_DIR / "data" / "raw" / "orders.csv"
DEFAULT_PRODUCTS_PATH = BASE_DIR / "data" / "raw" / "products.csv"
MODELS_DIR = BASE_DIR / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "reorder_model.pkl"
DEFAULT_CLASS_NAMES_PATH = MODELS_DIR / "class_names.pkl"

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

    # Stable ordering for sequence generation
    sort_cols = ["customer_id", "order_date"]
    if "order_id" in df.columns:
        sort_cols.append("order_id")

    return df.sort_values(sort_cols).reset_index(drop=True)


def _load_class_names(products_path: str | Path = DEFAULT_PRODUCTS_PATH) -> List[str]:
    products_path = Path(products_path)

    if products_path.exists():
        products_df = pd.read_csv(products_path)
        if "product" not in products_df.columns:
            raise ValueError("products.csv must contain a 'product' column")
        return products_df["product"].astype(str).tolist()

    # Fallback from orders.csv if products.csv is missing
    orders_df = _load_orders()
    return sorted(orders_df["product"].astype(str).unique().tolist())


def _empty_feature_dict(class_names: Sequence[str]) -> Dict[str, float]:
    feature_row: Dict[str, float] = {}
    for product in class_names:
        feature_row[f"freq_{product}"] = 0.0
        feature_row[f"qty_{product}"] = 0.0
    return feature_row


def _history_to_feature_row(
    history_df: pd.DataFrame,
    class_names: Sequence[str],
) -> Dict[str, float]:
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
    class_names: Sequence[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build supervised samples:
    features = customer's purchase history up to time t
    label    = next product purchased at time t+1
    """
    X_rows: List[Dict[str, float]] = []
    y_rows: List[str] = []

    for customer_id, customer_df in orders_df.groupby("customer_id"):
        customer_df = customer_df.sort_values(["order_date"] + (["order_id"] if "order_id" in customer_df.columns else []))
        customer_df = customer_df.reset_index(drop=True)

        # Need at least 2 orders to predict a next item
        if len(customer_df) < 2:
            continue

        for idx in range(1, len(customer_df)):
            history = customer_df.iloc[:idx]
            next_product = str(customer_df.iloc[idx]["product"])

            feature_row = _history_to_feature_row(history, class_names)
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
    orders_df = _load_orders(orders_path)
    class_names = _load_class_names(products_path)

    X, y = build_training_dataset(orders_df, class_names)

    # Keep only labels that are in class_names
    valid_mask = y.isin(class_names)
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)

    if X.empty or y.empty:
        raise ValueError("Training dataset is empty after filtering valid labels.")

    # If some classes are extremely rare, stratify may fail
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