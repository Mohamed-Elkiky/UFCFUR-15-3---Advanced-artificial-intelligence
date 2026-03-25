from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


REQUIRED_COLUMNS = [
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

CATEGORICAL_COLUMNS = [
    "product",
    "customer_type",
    "season",
    "producer_id",
]

NUMERIC_COLUMNS_TO_SCALE = [
    "quantity",
    "unit_price",
    "total_price",
    "month",
    "purchase_frequency",
    "days_since_last_order",
    "avg_quantity",
]

TARGET_COLUMN = "is_reorder"


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_config_path() -> Path:
    return get_project_root() / "config.yaml"


def get_raw_data_path() -> Path:
    return get_project_root() / "data" / "raw" / "orders.csv"


def get_processed_dir() -> Path:
    processed_dir = get_project_root() / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir


def get_artifacts_dir() -> Path:
    artifacts_dir = get_project_root() / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir


def load_config() -> Dict:
    config_path = get_config_path()
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    return config


def resolve_test_size(config: Dict) -> float:
    """
    Accepts either:
    - test_size at the root of config.yaml
    - preprocessing.test_size
    """
    if "test_size" in config:
        return float(config["test_size"])

    preprocessing_cfg = config.get("preprocessing", {})
    if "test_size" in preprocessing_cfg:
        return float(preprocessing_cfg["test_size"])

    return 0.2


def load_raw_data() -> pd.DataFrame:
    """
    Read the raw orders CSV and enforce the expected schema.
    """
    raw_path = get_raw_data_path()
    if not raw_path.exists():
        raise FileNotFoundError(f"orders.csv not found at: {raw_path}")

    df = pd.read_csv(raw_path)

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"orders.csv is missing required columns: {missing_cols}")

    df = df[REQUIRED_COLUMNS].copy()

    df["order_id"] = df["order_id"].astype("string")
    df["customer_id"] = df["customer_id"].astype("string")
    df["customer_type"] = df["customer_type"].astype("string")
    df["producer_id"] = df["producer_id"].astype("string")
    df["product"] = df["product"].astype("string")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    df["total_price"] = pd.to_numeric(df["total_price"], errors="coerce")
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["month"] = pd.to_numeric(df["month"], errors="coerce")
    df["season"] = df["season"].astype("string")
    df["is_reorder"] = pd.to_numeric(df["is_reorder"], errors="coerce")

    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Missing-value strategy:
    1. Drop rows missing critical identifiers or target:
       - order_id, customer_id, customer_type, producer_id, product, order_date, is_reorder
    2. Impute numeric business fields with median:
       - quantity, unit_price, total_price, month
    3. Impute season from month where possible; otherwise use mode.
    4. Recompute total_price from quantity * unit_price after imputation.
    5. Cast final types cleanly.

    This is documented in code because the coursework asks for the strategy to be explicit.
    """
    df = df.copy()

    critical_fields = [
        "order_id",
        "customer_id",
        "customer_type",
        "producer_id",
        "product",
        "order_date",
        "is_reorder",
    ]

    df = df.dropna(subset=critical_fields).copy()

    for col in ["quantity", "unit_price", "month"]:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

    def month_to_season(month_value: int) -> str:
        month_value = int(month_value)
        if month_value in (12, 1, 2):
            return "winter"
        if month_value in (3, 4, 5):
            return "spring"
        if month_value in (6, 7, 8):
            return "summer"
        return "autumn"

    if df["season"].isnull().any():
        derived_season = df["month"].apply(month_to_season)
        df["season"] = df["season"].fillna(derived_season)

    if df["season"].isnull().any():
        mode_season = df["season"].mode(dropna=True)
        fill_value = mode_season.iloc[0] if not mode_season.empty else "winter"
        df["season"] = df["season"].fillna(fill_value)

    df["total_price"] = df["total_price"].fillna(df["quantity"] * df["unit_price"])
    df["total_price"] = (df["quantity"] * df["unit_price"]).round(2)

    df["quantity"] = df["quantity"].round().astype("int64")
    df["unit_price"] = df["unit_price"].astype("float64").round(2)
    df["total_price"] = df["total_price"].astype("float64").round(2)
    df["month"] = df["month"].round().astype("int64")
    df["is_reorder"] = df["is_reorder"].round().astype("int64")
    df["order_date"] = pd.to_datetime(df["order_date"], errors="raise")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add:
    - purchase_frequency: cumulative count of previous purchases for each customer-product pair
    - days_since_last_order: days since previous order for each customer-product pair
    - avg_quantity: expanding mean quantity for previous orders of that customer-product pair
    """
    df = df.copy()
    df = df.sort_values(["customer_id", "product", "order_date", "order_id"]).reset_index(drop=True)

    group_cols = ["customer_id", "product"]

    df["purchase_frequency"] = df.groupby(group_cols).cumcount()

    previous_order_date = df.groupby(group_cols)["order_date"].shift(1)
    df["days_since_last_order"] = (df["order_date"] - previous_order_date).dt.days
    df["days_since_last_order"] = df["days_since_last_order"].fillna(0).astype("float64")

    cumulative_quantity = df.groupby(group_cols)["quantity"].cumsum() - df["quantity"]
    previous_purchase_count = df["purchase_frequency"]

    df["avg_quantity"] = 0.0
    mask = previous_purchase_count > 0
    df.loc[mask, "avg_quantity"] = (
        cumulative_quantity[mask] / previous_purchase_count[mask]
    )
    df["avg_quantity"] = df["avg_quantity"].astype("float64")

    return df


def encode_categoricals(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    LabelEncode:
    - product
    - customer_type
    - season
    - producer_id

    Save encoders with joblib under task1_purchase_prediction/artifacts/
    """
    df = df.copy()
    artifacts_dir = get_artifacts_dir()

    encoders: Dict[str, LabelEncoder] = {}

    for col in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))
        encoders[col] = encoder
        joblib.dump(encoder, artifacts_dir / f"{col}_label_encoder.joblib")

    return df, encoders


def normalize_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Standardize numeric feature columns using train-only fit.
    Save scaler with joblib.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    scaler = StandardScaler()
    train_df[NUMERIC_COLUMNS_TO_SCALE] = scaler.fit_transform(train_df[NUMERIC_COLUMNS_TO_SCALE])
    test_df[NUMERIC_COLUMNS_TO_SCALE] = scaler.transform(test_df[NUMERIC_COLUMNS_TO_SCALE])

    joblib.dump(scaler, get_artifacts_dir() / "numeric_scaler.joblib")
    return train_df, test_df, scaler


def split_data(df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset using test_size from config.yaml.
    Stratify on is_reorder to preserve class balance between train and test.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df[TARGET_COLUMN],
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, test_df


def save_processed_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    processed_dir = get_processed_dir()
    train_df.to_csv(processed_dir / "train.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)


def main() -> None:
    config = load_config()
    test_size = resolve_test_size(config)

    df = load_raw_data()
    df = handle_missing(df)
    df = engineer_features(df)
    df, _ = encode_categoricals(df)

    train_df, test_df = split_data(df, test_size=test_size)
    train_df, test_df, _ = normalize_features(train_df, test_df)

    save_processed_data(train_df, test_df)

    print("Preprocessing complete.")
    print("Train shape:", train_df.shape)
    print("Test shape :", test_df.shape)
    print("Processed files saved to:", get_processed_dir())
    print("Artifacts saved to      :", get_artifacts_dir())


if __name__ == "__main__":
    main()