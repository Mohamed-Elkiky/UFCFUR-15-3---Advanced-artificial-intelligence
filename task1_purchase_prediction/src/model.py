from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


TARGET_COLUMN = "is_reorder"

DROP_COLUMNS = [
    "order_id",
    "customer_id",
    "order_date",
    TARGET_COLUMN,
]


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_config_path() -> Path:
    return get_project_root() / "config.yaml"


def get_processed_dir() -> Path:
    return get_project_root() / "data" / "processed"


def get_models_dir() -> Path:
    models_dir = get_project_root() / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def load_config() -> Dict:
    config_path = get_config_path()
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    return config


def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    processed_dir = get_processed_dir()
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"train.csv not found at: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"test.csv not found at: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"{TARGET_COLUMN} column not found in dataframe.")

    missing_drop_cols = [col for col in DROP_COLUMNS if col not in df.columns]
    if missing_drop_cols:
        raise ValueError(f"Expected columns missing from dataframe: {missing_drop_cols}")

    X = df.drop(columns=DROP_COLUMNS).copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


def get_model() -> RandomForestClassifier:
    """
    Returns a RandomForestClassifier using hyperparameters from config.yaml.

    Accepted config layouts:
    1)
    model:
      random_forest:
        n_estimators: 200
        max_depth: 10
        min_samples_split: 2
        min_samples_leaf: 1
        random_state: 42

    2)
    random_forest:
      n_estimators: 200
      max_depth: 10
      min_samples_split: 2
      min_samples_leaf: 1
      random_state: 42
    """
    config = load_config()

    rf_cfg = {}
    if "model" in config and isinstance(config["model"], dict):
        rf_cfg = config["model"].get("random_forest", {}) or {}
    elif "random_forest" in config and isinstance(config["random_forest"], dict):
        rf_cfg = config["random_forest"] or {}

    n_estimators = int(rf_cfg.get("n_estimators", 200))
    max_depth = rf_cfg.get("max_depth", None)
    min_samples_split = int(rf_cfg.get("min_samples_split", 2))
    min_samples_leaf = int(rf_cfg.get("min_samples_leaf", 1))
    random_state = int(rf_cfg.get("random_state", 42))

    if max_depth is not None:
        max_depth = int(max_depth)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )
    return model


def train(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    model.fit(X_train, y_train)

    model_path = get_models_dir() / "model.pkl"
    joblib.dump(model, model_path)

    return model


def load() -> RandomForestClassifier:
    model_path = get_models_dir() / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found at: {model_path}")

    return joblib.load(model_path)


def evaluate_training_accuracy(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> float:
    y_pred = model.predict(X_train)
    return accuracy_score(y_train, y_pred)


def main() -> None:
    train_df, test_df = load_processed_data()

    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)

    model = get_model()
    model = train(model, X_train, y_train)

    train_acc = evaluate_training_accuracy(model, X_train, y_train)

    print("Model training complete.")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape :", X_test.shape)
    print("y_test shape :", y_test.shape)
    print("Training accuracy:", round(train_acc, 4))
    print("Saved model to:", get_models_dir() / "model.pkl")


if __name__ == "__main__":
    main()