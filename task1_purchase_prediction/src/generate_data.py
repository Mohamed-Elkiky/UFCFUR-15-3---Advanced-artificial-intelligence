from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


SEED = 42
random.seed(SEED)
np.random.seed(SEED)


@dataclass(frozen=True)
class Customer:
    customer_id: str
    customer_type: str


@dataclass(frozen=True)
class Producer:
    producer_id: str
    producer_name: str
    category: str


@dataclass(frozen=True)
class Product:
    product: str
    base_price: float
    producer_category: str
    unit: str


CUSTOMERS: List[Customer] = [
    Customer("CUST001", "household"),
    Customer("CUST002", "household"),
    Customer("CUST003", "household"),
    Customer("CUST004", "household"),
    Customer("CUST005", "household"),
    Customer("CUST006", "restaurant"),
    Customer("CUST007", "restaurant"),
    Customer("CUST008", "retailer"),
    Customer("CUST009", "retailer"),
    Customer("CUST010", "cafe"),
    Customer("CUST011", "cafe"),
    Customer("CUST012", "hotel"),
    Customer("CUST013", "household"),
    Customer("CUST014", "restaurant"),
    Customer("CUST015", "retailer"),
]

PRODUCERS: List[Producer] = [
    Producer("PROD001", "Green Valley Farm", "farm"),
    Producer("PROD002", "Sunrise Dairy", "dairy"),
    Producer("PROD003", "Golden Crust Bakery", "bakery"),
    Producer("PROD004", "Applewood Orchard", "orchard"),
    Producer("PROD005", "Riverfresh Fishery", "fishery"),
    Producer("PROD006", "Meadow Farm Co", "farm"),
    Producer("PROD007", "Highland Dairy", "dairy"),
    Producer("PROD008", "Seasonal Orchard Ltd", "orchard"),
]

PRODUCTS: List[Product] = [
    Product("Tomatoes", 1.80, "farm", "kg"),
    Product("Potatoes", 1.20, "farm", "kg"),
    Product("Carrots", 1.10, "farm", "kg"),
    Product("Lettuce", 0.95, "farm", "unit"),
    Product("Strawberries", 3.20, "orchard", "box"),
    Product("Apples", 2.10, "orchard", "kg"),
    Product("Milk", 1.40, "dairy", "litre"),
    Product("Cheese", 3.80, "dairy", "pack"),
    Product("Yogurt", 1.25, "dairy", "pot"),
    Product("Bread", 1.50, "bakery", "loaf"),
    Product("Croissant", 1.10, "bakery", "unit"),
    Product("Salmon", 6.50, "fishery", "pack"),
]

SEASONAL_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    "Tomatoes": {"winter": 0.8, "spring": 1.0, "summer": 1.35, "autumn": 1.05},
    "Potatoes": {"winter": 1.2, "spring": 1.0, "summer": 0.9, "autumn": 1.1},
    "Carrots": {"winter": 1.1, "spring": 1.0, "summer": 0.95, "autumn": 1.15},
    "Lettuce": {"winter": 0.75, "spring": 1.05, "summer": 1.4, "autumn": 0.95},
    "Strawberries": {"winter": 0.5, "spring": 1.15, "summer": 1.8, "autumn": 0.7},
    "Apples": {"winter": 0.95, "spring": 0.85, "summer": 0.9, "autumn": 1.45},
    "Milk": {"winter": 1.1, "spring": 1.0, "summer": 0.95, "autumn": 1.0},
    "Cheese": {"winter": 1.15, "spring": 1.0, "summer": 0.9, "autumn": 1.05},
    "Yogurt": {"winter": 0.9, "spring": 1.0, "summer": 1.2, "autumn": 0.95},
    "Bread": {"winter": 1.1, "spring": 1.0, "summer": 0.95, "autumn": 1.0},
    "Croissant": {"winter": 1.0, "spring": 1.0, "summer": 1.05, "autumn": 1.0},
    "Salmon": {"winter": 1.2, "spring": 1.0, "summer": 0.9, "autumn": 1.05},
}

CUSTOMER_TYPE_ORDER_MULTIPLIER: Dict[str, float] = {
    "household": 1.0,
    "restaurant": 2.8,
    "retailer": 3.5,
    "cafe": 1.8,
    "hotel": 3.0,
}

PRODUCT_BASE_QUANTITY: Dict[str, Tuple[int, int]] = {
    "Tomatoes": (1, 8),
    "Potatoes": (1, 10),
    "Carrots": (1, 8),
    "Lettuce": (1, 6),
    "Strawberries": (1, 5),
    "Apples": (1, 8),
    "Milk": (1, 12),
    "Cheese": (1, 6),
    "Yogurt": (1, 10),
    "Bread": (1, 8),
    "Croissant": (2, 16),
    "Salmon": (1, 5),
}

OUTPUT_COLUMNS = [
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

CRITICAL_FIELDS = [
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


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_raw_data_dir() -> Path:
    raw_dir = get_project_root() / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


def month_to_season(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def random_date(start_date: datetime, end_date: datetime) -> datetime:
    delta_days = (end_date - start_date).days
    offset = random.randint(0, delta_days)
    return start_date + timedelta(days=offset)


def build_product_lookup(products: List[Product]) -> Dict[str, Product]:
    return {p.product: p for p in products}


def build_producers_by_category(producers: List[Producer]) -> Dict[str, List[Producer]]:
    grouped: Dict[str, List[Producer]] = {}
    for producer in producers:
        grouped.setdefault(producer.category, []).append(producer)
    return grouped


def choose_product(customer_type: str) -> str:
    weighted_products = {
        "household": [
            "Milk", "Bread", "Apples", "Tomatoes", "Potatoes",
            "Yogurt", "Carrots", "Lettuce", "Strawberries", "Cheese",
            "Croissant", "Salmon",
        ],
        "restaurant": [
            "Tomatoes", "Potatoes", "Carrots", "Lettuce", "Milk",
            "Cheese", "Bread", "Salmon", "Apples", "Strawberries",
            "Yogurt", "Croissant",
        ],
        "retailer": [
            "Milk", "Bread", "Apples", "Tomatoes", "Potatoes",
            "Carrots", "Yogurt", "Cheese", "Lettuce", "Strawberries",
            "Croissant", "Salmon",
        ],
        "cafe": [
            "Croissant", "Bread", "Milk", "Cheese", "Yogurt",
            "Strawberries", "Apples", "Lettuce", "Tomatoes", "Carrots",
            "Potatoes", "Salmon",
        ],
        "hotel": [
            "Milk", "Bread", "Croissant", "Cheese", "Salmon",
            "Tomatoes", "Potatoes", "Apples", "Strawberries", "Yogurt",
            "Lettuce", "Carrots",
        ],
    }
    pool = weighted_products.get(customer_type, [p.product for p in PRODUCTS])
    weights = np.linspace(len(pool), 1, len(pool))
    return random.choices(pool, weights=weights, k=1)[0]


def calculate_quantity(product_name: str, customer_type: str, season: str) -> int:
    low, high = PRODUCT_BASE_QUANTITY[product_name]
    base_qty = random.randint(low, high)
    seasonal_factor = SEASONAL_MULTIPLIERS[product_name][season]
    customer_factor = CUSTOMER_TYPE_ORDER_MULTIPLIER[customer_type]
    quantity = int(round(base_qty * seasonal_factor * customer_factor))
    return max(1, quantity)


def calculate_unit_price(base_price: float, season_multiplier: float) -> float:
    noise = random.uniform(0.93, 1.08)
    price = base_price * (0.96 + 0.08 * season_multiplier) * noise
    return round(price, 2)


def enforce_orders_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df[OUTPUT_COLUMNS]

    df["order_id"] = df["order_id"].astype("string")
    df["customer_id"] = df["customer_id"].astype("string")
    df["customer_type"] = df["customer_type"].astype("string")
    df["producer_id"] = df["producer_id"].astype("string")
    df["product"] = df["product"].astype("string")
    df["quantity"] = df["quantity"].astype("int64")
    df["unit_price"] = df["unit_price"].astype("float64").round(2)
    df["total_price"] = df["total_price"].astype("float64").round(2)
    df["order_date"] = pd.to_datetime(df["order_date"]).dt.strftime("%Y-%m-%d")
    df["order_date"] = df["order_date"].astype("string")
    df["month"] = df["month"].astype("int64")
    df["season"] = df["season"].astype("string")
    df["is_reorder"] = df["is_reorder"].astype("int64")

    return df


def validate_orders_dataframe(df: pd.DataFrame) -> None:
    missing_columns = [col for col in OUTPUT_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    null_counts = df[CRITICAL_FIELDS].isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if not null_counts.empty:
        raise ValueError(f"Nulls found in critical fields:\n{null_counts}")

    if df["order_id"].duplicated().any():
        raise ValueError("Duplicate order_id values found.")

    if (df["quantity"] <= 0).any():
        raise ValueError("quantity must be strictly positive.")

    if (df["unit_price"] <= 0).any():
        raise ValueError("unit_price must be strictly positive.")

    if (df["total_price"] <= 0).any():
        raise ValueError("total_price must be strictly positive.")

    if not df["month"].between(1, 12).all():
        raise ValueError("month must be between 1 and 12.")

    valid_seasons = {"winter", "spring", "summer", "autumn"}
    if not df["season"].isin(valid_seasons).all():
        raise ValueError("Invalid season values found.")

    if not df["is_reorder"].isin([0, 1]).all():
        raise ValueError("is_reorder must contain only 0 or 1.")

    expected_total = (df["quantity"] * df["unit_price"]).round(2)
    if not np.allclose(df["total_price"].to_numpy(), expected_total.to_numpy(), atol=1e-9):
        raise ValueError("total_price does not match quantity * unit_price.")

    pd.to_datetime(df["order_date"], format="%Y-%m-%d", errors="raise")


def generate_orders(
    num_orders: int = 2500,
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
) -> pd.DataFrame:
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    product_lookup = build_product_lookup(PRODUCTS)
    producers_by_category = build_producers_by_category(PRODUCERS)

    customer_product_history: Dict[Tuple[str, str], int] = {}
    rows = []

    for idx in range(1, num_orders + 1):
        customer = random.choice(CUSTOMERS)
        product_name = choose_product(customer.customer_type)
        product_meta = product_lookup[product_name]

        compatible_producers = producers_by_category[product_meta.producer_category]
        producer = random.choice(compatible_producers)

        order_dt = random_date(start_dt, end_dt)
        month = order_dt.month
        season = month_to_season(month)

        quantity = calculate_quantity(product_name, customer.customer_type, season)
        seasonal_multiplier = SEASONAL_MULTIPLIERS[product_name][season]
        unit_price = calculate_unit_price(product_meta.base_price, seasonal_multiplier)
        total_price = round(quantity * unit_price, 2)

        history_key = (customer.customer_id, product_name)
        is_reorder = 1 if customer_product_history.get(history_key, 0) > 0 else 0
        customer_product_history[history_key] = customer_product_history.get(history_key, 0) + 1

        rows.append(
            {
                "order_id": f"ORD{idx:06d}",
                "customer_id": customer.customer_id,
                "customer_type": customer.customer_type,
                "producer_id": producer.producer_id,
                "product": product_name,
                "quantity": quantity,
                "unit_price": unit_price,
                "total_price": total_price,
                "order_date": order_dt.strftime("%Y-%m-%d"),
                "month": month,
                "season": season,
                "is_reorder": is_reorder,
            }
        )

    orders_df = pd.DataFrame(rows).sort_values("order_date").reset_index(drop=True)
    orders_df = enforce_orders_schema(orders_df)
    validate_orders_dataframe(orders_df)
    return orders_df


def save_reference_tables(raw_dir: Path) -> None:
    customers_df = pd.DataFrame([asdict(c) for c in CUSTOMERS])
    producers_df = pd.DataFrame([asdict(p) for p in PRODUCERS])
    products_df = pd.DataFrame([asdict(p) for p in PRODUCTS])

    customers_df["customer_id"] = customers_df["customer_id"].astype("string")
    customers_df["customer_type"] = customers_df["customer_type"].astype("string")

    producers_df["producer_id"] = producers_df["producer_id"].astype("string")
    producers_df["producer_name"] = producers_df["producer_name"].astype("string")
    producers_df["category"] = producers_df["category"].astype("string")

    products_df["product"] = products_df["product"].astype("string")
    products_df["base_price"] = products_df["base_price"].astype("float64").round(2)
    products_df["producer_category"] = products_df["producer_category"].astype("string")
    products_df["unit"] = products_df["unit"].astype("string")

    customers_df.to_csv(raw_dir / "customers.csv", index=False)
    producers_df.to_csv(raw_dir / "producers.csv", index=False)
    products_df.to_csv(raw_dir / "products.csv", index=False)


def print_orders_schema(df: pd.DataFrame) -> None:
    print("\nConfirmed orders.csv schema:")
    for column in OUTPUT_COLUMNS:
        print(f"- {column}: {df[column].dtype}")

    print("\nCritical field null counts:")
    print(df[CRITICAL_FIELDS].isnull().sum())

    print("\norders.csv shape:", df.shape)
    print(df.head())


def main() -> None:
    raw_dir = get_raw_data_dir()

    orders_df = generate_orders(num_orders=2500)
    orders_df.to_csv(raw_dir / "orders.csv", index=False)

    save_reference_tables(raw_dir)
    print(f"Saved files to: {raw_dir}")
    print_orders_schema(orders_df)


if __name__ == "__main__":
    main()