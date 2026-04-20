"""App configuration loaded from environment variables."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = Path(os.getenv(
    "MODEL_DIR",
    str(REPO_ROOT / "task2_3_4_cv_quality" / "models"),
))

QUALITY_MODEL_FILE = os.getenv("QUALITY_MODEL_FILE", "resnet50_best.pth")
QUALITY_MODEL_TYPE = os.getenv("QUALITY_MODEL_TYPE", "resnet50")

REORDER_MODEL_FILE = os.getenv(
    "REORDER_MODEL_FILE",
    str(REPO_ROOT / "task1_purchase_prediction" / "models" / "reorder_model.pkl"),
)
REORDER_CLASS_NAMES_FILE = os.getenv(
    "REORDER_CLASS_NAMES_FILE",
    str(REPO_ROOT / "task1_purchase_prediction" / "models" / "class_names.pkl"),
)

UPLOAD_DIR = MODEL_DIR / "uploaded"

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))