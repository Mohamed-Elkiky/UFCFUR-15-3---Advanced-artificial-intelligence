"""Flask application for the Bristol Regional Food Network AI service."""

import sys
from pathlib import Path

import torch
from flask import Flask, jsonify

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api.config import (
    MODEL_DIR,
    QUALITY_MODEL_FILE,
    QUALITY_MODEL_TYPE,
    REORDER_CLASS_NAMES_FILE,
    REORDER_MODEL_FILE,
)
from api.routes import register_routes


def _load_quality_model(app: Flask) -> None:
    """Load the CV quality model and class names into app.config."""
    from task2_3_4_cv_quality.src.model import get_model
    from task2_3_4_cv_quality.src.train import create_dataloaders

    device = "cuda" if torch.cuda.is_available() else "cpu"

    bundle = create_dataloaders(verbose=False)
    class_names = bundle.classes

    model = get_model(
        num_classes=len(class_names),
        model_type=QUALITY_MODEL_TYPE,
        pretrained=False,
    )
    ckpt_path = MODEL_DIR / QUALITY_MODEL_FILE
    checkpoint = torch.load(ckpt_path, map_location=device)
    state = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state)
    model.eval()

    app.config["QUALITY_MODEL"] = model
    app.config["QUALITY_CLASSES"] = class_names
    app.config["DEVICE"] = device
    print(f"✓ Quality model loaded: {QUALITY_MODEL_TYPE} from {ckpt_path.name}")


def _load_reorder_model(app: Flask) -> None:
    """Load the Task 1 reorder model into app.config."""
    import joblib

    from task1_purchase_prediction.src.predict import predict_reorder

    model_path = Path(REORDER_MODEL_FILE)
    if not model_path.exists():
        print(f"⚠  Reorder model not found at {model_path} — endpoint will error")
        app.config["REORDER_PREDICT_FN"] = lambda *a, **kw: []
        return

    app.config["REORDER_PREDICT_FN"] = predict_reorder
    print(f"✓ Reorder model loaded from {model_path.name}")


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    register_routes(app)

    with app.app_context():
        _load_quality_model(app)
        _load_reorder_model(app)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5000)