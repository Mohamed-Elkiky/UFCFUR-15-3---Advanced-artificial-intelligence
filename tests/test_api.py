"""Integration tests for the Flask API endpoints."""

import base64
import io
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api.app import Flask
from api.routes import register_routes


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app():
    """Create a test Flask app with mocked models."""
    app = Flask(__name__)
    app.config["TESTING"] = True

    # Mock quality model: returns logits for 28 classes
    mock_model = MagicMock()
    mock_logits = torch.zeros(1, 28)
    mock_logits[0, 0] = 10.0  # High confidence for class 0
    mock_model.return_value = mock_logits
    mock_model.eval = MagicMock()

    app.config["QUALITY_MODEL"] = mock_model
    app.config["QUALITY_CLASSES"] = [
        "Apple__Healthy", "Apple__Rotten",
        "Banana__Healthy", "Banana__Rotten",
        "Bellpepper__Healthy", "Bellpepper__Rotten",
        "Carrot__Healthy", "Carrot__Rotten",
        "Cucumber__Healthy", "Cucumber__Rotten",
        "Grape__Healthy", "Grape__Rotten",
        "Guava__Healthy", "Guava__Rotten",
        "Jujube__Healthy", "Jujube__Rotten",
        "Mango__Healthy", "Mango__Rotten",
        "Orange__Healthy", "Orange__Rotten",
        "Pomegranate__Healthy", "Pomegranate__Rotten",
        "Potato__Healthy", "Potato__Rotten",
        "Strawberry__Healthy", "Strawberry__Rotten",
        "Tomato__Healthy", "Tomato__Rotten",
    ]
    app.config["DEVICE"] = "cpu"

    # Mock reorder prediction function
    app.config["REORDER_PREDICT_FN"] = lambda cid, top_n=5: [
        {"product": "Tomatoes", "score": 0.8},
        {"product": "Apples", "score": 0.6},
    ]

    # Health endpoint
    from flask import jsonify
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    register_routes(app)

    return app


@pytest.fixture
def client(app):
    return app.test_client()


def _make_dummy_image_b64() -> str:
    """Create a small dummy PNG image as a base64 string."""
    img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_predict_quality_returns_grade(client):
    resp = client.post(
        "/predict/quality",
        json={"image": _make_dummy_image_b64()},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert "grade" in data
    assert data["grade"] in ("A", "B", "C")
    assert "color_score" in data
    assert "size_score" in data
    assert "ripeness_score" in data
    assert "recommendation" in data


def test_predict_quality_missing_image_returns_400(client):
    resp = client.post("/predict/quality", json={})
    assert resp.status_code == 400


def test_predict_quality_invalid_image_returns_400(client):
    resp = client.post(
        "/predict/quality",
        json={"image": "not_valid_base64!!!"},
    )
    assert resp.status_code == 400


def test_predict_reorder_returns_suggestions(client):
    resp = client.post(
        "/predict/reorder",
        json={"customer_id": "CUST001"},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert "suggestions" in data
    assert isinstance(data["suggestions"], list)
    assert len(data["suggestions"]) > 0
    assert "product" in data["suggestions"][0]


def test_predict_reorder_missing_customer_returns_400(client):
    resp = client.post("/predict/reorder", json={})
    assert resp.status_code == 400


def test_model_upload_saves_file(client):
    dummy_data = b"fake model weights"
    resp = client.post(
        "/model/upload",
        data={"file": (io.BytesIO(dummy_data), "test_model.pt")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 201
    data = resp.get_json()
    assert "filename" in data
    assert data["filename"].endswith(".pt")


def test_model_upload_rejects_bad_extension(client):
    resp = client.post(
        "/model/upload",
        data={"file": (io.BytesIO(b"data"), "model.exe")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400


def test_model_upload_no_file_returns_400(client):
    resp = client.post("/model/upload")
    assert resp.status_code == 400


def test_admin_interactions_returns_list(client):
    resp = client.get("/admin/interactions")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "interactions" in data
    assert isinstance(data["interactions"], list)