"""Quality prediction endpoint (Task 2)."""

import base64
import hashlib
import io

import torch
from flask import Blueprint, current_app, jsonify, request
from PIL import Image
from torchvision import transforms

from api.database import log_interaction
from task2_3_4_cv_quality.src.grading import grade_produce

quality_bp = Blueprint("quality", __name__)

# Same transforms used during evaluation
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


@quality_bp.route("/predict/quality", methods=["POST"])
def predict_quality():
    """Accept a base64 image, return quality grade and scores.

    Request JSON: {"image": "<base64 string>"}
    Response JSON: {
        "predicted_class", "produce_type", "state", "confidence",
        "color_score", "size_score", "ripeness_score",
        "grade", "recommendation"
    }
    """
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "Request must include 'image' (base64)"}), 400

    # Decode image
    try:
        img_bytes = base64.b64decode(data["image"])
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    # Transform and predict
    model = current_app.config["QUALITY_MODEL"]
    class_names = current_app.config["QUALITY_CLASSES"]
    device = current_app.config["DEVICE"]

    tensor = _transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = probs.max(dim=1)

    predicted_class = class_names[pred_idx.item()]
    conf = confidence.item()

    # Run grading pipeline
    result = grade_produce({
        "predicted_class": predicted_class,
        "confidence": conf,
    })

    # Log interaction
    img_hash = hashlib.md5(img_bytes).hexdigest()[:12]
    log_interaction(
        endpoint="/predict/quality",
        prediction=predicted_class,
        confidence=conf,
        grade=result["grade"],
        input_hash=img_hash,
        metadata={"color": result["color_score"],
                  "size": result["size_score"],
                  "ripeness": result["ripeness_score"]},
    )

    return jsonify(result)