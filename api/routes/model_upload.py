"""Model upload and admin interaction log endpoints (Task 3)."""

from datetime import datetime
from pathlib import Path

from flask import Blueprint, jsonify, request

from api.config import MAX_UPLOAD_MB, UPLOAD_DIR
from api.database import get_all_interactions

upload_bp = Blueprint("upload", __name__)

ALLOWED_EXTENSIONS = {".pt", ".pth", ".pkl"}


@upload_bp.route("/model/upload", methods=["POST"])
def upload_model():
    """Upload a new model file for deployment.

    Accepts multipart file upload. Validates extension and size.
    Saves to models/uploaded/ with a timestamp in the filename.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Validate extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": f"Invalid extension '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        }), 400

    # Validate size
    file.seek(0, 2)
    size_mb = file.tell() / (1024 * 1024)
    file.seek(0)

    if size_mb > MAX_UPLOAD_MB:
        return jsonify({
            "error": f"File too large ({size_mb:.1f} MB). Max: {MAX_UPLOAD_MB} MB"
        }), 400

    # Save with timestamp
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = Path(file.filename).stem.replace(" ", "_")
    dest_name = f"{safe_name}_{timestamp}{ext}"
    dest_path = UPLOAD_DIR / dest_name

    file.save(str(dest_path))

    return jsonify({
        "message": "Model uploaded successfully",
        "filename": dest_name,
        "size_mb": round(size_mb, 2),
        "path": str(dest_path),
    }), 201


@upload_bp.route("/admin/interactions", methods=["GET"])
def admin_interactions():
    """Return all logged prediction interactions as JSON."""
    interactions = get_all_interactions()
    return jsonify({
        "count": len(interactions),
        "interactions": interactions,
    })