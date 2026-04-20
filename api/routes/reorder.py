"""Reorder prediction endpoint (Task 1)."""

from flask import Blueprint, current_app, jsonify, request

from api.database import log_interaction

reorder_bp = Blueprint("reorder", __name__)


@reorder_bp.route("/predict/reorder", methods=["POST"])
def predict_reorder():
    """Predict next product for a customer based on purchase history.

    Request JSON: {"customer_id": "CUST001"}
    Response JSON: {"customer_id", "suggestions": [{"product", "score"}, ...]}
    """
    data = request.get_json(silent=True)
    if not data or "customer_id" not in data:
        return jsonify({"error": "Request must include 'customer_id'"}), 400

    customer_id = str(data["customer_id"])
    top_n = int(data.get("top_n", 5))

    predict_fn = current_app.config["REORDER_PREDICT_FN"]

    try:
        suggestions = predict_fn(customer_id, top_n=top_n)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    # Log interaction
    top_product = suggestions[0]["product"] if suggestions else "none"
    log_interaction(
        endpoint="/predict/reorder",
        prediction=top_product,
        metadata={"customer_id": customer_id, "n_suggestions": len(suggestions)},
    )

    return jsonify({
        "customer_id": customer_id,
        "suggestions": suggestions,
    })