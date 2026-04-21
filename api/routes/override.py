"""Override endpoint — lets users correct model predictions (AA-51)."""

from flask import Blueprint, jsonify, request

from api.database import mark_overridden

override_bp = Blueprint("override", __name__)


@override_bp.route("/interaction/<int:interaction_id>/override", methods=["POST"])
def override_interaction(interaction_id: int):
    """Mark an interaction as overridden and record the corrected value.

    Overrides are batched weekly and used as additional training labels
    to keep the model aligned with real user decisions.

    Request JSON: {"override_value": "<corrected prediction>"}
    Response JSON: {"interaction_id", "override_value", "status"}
    """
    data = request.get_json(silent=True)
    if not data or "override_value" not in data:
        return jsonify({"error": "Request must include 'override_value'"}), 400

    override_value = str(data["override_value"]).strip()
    if not override_value:
        return jsonify({"error": "'override_value' must not be empty"}), 400

    updated = mark_overridden(interaction_id, override_value)
    if not updated:
        return jsonify({"error": f"Interaction {interaction_id} not found"}), 404

    return jsonify({
        "interaction_id": interaction_id,
        "override_value": override_value,
        "status": "overridden",
    })
