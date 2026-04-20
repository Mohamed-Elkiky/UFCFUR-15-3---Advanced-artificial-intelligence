"""Request logging middleware for the Flask API."""

from flask import Flask, request


def register_logger(app: Flask) -> None:
    """Register an after_request hook that logs /predict calls."""

    @app.after_request
    def log_predict_calls(response):
        if request.path.startswith("/predict") and request.method == "POST":
            app.logger.info(
                "%s %s — %s",
                request.method,
                request.path,
                response.status,
            )
        return response