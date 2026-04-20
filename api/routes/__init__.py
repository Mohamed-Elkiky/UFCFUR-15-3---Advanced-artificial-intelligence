from flask import Flask

from api.routes.quality import quality_bp
from api.routes.reorder import reorder_bp
from api.routes.model_upload import upload_bp


def register_routes(app: Flask) -> None:
    app.register_blueprint(quality_bp)
    app.register_blueprint(reorder_bp)
    app.register_blueprint(upload_bp)