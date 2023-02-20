from flask import Flask
from flask_smorest import Api
from app.routes.health import blp as HealthBlueprint
from app.routes.alpha import blp as AlphaBlueprint
from app.routes.matrix import blp as MatrixBlueprint
from app.routes.simplicial import blp as SimplicialBlueprint
from app.routes.vietoris import blp as VietorisBlueprint


def create_app():
    # Initialize flask app
    app = Flask(__name__)
    app.config["API_TITLE"] = "Simplicial Complex REST API"
    app.config["API_VERSION"] = "v1"
    app.config["OPENAPI_VERSION"] = "3.0.3"
    app.config["OPENAPI_URL_PREFIX"] = "/"
    app.config["OPENAPI_SWAGGER_UI_PATH"] = "/swagger-ui"
    app.config["OPENAPI_SWAGGER_UI_URL"] = "https://cdn.jsdelivr.net/npm/swagger-ui-dist/"
    api = Api(app)

    # Register routes blueprints
    api.register_blueprint(HealthBlueprint)
    api.register_blueprint(AlphaBlueprint)
    api.register_blueprint(MatrixBlueprint)
    api.register_blueprint(SimplicialBlueprint)
    api.register_blueprint(VietorisBlueprint)

    return app
