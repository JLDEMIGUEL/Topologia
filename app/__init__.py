from flask import Flask
from app.routes.health import blp as HealthBlueprint


def create_app():
    # Initialize flask app
    app = Flask(__name__)

    # Register routes blueprints
    app.register_blueprint(HealthBlueprint)

    return app
