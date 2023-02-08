from flask import Flask
from app.routes.health import blp as HealthBlueprint
from app.routes.alpha import blp as AlphaBlueprint
from app.routes.homology import blp as HomologyBlueprint
from app.routes.matrix import blp as MatrixBlueprint
from app.routes.persistence import blp as PersistenceBlueprint
from app.routes.simplicial import blp as SimplicialBlueprint
from app.routes.vietoris import blp as VietorisBlueprint


def create_app():
    # Initialize flask app
    app = Flask(__name__)

    # Register routes blueprints
    app.register_blueprint(HealthBlueprint)
    app.register_blueprint(AlphaBlueprint)
    app.register_blueprint(HomologyBlueprint)
    app.register_blueprint(MatrixBlueprint)
    app.register_blueprint(PersistenceBlueprint)
    app.register_blueprint(SimplicialBlueprint)
    app.register_blueprint(VietorisBlueprint)

    return app
