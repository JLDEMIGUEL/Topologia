from flask import jsonify
from flask.views import MethodView
from flask_smorest import Blueprint

blp = Blueprint("Health", __name__, description="Health check")


@blp.route("/")
def get():
    return jsonify({"result": "ok"})
