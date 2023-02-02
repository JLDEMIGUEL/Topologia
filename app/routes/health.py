from flask import jsonify, Blueprint


blp = Blueprint("Health", __name__)


@blp.route("/")
def get():
    return jsonify({"result": "ok"})
