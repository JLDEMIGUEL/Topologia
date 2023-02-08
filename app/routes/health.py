from flask import jsonify, Blueprint


blp = Blueprint("Health", __name__)


@blp.route('/', methods=['GET'])
def health():
    response = {
        'status': 'OK'
    }
    return jsonify(response), 200
