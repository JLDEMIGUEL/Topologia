from flask import jsonify
from flask_smorest import Blueprint

blp = Blueprint("Health", __name__)


@blp.route('/', methods=['GET'])
@blp.doc(summary="Health checker",
         description="Checks the system is up.",
         responses={
             "200": {
                 "description": "Json with all the complex attributes.",
                 "content": {
                     "application/json": {
                         "example": {
                             'status': 'OK'
                         }
                     }
                 }
             }
         }
         )
def health():
    response = {
        'status': 'OK'
    }
    return jsonify(response), 200
