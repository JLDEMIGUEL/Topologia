import os
from ast import literal_eval

from flask import Blueprint, request, Response, abort

from SimplicialComplex.AlphaComplex import AlphaComplex

blp = Blueprint("Alpha", __name__, url_prefix="/alpha")


@blp.route("/gif", methods=['POST'])
def faces_list():
    req = request.get_json()
    points = literal_eval(req['points'])
    ac = AlphaComplex(points)
    filename = None
    try:
        filename = ac.gif_alpha()
        with open(filename, 'rb') as f:
            file_content = f.read()
        response = Response(file_content, content_type="image/gif")
        return response
    except Exception:
        # Handle exceptions here
        abort(500, "An error occurred when generating the gif")
    finally:
        if os.path.exists(filename):
            os.remove( filename)
