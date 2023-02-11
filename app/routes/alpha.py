import os
from ast import literal_eval
from io import BytesIO

import matplotlib.pyplot as plt
from flask import Blueprint, request, Response, abort, jsonify

from SimplicialComplex.AlphaComplex import AlphaComplex

blp = Blueprint("Alpha", __name__, url_prefix="/alpha")


@blp.route("/all", methods=['POST'])
def all():
    req = request.get_json()
    points = literal_eval(req['points'])
    ac = AlphaComplex(points)
    resp = {
        "faces": {str(key): value for key, value in ac.faces.items()},
        "faces_list": ac.faces_list(),
        "dimension": ac.dimension(),
        "euler": ac.euler_characteristic(),
        "components": ac.connected_components(),
        "bettis": ac.all_betti_numbers(),
        "general_boundary_matrix": ac.generalized_boundary_matrix().tolist(),
        "filtration_order": ac.filtration_order(),
        "threshold_values": ac.threshold_values()
    }
    if (n_faces_dim := req.get('n_faces_dim')) is not None:
        resp["n_faces"] = ac.n_faces(literal_eval(n_faces_dim))
    if (skeleton_dim := req.get('skeleton_dim')) is not None:
        resp["skeleton"] = ac.skeleton(literal_eval(skeleton_dim))
    if (boundary_matrix_dim := req.get('boundary_matrix_dim')) is not None:
        resp["boundary_matrix"] = ac.boundary_matrix(literal_eval(boundary_matrix_dim)).tolist()
    if (star_face := req.get('star_face')) is not None:
        resp["star"] = ac.star(literal_eval(star_face))
    if (closed_star_face := req.get('closed_star_face')) is not None:
        resp["closed_star"] = ac.closed_star(literal_eval(closed_star_face))
    if (link_face := req.get('link_face')) is not None:
        resp["link"] = ac.link(literal_eval(link_face))
    return jsonify(resp), 200


@blp.route("/faces", methods=['POST'])
def faces():
    req = request.get_json()
    points = literal_eval(req['points'])
    ac = AlphaComplex(points)
    return jsonify({"faces": {str(key): value for key, value in ac.faces.items()}}), 200


@blp.route("/faces_list", methods=['POST'])
def faces_list():
    req = request.get_json()
    points = literal_eval(req['points'])
    ac = AlphaComplex(points)
    return jsonify({"faces_list": ac.faces_list()}), 200


@blp.route("/filtration_order", methods=['POST'])
def filtration_order():
    req = request.get_json()
    points = literal_eval(req['points'])
    ac = AlphaComplex(points)
    return jsonify({"filtration_order": ac.filtration_order()}), 200


@blp.route("/threshold_values", methods=['POST'])
def threshold_values():
    req = request.get_json()
    points = literal_eval(req['points'])
    ac = AlphaComplex(points)
    return jsonify({"threshold_values": ac.threshold_values()}), 200


@blp.route("/persistence_diagram", methods=['POST'])
def persistence_diagram():
    req = request.get_json()
    points = literal_eval(req['points'])
    ac = AlphaComplex(points)

    return Response(ac.png_persistence_diagram(), content_type='image/png'), 200


@blp.route("/barcode_diagram", methods=['POST'])
def barcode_diagram():
    req = request.get_json()
    points = literal_eval(req['points'])
    ac = AlphaComplex(points)

    return Response(ac.png_barcode_diagram(), content_type='image/png'), 200


@blp.route("/gif", methods=['POST'])
def gif_alpha():
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
            os.remove(filename)
