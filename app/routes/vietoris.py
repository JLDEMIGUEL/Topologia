from ast import literal_eval

from flask import jsonify, Blueprint, request, Response

from SimplicialComplex.VietorisRipsComplex import Vietoris_RipsComplex

blp = Blueprint("Vietoris", __name__, url_prefix="/vietoris")


@blp.route("/all", methods=['POST'])
def all():
    req = request.get_json()
    points = literal_eval(req['points'])
    vrc = Vietoris_RipsComplex(points, efficient=False)
    resp = {
        "faces": {str(key): value for key, value in vrc.faces.items()},
        "faces_list": vrc.faces_list(),
        "dimension": vrc.dimension(),
        "euler": vrc.euler_characteristic(),
        "components": vrc.connected_components(),
        "bettis": vrc.all_betti_numbers(),
        "general_boundary_matrix": vrc.generalized_boundary_matrix().tolist(),
        "filtration_order": vrc.filtration_order(),
        "threshold_values": vrc.threshold_values()
    }
    if (n_faces_dim := req.get('n_faces_dim')) is not None:
        resp["n_faces"] = vrc.n_faces(literal_eval(n_faces_dim))
    if (skeleton_dim := req.get('skeleton_dim')) is not None:
        resp["skeleton"] = vrc.skeleton(literal_eval(skeleton_dim))
    if (boundary_matrix_dim := req.get('boundary_matrix_dim')) is not None:
        resp["boundary_matrix"] = vrc.boundary_matrix(literal_eval(boundary_matrix_dim)).tolist()
    if (star_face := req.get('star_face')) is not None:
        resp["star"] = vrc.star(literal_eval(star_face))
    if (closed_star_face := req.get('closed_star_face')) is not None:
        resp["closed_star"] = vrc.closed_star(literal_eval(closed_star_face))
    if (link_face := req.get('link_face')) is not None:
        resp["link"] = vrc.link(literal_eval(link_face))
    return jsonify(resp), 200


@blp.route("/faces", methods=['POST'])
def faces():
    req = request.get_json()
    points = literal_eval(req['points'])
    vrc = Vietoris_RipsComplex(points)
    return jsonify({"faces": {str(key): value for key, value in vrc.faces.items()}}), 200


@blp.route("/faces_list", methods=['POST'])
def faces_list():
    req = request.get_json()
    points = literal_eval(req['points'])
    vrc = Vietoris_RipsComplex(points)
    return jsonify({"faces_list": vrc.faces_list()}), 200


@blp.route("/filtration_order", methods=['POST'])
def filtration_order():
    req = request.get_json()
    points = literal_eval(req['points'])
    vrc = Vietoris_RipsComplex(points)
    return jsonify({"filtration_order": vrc.filtration_order()}), 200


@blp.route("/threshold_values", methods=['POST'])
def threshold_values():
    req = request.get_json()
    points = literal_eval(req['points'])
    vrc = Vietoris_RipsComplex(points)
    return jsonify({"threshold_values": vrc.threshold_values()}), 200


@blp.route("/persistence_diagram", methods=['POST'])
def persistence_diagram():
    req = request.get_json()
    points = literal_eval(req['points'])
    vrc = Vietoris_RipsComplex(points)

    return Response(vrc.png_persistence_diagram(), content_type='image/png'), 200


@blp.route("/barcode_diagram", methods=['POST'])
def barcode_diagram():
    req = request.get_json()
    points = literal_eval(req['points'])
    vrc = Vietoris_RipsComplex(points)

    return Response(vrc.png_barcode_diagram(), content_type='image/png'), 200
