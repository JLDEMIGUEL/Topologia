from ast import literal_eval
from flask import jsonify, Blueprint, request

from SimplicialComplex.SimplicialComplex import SimplicialComplex

blp = Blueprint("Simplicial", __name__, url_prefix="/simplicial")


@blp.route("/all", methods=['POST'])
def all():
    req = request.get_json()
    faces = literal_eval(req['faces'])
    sc = SimplicialComplex(faces)
    resp = {
        "faces_list": sc.faces_list(),
        "dimension": sc.dimension(),
        "euler": sc.euler_characteristic(),
        "components": sc.connected_components(),
        "bettis": sc.all_betti_numbers(),
        "general_boundary_matrix": sc.generalized_boundary_matrix().tolist()
    }
    if (n_faces_dim := req.get('n_faces_dim')) is not None:
        resp["n_faces"] = sc.n_faces(literal_eval(n_faces_dim))
    if (skeleton_dim := req.get('skeleton_dim')) is not None:
        resp["skeleton"] = sc.skeleton(literal_eval(skeleton_dim))
    if (boundary_matrix_dim := req.get('boundary_matrix_dim')) is not None:
        resp["boundary_matrix"] = sc.boundary_matrix(literal_eval(boundary_matrix_dim)).tolist()
    if (star_face := req.get('star_face')) is not None:
        resp["star"] = sc.star(literal_eval(star_face))
    if (closed_star_face := req.get('closed_star_face')) is not None:
        resp["closed_star"] = sc.closed_star(literal_eval(closed_star_face))
    if (link_face := req.get('link_face')) is not None:
        resp["link"] = sc.link(literal_eval(link_face))
    return jsonify(resp), 200


@blp.route("/faces_list", methods=['POST'])
def faces_list():
    req = request.get_json()
    faces = literal_eval(req['faces'])
    sc = SimplicialComplex(faces)
    return jsonify({"faces_list": sc.faces_list()}), 200


@blp.route("/dimension", methods=['POST'])
def dimension():
    req = request.get_json()
    faces = literal_eval(req['faces'])
    sc = SimplicialComplex(faces)
    return jsonify({"dimension": sc.dimension()}), 200


@blp.route("/n_faces/<int:dim>", methods=['POST'])
def n_faces(dim):
    req = request.get_json()
    faces = literal_eval(req['faces'])
    sc = SimplicialComplex(faces)
    return jsonify({"n_faces": sc.n_faces(dim)}), 200


@blp.route("/star", methods=['POST'])
def star():
    req = request.get_json()
    faces = literal_eval(req['faces'])
    star_face = literal_eval(req['star_face'])
    sc = SimplicialComplex(faces)
    return jsonify({"star": sc.star(star_face)}), 200


@blp.route("/closed_star", methods=['POST'])
def closed_star():
    req = request.get_json()
    faces = literal_eval(req['faces'])
    closed_star_face = literal_eval(req['closed_star_face'])
    sc = SimplicialComplex(faces)
    return jsonify({"closed_star": sc.closed_star(closed_star_face)}), 200


@blp.route("/link", methods=['POST'])
def link():
    req = request.get_json()
    faces = literal_eval(req['faces'])
    link_face = literal_eval(req['link_face'])
    sc = SimplicialComplex(faces)
    return jsonify({"link": sc.link(link_face)}), 200


@blp.route("/skeleton/<int:dim>", methods=['POST'])
def skeleton(dim):
    req = request.get_json()
    faces = literal_eval(req['faces'])
    sc = SimplicialComplex(faces)
    return jsonify({"skeleton": sc.skeleton(dim)}), 200


@blp.route("/euler_characteristic", methods=['POST'])
def euler_characteristic():
    req = request.get_json()
    faces = literal_eval(req['faces'])
    sc = SimplicialComplex(faces)
    return jsonify({"euler_characteristic": sc.euler_characteristic()}), 200


@blp.route("/connected_components", methods=['POST'])
def connected_components():
    req = request.get_json()
    faces = literal_eval(req['faces'])
    sc = SimplicialComplex(faces)
    return jsonify({"connected_components": sc.connected_components()}), 200


@blp.route("/boundary_matrix/<int:dim>", methods=['POST'])
def boundary_matrix(dim):
    req = request.get_json()
    faces = literal_eval(req['faces'])
    sc = SimplicialComplex(faces)
    return jsonify({"boundary_matrix": sc.boundary_matrix(dim).tolist()}), 200


@blp.route("/generalized_boundary_matrix", methods=['POST'])
def generalized_boundary_matrix():
    req = request.get_json()
    faces = literal_eval(req['faces'])
    sc = SimplicialComplex(faces)
    return jsonify({"generalized_boundary_matrix": sc.generalized_boundary_matrix().tolist()}), 200


@blp.route("/betti_number/<int:dim>", methods=['POST'])
def betti_number(dim):
    req = request.get_json()
    faces = literal_eval(req['faces'])
    sc = SimplicialComplex(faces)
    return jsonify({"betti_number": sc.betti_number(dim)}), 200


@blp.route("/incremental_algth", methods=['POST'])
def incremental_algth():
    req = request.get_json()
    faces = literal_eval(req['faces'])
    sc = SimplicialComplex(faces)
    return jsonify({"bettis": sc.incremental_algth()}), 200
