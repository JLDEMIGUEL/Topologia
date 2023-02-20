from ast import literal_eval
from flask import jsonify, request
from flask_smorest import Blueprint

from SimplicialComplex.SimplicialComplex import SimplicialComplex
from app.routes.requests import SimplicialAllRequest, FacesRequest, FaceAndFacesRequest
from app.routes.responses import BasicAllResponse, FacesResponse

blp = Blueprint("Simplicial", __name__, url_prefix="/simplicial")


@blp.route("/all", methods=['POST'])
@blp.arguments(SimplicialAllRequest)
@blp.response(200, BasicAllResponse)
@blp.doc(summary="All complex attributes",
         description="Generates json with all the complex attributes.",
         requestBody={
             "content": {
                 "application/json": {
                     "example": {
                         "faces": [[1, 2, 3]],
                         "n_faces_dim": "1",
                         "skeleton_dim": "1",
                         "boundary_matrix_dim": "1",
                         "star_face": [1],
                         "closed_star_face": [1],
                         "link_face": [1]
                     }
                 }
             }
         },
         responses={
             "200": {
                 "description": "Json with all the complex attributes.",
                 "content": {
                     "application/json": {
                         "example": {
                             "bettis": [1, 0],
                             "components": 1,
                             "dimension": 2,
                             "euler": 1,
                             "faces_list": ["..."],
                             "general_boundary_matrix": ["..."],
                             "n_faces": ["..."],
                             "skeleton": ["..."],
                             "boundary_matrix": ["..."],
                             "star": ["..."],
                             "closed_star": ["..."],
                             "link": ["..."]
                         }
                     }
                 }
             }
         }
         )
def all(all_request):
    faces = all_request['faces']
    sc = SimplicialComplex(faces)
    resp = {
        "faces_list": sc.faces_list(),
        "dimension": sc.dimension(),
        "euler": sc.euler_characteristic(),
        "components": sc.connected_components(),
        "bettis": sc.all_betti_numbers(),
        "general_boundary_matrix": sc.generalized_boundary_matrix().tolist()
    }
    if (n_faces_dim := all_request.get('n_faces_dim')) is not None:
        resp["n_faces"] = sc.n_faces(n_faces_dim)
    if (skeleton_dim := all_request.get('skeleton_dim')) is not None:
        resp["skeleton"] = sc.skeleton(skeleton_dim)
    if (boundary_matrix_dim := all_request.get('boundary_matrix_dim')) is not None:
        resp["boundary_matrix"] = sc.boundary_matrix(boundary_matrix_dim).tolist()
    if (star_face := all_request.get('star_face')) is not None:
        resp["star"] = sc.star(tuple(star_face))
    if (closed_star_face := all_request.get('closed_star_face')) is not None:
        resp["closed_star"] = sc.closed_star(tuple(closed_star_face))
    if (link_face := all_request.get('link_face')) is not None:
        resp["link"] = sc.link(tuple(link_face))
    return resp


@blp.route("/faces_list", methods=['POST'])
@blp.arguments(FacesRequest)
@blp.response(200, FacesResponse)
@blp.doc(summary="Complex faces",
         description="Generates a list with the complex faces.",
         requestBody={
             "content": {
                 "application/json": {
                     "example": {
                         "faces": [[1, 2, 3]],
                     }
                 }
             }
         },
         responses={
             "200": {
                 "description": "List with the complex faces.",
                 "content": {
                     "application/json": {
                         "example": {
                             "faces": [[], [0], [1], [2], [3], "..."]
                         }
                     }
                 }
             }
         }
         )
def faces_list(faces_request):
    faces = faces_request['faces']
    sc = SimplicialComplex(faces)
    return {"faces": sc.faces_list()}


@blp.route("/dimension", methods=['POST'])
@blp.arguments(FacesRequest)
@blp.doc(summary="Complex dimension",
         description="Returns the complex dimension.",
         requestBody={
             "content": {
                 "application/json": {
                     "example": {
                         "faces": [[1, 2, 3]],
                     }
                 }
             }
         },
         responses={
             "200": {
                 "description": "Complex dimension.",
                 "content": {
                     "application/json": {
                         "example": {
                             "dimension": "2"
                         }
                     }
                 }
             }
         }
         )
def dimension(faces_request):
    faces = faces_request['faces']
    sc = SimplicialComplex(faces)
    return jsonify({"dimension": sc.dimension()}), 200


@blp.route("/n_faces/<int:dim>", methods=['POST'])
@blp.arguments(FacesRequest)
@blp.response(200, FacesResponse)
@blp.doc(summary="Complex faces of dim",
         description="Generates a list with the complex faces of the selected dimension.",
         requestBody={
             "content": {
                 "application/json": {
                     "example": {
                         "faces": [[1, 2, 3]]
                     }
                 }
             }
         },
         responses={
             "200": {
                 "description": "List with the complex faces of the selected dimension.",
                 "content": {
                     "application/json": {
                         "example": {
                             "n_faces": [[1, 2, 3]]
                         }
                     }
                 }
             }
         }
         )
def n_faces(faces_request, dim):
    faces = faces_request['faces']
    sc = SimplicialComplex(faces)
    return jsonify({"n_faces": sc.n_faces(dim)}), 200


@blp.route("/star", methods=['POST'])
@blp.arguments(FaceAndFacesRequest)
@blp.doc(summary="Complex star",
         description="Generates the star of the selected face.",
         requestBody={
             "content": {
                 "application/json": {
                     "example": {
                         "faces": [[1, 2, 3]],
                         "face": [1]
                     }
                 }
             }
         },
         responses={
             "200": {
                 "description": "Star of the selected face.",
                 "content": {
                     "application/json": {
                         "example": {
                             "star": [[1, 2, 3]]
                         }
                     }
                 }
             }
         }
         )
def star(face_and_faces_request):
    faces = face_and_faces_request['faces']
    star_face = tuple(face_and_faces_request['face'])
    sc = SimplicialComplex(faces)
    return jsonify({"star": sc.star(star_face)}), 200


@blp.route("/closed_star", methods=['POST'])
@blp.arguments(FaceAndFacesRequest)
@blp.doc(summary="Complex closed star",
         description="Generates the closed star of the selected face.",
         requestBody={
             "content": {
                 "application/json": {
                     "example": {
                         "faces": [[1, 2, 3]],
                         "face": [1]
                     }
                 }
             }
         },
         responses={
             "200": {
                 "description": "Closed star of the selected face.",
                 "content": {
                     "application/json": {
                         "example": {
                             "star": [[1, 2, 3]]
                         }
                     }
                 }
             }
         }
         )
def closed_star(face_and_faces_request):
    faces = face_and_faces_request['faces']
    closed_star_face = tuple(face_and_faces_request['face'])
    sc = SimplicialComplex(faces)
    return jsonify({"closed_star": sc.closed_star(closed_star_face)}), 200


@blp.route("/link", methods=['POST'])
@blp.arguments(FaceAndFacesRequest)
@blp.doc(summary="Complex link",
         description="Generates the link of the selected face.",
         requestBody={
             "content": {
                 "application/json": {
                     "example": {
                         "faces": [[1, 2, 3]],
                         "face": [1]
                     }
                 }
             }
         },
         responses={
             "200": {
                 "description": "Link of the selected face.",
                 "content": {
                     "application/json": {
                         "example": {
                             "link": [[1, 2, 3]]
                         }
                     }
                 }
             }
         }
         )
def link(face_and_faces_request):
    faces = face_and_faces_request['faces']
    link_face = tuple(face_and_faces_request['face'])
    sc = SimplicialComplex(faces)
    return jsonify({"link": sc.link(link_face)}), 200


@blp.route("/skeleton/<int:dim>", methods=['POST'])
@blp.arguments(FacesRequest)
@blp.doc(summary="Complex skeleton",
         description="Generates the skeleton of the selected dimension.",
         requestBody={
             "content": {
                 "application/json": {
                     "example": {
                         "faces": [[1, 2, 3]]
                     }
                 }
             }
         },
         responses={
             "200": {
                 "description": "Skeleton of the selected dimension.",
                 "content": {
                     "application/json": {
                         "example": {
                             "skeleton": [[1, 2, 3]]
                         }
                     }
                 }
             }
         }
         )
def skeleton(faces_request, dim):
    faces = faces_request['faces']
    sc = SimplicialComplex(faces)
    return jsonify({"skeleton": sc.skeleton(dim)}), 200


@blp.route("/euler_characteristic", methods=['POST'])
@blp.arguments(FacesRequest)
@blp.doc(summary="Complex euler characteristic",
         description="Compute the euler characteristic of the complex.",
         requestBody={
             "content": {
                 "application/json": {
                     "example": {
                         "faces": [[1, 2, 3]]
                     }
                 }
             }
         },
         responses={
             "200": {
                 "description": "Euler characteristic of the complex.",
                 "content": {
                     "application/json": {
                         "example": {
                             "euler_characteristic": 1
                         }
                     }
                 }
             }
         }
         )
def euler_characteristic(faces_request):
    faces = faces_request['faces']
    sc = SimplicialComplex(faces)
    return jsonify({"euler_characteristic": sc.euler_characteristic()}), 200


@blp.route("/connected_components", methods=['POST'])
@blp.arguments(FacesRequest)
@blp.doc(summary="Complex connected components",
         description="Compute the connected components of the complex.",
         requestBody={
             "content": {
                 "application/json": {
                     "example": {
                         "faces": [[1, 2, 3]]
                     }
                 }
             }
         },
         responses={
             "200": {
                 "description": "Connected components of the complex.",
                 "content": {
                     "application/json": {
                         "example": {
                             "connected_components": 1
                         }
                     }
                 }
             }
         }
         )
def connected_components(faces_request):
    faces = faces_request['faces']
    sc = SimplicialComplex(faces)
    return jsonify({"connected_components": sc.connected_components()}), 200


@blp.route("/boundary_matrix/<int:dim>", methods=['POST'])
@blp.arguments(FacesRequest)
@blp.doc(summary="Complex boundary matrix",
         description="Generates the boundary matrix for the selected dimension.",
         requestBody={
             "content": {
                 "application/json": {
                     "example": {
                         "faces": [[1, 2, 3]]
                     }
                 }
             }
         },
         responses={
             "200": {
                 "description": "Boundary matrix for the selected dimension.",
                 "content": {
                     "application/json": {
                         "example": {
                             "boundary_matrix": ["..."]
                         }
                     }
                 }
             }
         }
         )
def boundary_matrix(faces_request, dim):
    faces = faces_request['faces']
    sc = SimplicialComplex(faces)
    return jsonify({"boundary_matrix": sc.boundary_matrix(dim).tolist()}), 200


@blp.route("/generalized_boundary_matrix", methods=['POST'])
@blp.arguments(FacesRequest)
@blp.doc(summary="Complex generalized boundary matrix",
         description="Generates the generalized boundary matrix.",
         requestBody={
             "content": {
                 "application/json": {
                     "example": {
                         "faces": [[1, 2, 3]]
                     }
                 }
             }
         },
         responses={
             "200": {
                 "description": "Generalized boundary matrix.",
                 "content": {
                     "application/json": {
                         "example": {
                             "generalized_boundary_matrix": ["..."]
                         }
                     }
                 }
             }
         }
         )
def generalized_boundary_matrix(faces_request):
    faces = faces_request['faces']
    sc = SimplicialComplex(faces)
    return jsonify({"generalized_boundary_matrix": sc.generalized_boundary_matrix().tolist()}), 200


@blp.route("/betti_number/<int:dim>", methods=['POST'])
@blp.arguments(FacesRequest)
@blp.doc(summary="Complex betti number",
         description="Computes the selected betti number.",
         requestBody={
             "content": {
                 "application/json": {
                     "example": {
                         "faces": [[1, 2, 3]]
                     }
                 }
             }
         },
         responses={
             "200": {
                 "description": "Selected betti number.",
                 "content": {
                     "application/json": {
                         "example": {
                             "betti_number": 1
                         }
                     }
                 }
             }
         }
         )
def betti_number(faces_request, dim):
    faces = faces_request['faces']
    sc = SimplicialComplex(faces)
    return jsonify({"betti_number": sc.betti_number(dim)}), 200


@blp.route("/incremental_algth", methods=['POST'])
@blp.arguments(FacesRequest)
@blp.doc(summary="Complex betti number",
         description="Computes the 2 first betti numbers (plain complex's).",
         requestBody={
             "content": {
                 "application/json": {
                     "example": {
                         "faces": [[1, 2, 3]]
                     }
                 }
             }
         },
         responses={
             "200": {
                 "description": "2 first betti numbers (plain complex's).",
                 "content": {
                     "application/json": {
                         "example": {
                             "bettis": [1, 2, 3]
                         }
                     }
                 }
             }
         }
         )
def incremental_algth(faces_request):
    faces = faces_request['faces']
    sc = SimplicialComplex(faces)
    return jsonify({"bettis": sc.incremental_algth()}), 200
