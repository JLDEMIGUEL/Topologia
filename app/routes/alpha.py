import os
from ast import literal_eval

from flask import Response, abort
from flask_smorest import Blueprint

from SimplicialComplex.AlphaComplex import AlphaComplex
from app.routes.requests import PointsRequest, ComplexAllRequest
from app.routes.responses import ThresholdValuesResponse, FacesResponse, FacesDictResponse, ComplexAllResponse

blp = Blueprint("Alpha", __name__, url_prefix="/alpha")


@blp.route("/all", methods=['POST'])
@blp.arguments(ComplexAllRequest)
@blp.response(200, ComplexAllResponse)
@blp.doc(summary="All complex attributes",
         description="Generates json with all the complex attributes.",
         requestBody={
             "content": {
                 "application/json": {
                     "example": {
                         "points": [[-2, 2], [6, 4], [3, -2], [-1, -5], [1.7, -1.8], [5, -5], [4, 7], [3, 5]],
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
                             "link": ["..."],
                             "filtration_order": ["..."],
                             "threshold_values": ["..."],
                         }
                     }
                 }
             }
         }
         )
def all(all_request):
    points = all_request['points']
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
    if (n_faces_dim := all_request.get('n_faces_dim')) is not None:
        resp["n_faces"] = ac.n_faces(n_faces_dim)
    if (skeleton_dim := all_request.get('skeleton_dim')) is not None:
        resp["skeleton"] = ac.skeleton(skeleton_dim)
    if (boundary_matrix_dim := all_request.get('boundary_matrix_dim')) is not None:
        resp["boundary_matrix"] = ac.boundary_matrix(boundary_matrix_dim).tolist()
    if (star_face := all_request.get('star_face')) is not None:
        resp["star"] = ac.star(tuple(star_face))
    if (closed_star_face := all_request.get('closed_star_face')) is not None:
        resp["closed_star"] = ac.closed_star(tuple(closed_star_face))
    if (link_face := all_request.get('link_face')) is not None:
        resp["link"] = ac.link(tuple(link_face))
    return resp


@blp.route("/faces", methods=['POST'])
@blp.arguments(PointsRequest)
@blp.response(200, FacesDictResponse)
@blp.doc(summary="Complex faces and values",
         description="Generates json with the faces and its associated values.",
         requestBody={
             "content": {
                 "application/json": {
                     "example": {
                         "points": [[-2, 2], [6, 4], [3, -2], [-1, -5], [1.7, -1.8], [5, -5]]
                     }
                 }
             }
         },
         responses={
             "200": {
                 "description": "Json with the faces and its associated values.",
                 "content": {
                     "application/json": {
                         "example": {
                             "faces": {"()": 0, "(0, 11)": 2.236067977499791, "(0, 3)": 3.552536567345182,
                                       "(0, 3, 4)": "..."}
                         }
                     }
                 }
             }
         }
         )
def faces(points_request):
    points = points_request['points']
    ac = AlphaComplex(points)
    return {"faces": {str(key): value for key, value in ac.faces.items()}}


@blp.route("/faces_list", methods=['POST'])
@blp.arguments(PointsRequest)
@blp.response(200, FacesResponse)
@blp.doc(summary="Complex faces",
         description="Generates a list with the complex faces.",
         requestBody={
             "content": {
                 "application/json": {
                     "example": {
                         "points": [[-2, 2], [6, 4], [3, -2], [-1, -5], [1.7, -1.8], [5, -5]]
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
                             "faces": [[], [0], [1], [2], [3], [4], "..."]
                         }
                     }
                 }
             }
         }
         )
def faces_list(points_request):
    points = points_request['points']
    ac = AlphaComplex(points)
    return {"faces": ac.faces_list()}


@blp.route("/filtration_order", methods=['POST'])
@blp.arguments(PointsRequest)
@blp.response(200, FacesResponse)
@blp.doc(summary="Complex filtration in order",
         description="Generates a list with the complex faces sorted by its associated float.",
         requestBody={
             "content": {
                 "application/json": {
                     "example": {
                         "points": [[-2, 2], [6, 4], [3, -2], [-1, -5], [1.7, -1.8], [5, -5]]
                     }
                 }
             }
         },
         responses={
             "200": {
                 "description": "List with the complex faces sorted by its associated float.",
                 "content": {
                     "application/json": {
                         "example": {
                             "faces": [[], [0], [1], [2], [3], [4], "..."]
                         }
                     }
                 }
             }
         }
         )
def filtration_order(points_request):
    points = points_request['points']
    ac = AlphaComplex(points)
    return {"faces": ac.filtration_order()}


@blp.route("/threshold_values", methods=['POST'])
@blp.arguments(PointsRequest)
@blp.response(200, ThresholdValuesResponse)
@blp.doc(summary="Complex threshold values",
         description="Generates a list with the threshold values of the complex.",
         requestBody={
             "content": {
                 "application/json": {
                     "example": {
                         "points": [[-2, 2], [6, 4], [3, -2], [-1, -5], [1.7, -1.8], [5, -5]]
                     }
                 }
             }
         },
         responses={
             "200": {
                 "description": "List with the threshold values of the complex.",
                 "content": {
                     "application/json": {
                         "example": {
                             "threshold_values": [0, 0.6576473218982953, 1.8027756377319946, 2.0934421415458324, "..."]
                         }
                     }
                 }
             }
         }
         )
def threshold_values(points_request):
    points = points_request['points']
    ac = AlphaComplex(points)
    return {"threshold_values": ac.threshold_values()}


@blp.route("/persistence_diagram", methods=['POST'])
@blp.arguments(PointsRequest)
@blp.response(200, {"format": "binary", "type": "string"}, content_type="image/png")
@blp.doc(summary="Persistence Diagram image",
         description="Generates a Persistence Diagram image from the given points.",
         requestBody={
             "content": {
                 "application/json": {
                     "example": {
                         "points": [[-2, 2], [6, 4], [3, -2], [-1, -5], [1.7, -1.8], [5, -5]]
                     }
                 }
             }
         },
         responses={
             "200": {
                 "description": "Persistence Diagram image from the given points.",
                 "content": {
                     "image/gif": {}
                 }
             }
         }
         )
def persistence_diagram(points_request):
    points = points_request['points']
    ac = AlphaComplex(points)

    return Response(ac.png_persistence_diagram(), content_type='image/png'), 200


@blp.route("/barcode_diagram", methods=['POST'])
@blp.arguments(PointsRequest)
@blp.response(200, {"format": "binary", "type": "string"}, content_type="image/png")
@blp.doc(summary="Barcode Diagram image",
         description="Generates a Barcode Diagram image from the given points.",
         requestBody={
             "content": {
                 "application/json": {
                     "example": {
                         "points": [[-2, 2], [6, 4], [3, -2], [-1, -5], [1.7, -1.8], [5, -5]]
                     }
                 }
             }
         },
         responses={
             "200": {
                 "description": "Barcode Diagram image from the given points.",
                 "content": {
                     "image/gif": {}
                 }
             }
         }
         )
def barcode_diagram(points_request):
    points = points_request['points']
    ac = AlphaComplex(points)

    return Response(ac.png_barcode_diagram(), content_type='image/png'), 200


@blp.route("/gif", methods=['POST'])
@blp.arguments(PointsRequest)
@blp.response(200, {"format": "binary", "type": "string"}, content_type="image/gif")
@blp.doc(summary="Alpha Complex GIF",
         description="Generates an Alpha Complex GIF from the given points showing its construction.",
         requestBody={
             "content": {
                 "application/json": {
                     "example": {
                         "points": [[-2, 2], [6, 4], [3, -2], [-1, -5], [1.7, -1.8], [5, -5]]
                     }
                 }
             }
         },
         responses={
             "200": {
                 "description": "Alpha Complex GIF from the given points showing its construction.",
                 "content": {
                     "image/gif": {}
                 }
             }
         }
         )
def gif_alpha(points_request):
    points = points_request['points']
    ac = AlphaComplex(points)
    try:
        file_content = ac.gif_alpha()
        response = Response(file_content, content_type="image/gif")
        return response
    except Exception as e:
        # Handle exceptions here
        abort(500, "An error occurred when generating the gif")
