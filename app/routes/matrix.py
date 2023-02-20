from fractions import Fraction

import numpy as np
from flask import abort
from flask_smorest import Blueprint
from marshmallow import Schema, fields
from sympy import isprime

from SimplicialComplex.utils import matrices_utils

blp = Blueprint("Matrix", __name__, url_prefix="/matrix")


class MatrixResponse(Schema):
    matrix = fields.List(fields.Field(), dump_only=True)
    rows_opp_matrix = fields.List(fields.Field(), dump_only=True)
    columns_opp_matrix = fields.List(fields.Field(), dump_only=True)


class MatrixRequest(Schema):
    matrix = fields.List(fields.List(fields.Field()), load_only=True)


@blp.route("/smith_normal_form/<group>", methods=['POST'])
@blp.arguments(MatrixRequest)
@blp.response(200, MatrixResponse)
@blp.doc(summary="Smith normal form", description="Calculates the Smith normal form of a matrix in a specific group.",
         requestBody={
             "content": {
                 "application/json": {
                     "example": {
                         "matrix": [
                             [1, 2],
                             [3, 4]
                         ]
                     }
                 }
             }
         },
         responses={
             "200": {
                 "description": "Returns the Smith normal form of the input matrix, along with the row and column permutation matrices.",
                 "content": {
                     "application/json": {
                         "example": {
                             "columns_opp_matrix": [
                                 [1, 3],
                                 [0, 4]
                             ],
                             "matrix": [
                                 [1, 0],
                                 [0, 1]
                             ],
                             "rows_opp_matrix": [
                                 [1, 0],
                                 [10, 4]
                             ]
                         }
                     }
                 }
             }
         }
         )
def smith_normal_form(matrix_request, group):
    matrix = matrix_request['matrix']
    if group.isnumeric() and isprime(int(group)):
        group = int(group)
    else:
        abort(500, "Group not supported, must be a prime number")

    smf_matrix, rows_opp_matrix, columns_opp_matrix = matrices_utils.smith_normal_form(np.array(matrix),
                                                                                       group=group)
    return {"matrix": smf_matrix.tolist(),
            "rows_opp_matrix": rows_opp_matrix.tolist(),
            "columns_opp_matrix": columns_opp_matrix.tolist()}


@blp.route("/smith_normal_form_q", methods=['POST'])
@blp.arguments(MatrixRequest)
@blp.response(200, MatrixResponse)
@blp.doc(summary="Smith normal form", description="Calculates the Smith normal form of a matrix with coefficients in Q",
         requestBody={
             "content": {
                 "application/json": {
                     "example": {
                         "matrix": [
                             ["1/2", "3/4"],
                             ["2/3", "4/5"]
                         ]
                     }
                 }
             }
         },
         responses={
             "200": {
                 "description": "Returns the Smith normal form of the input matrix, along with the row and column permutation matrices.",
                 "content": {
                     "application/json": {
                         "example": {
                             "columns_opp_matrix": [
                                 ["2", "10"],
                                 ["0", "-20/3"]
                             ],
                             "matrix": [
                                 ["1", "0"],
                                 ["0", "1"]
                             ],
                             "rows_opp_matrix": [
                                 ["1", "0"],
                                 ["-1", "3/4"]
                             ]
                         }
                     }
                 }
             }
         }
         )
def smith_normal_form_q(matrix_request):
    matrix = matrix_request['matrix']
    matrix = [[Fraction(fraction) for fraction in row] for row in matrix]

    smf_matrix, rows_opp_matrix, columns_opp_matrix = matrices_utils.smith_normal_form(np.array(matrix), group='Q')

    smf_matrix = np.array([[str(fraction) for fraction in row] for row in smf_matrix])
    rows_opp_matrix = np.array([[str(fraction) for fraction in row] for row in rows_opp_matrix])
    columns_opp_matrix = np.array([[str(fraction) for fraction in row] for row in columns_opp_matrix])

    return {"matrix": smf_matrix.tolist(),
            "rows_opp_matrix": rows_opp_matrix.tolist(),
            "columns_opp_matrix": columns_opp_matrix.tolist()}


@blp.route("/smith_normal_form_z", methods=['POST'])
@blp.arguments(MatrixRequest)
@blp.response(200, MatrixResponse)
@blp.doc(summary="Smith normal form in Z",
         description="Calculates the Smith normal form of a matrix with coefficients in Z.",
         requestBody={
             "content": {
                 "application/json": {
                     "example": {
                         "matrix": [
                             [1, 2],
                             [3, 4]
                         ]
                     }
                 }
             }
         },
         responses={
             "200": {
                 "description": "Returns the Smith normal form of the input matrix, along with the row and column permutation matrices.",
                 "content": {
                     "application/json": {
                         "example": {
                             "columns_opp_matrix": [
                                 [1, -2],
                                 [0, 1]
                             ],
                             "matrix": [
                                 [1, 0],
                                 [0, 2]
                             ],
                             "rows_opp_matrix": [
                                 [1, 0],
                                 [3, -1]
                             ]
                         }
                     }
                 }
             }
         }
         )
def smith_normal_form_z(matrix_request):
    matrix = matrix_request["matrix"]
    smf_matrix, rows_opp_matrix, columns_opp_matrix = matrices_utils.smith_normal_form_z(np.array(matrix))
    return {"matrix": smf_matrix.tolist(),
            "rows_opp_matrix": rows_opp_matrix.tolist(),
            "columns_opp_matrix": columns_opp_matrix.tolist()}
