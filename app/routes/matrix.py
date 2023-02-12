from fractions import Fraction

import numpy as np
from flask import jsonify, Blueprint, request, abort
from sympy import isprime

from SimplicialComplex.utils import matrices_utils

blp = Blueprint("Matrix", __name__, url_prefix="/matrix")


@blp.route("/smith_normal_form/<group>", methods=['POST'])
def smith_normal_form(group):
    req = request.get_json()
    matrix = req['matrix']
    if group.isnumeric() and isprime(int(group)):
        group = int(group)
    elif group == 'Q':
        matrix = [[Fraction(fraction) for fraction in row] for row in matrix]
    else:
        abort(500, "Group not supported, must be a prime number or q")
    smf_matrix, rows_opp_matrix, columns_opp_matrix = matrices_utils.smith_normal_form(np.array(matrix), group=group)
    if group == 'Q':
        smf_matrix = np.array([[str(fraction) for fraction in row] for row in smf_matrix])
        rows_opp_matrix = np.array([[str(fraction) for fraction in row] for row in rows_opp_matrix])
        columns_opp_matrix = np.array([[str(fraction) for fraction in row] for row in columns_opp_matrix])
    return jsonify({"matrix": smf_matrix.tolist(),
                    "rows_opp_matrix": rows_opp_matrix.tolist(),
                    "columns_opp_matrix": columns_opp_matrix.tolist()}), 200


@blp.route("/smith_normal_form_z", methods=['POST'])
def smith_normal_form_z():
    req = request.get_json()
    matrix = req['matrix']
    smf_matrix, rows_opp_matrix, columns_opp_matrix = matrices_utils.smith_normal_form_z(np.array(matrix))
    return jsonify({"matrix": smf_matrix.tolist(),
                    "rows_opp_matrix": rows_opp_matrix.tolist(),
                    "columns_opp_matrix": columns_opp_matrix.tolist()}), 200
