import numpy as np
from fractions import Fraction


def generalized_border_matrix(dic: dict) -> list[list[int]]:
    """
    Computes the generalized border matrix of the complex.
    Args:
        dic (dict): dictionary with faces
    Returns:
        list[list[int]]: the generalized border matrix
    """
    faces = sorted(dic.keys(), key=lambda face: (dic[face], len(face), face))
    faces.remove(faces[0])

    M = [[0 for _ in range(len(faces))] for _ in range(len(faces))]
    for i in range(len(faces)):
        for j in range(len(faces)):
            if len(faces[i]) is not len(faces[j]) - 1:
                continue
            condition = False
            for vert in faces[i]:
                if vert not in faces[j]:
                    condition = False
                    break
                condition = True
            if not condition:
                continue
            M[i][j] = 1
    return M


def smith_normal_form(matrix: np.array, rows_opp_matrix: np.array = None, columns_opp_matrix: np.array = None,
                      group: int = 2) -> tuple[np.array, np.array, np.array]:
    """
    Smith normal form of the given matrix.
    Args:
        matrix (np.array): target matrix
        rows_opp_matrix (np.array): rows matrix
        columns_opp_matrix (np.array): columns matrix
        group (int): group
    Returns:
        tuple[np.array, np.array, np.array]: smith normal form of the given matrix
    """
    matrix = matrix.copy()
    # Base case: return the matrix's
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return matrix, rows_opp_matrix, columns_opp_matrix
    # Build columns and rows matrix´s in the first iteration
    if rows_opp_matrix is None or columns_opp_matrix is None:
        columns_opp_matrix, rows_opp_matrix = _build_eye_matrix(*matrix.shape, group)
    # Search the first none zero number coordinates and return matrix´s in zero matrix case
    [x, y] = search_non_zero_elem(matrix)
    if matrix[x, y] == 0:
        return matrix, rows_opp_matrix, columns_opp_matrix
    # Swap and reduce first row and column
    matrix = swap(matrix, [x, y], [0, 0])
    rows_opp_matrix, columns_opp_matrix = swap_opp_matrix(rows_opp_matrix, columns_opp_matrix, [x, y], [0, 0])
    inv = inverse(matrix[0, 0], group)
    matrix[:, 0], columns_opp_matrix[:, 0] = inv * matrix[:, 0], inv * columns_opp_matrix[:, 0]
    if group != 'Q':
        matrix, columns_opp_matrix = matrix % group, columns_opp_matrix % group
    # Make zeros in the first row and column
    matrix, rows_opp_matrix, columns_opp_matrix = simplify_rows_and_columns(matrix, rows_opp_matrix, columns_opp_matrix,
                                                                            group)
    # Compute the smith normal form of the sub-matrix (without first row and column)
    sub_matrix, aux_rows, aux_columns = smith_normal_form(matrix[1:, 1:], rows_opp_matrix[1:, :],
                                                          columns_opp_matrix[:, 1:], group=group)
    # Re-construct all the matrix´s
    matrix, rows_opp_matrix, columns_opp_matrix = reconstruct_all(matrix, sub_matrix, rows_opp_matrix, aux_rows,
                                                                  columns_opp_matrix, aux_columns)
    return matrix, rows_opp_matrix, columns_opp_matrix


def smith_normal_form_z(matrix: np.array, rows_opp_matrix=None, columns_opp_matrix=None) -> np.array:
    """
    Smith normal form of the given matrix

    Args:
        columns_opp_matrix:
        rows_opp_matrix:
        matrix (np.array): target matrix
    Returns:
        np.array: smith normal form of the given matrix
    """
    matrix = matrix.copy()
    # Build columns and rows matrix´s in the first iteration
    if rows_opp_matrix is None or columns_opp_matrix is None:
        columns_opp_matrix, rows_opp_matrix = _build_eye_matrix(*matrix.shape)
    # Compute the gcd and return the matrix´s in case is zero
    gcd = matrix_gcd(matrix)
    if gcd == 0:
        return matrix, rows_opp_matrix, columns_opp_matrix
    # Swap and reduce until first element is the gcd
    while gcd != matrix[0, 0]:
        min_pos = min_abs_position(matrix)
        matrix, rows_opp_matrix, columns_opp_matrix = swap_and_sign(matrix, rows_opp_matrix, columns_opp_matrix,
                                                                    min_pos, [0, 0])
        if matrix[0, 0] > gcd:
            coords = _find_element_with_property(matrix)
            if coords is not None:
                matrix, rows_opp_matrix, columns_opp_matrix = _process_reduction(matrix, coords, rows_opp_matrix,
                                                                                 columns_opp_matrix)
    # Make zeros in the first row and column
    matrix, rows_opp_matrix, columns_opp_matrix = reduce_rows_columns(matrix, rows_opp_matrix, columns_opp_matrix)

    # Compute the smith normal form of the sub-matrix (without first row and column)
    aux_matrix, aux_rows, aux_columns = smith_normal_form_z(matrix[1:, 1:], rows_opp_matrix[1:, :],
                                                            columns_opp_matrix[:, 1:])

    # Re-construct all the matrix´s
    matrix, rows_opp_matrix, columns_opp_matrix = reconstruct_all(matrix, aux_matrix, rows_opp_matrix, aux_rows,
                                                                  columns_opp_matrix, aux_columns)
    return matrix, rows_opp_matrix, columns_opp_matrix


def search_non_zero_elem(matrix: np.array) -> list:
    """
    Searches a one with lower indexes in the given matrix.
    Args:
        matrix (np.array): target matrix
    Returns:
        list: indexes of the one
    """
    [rows, columns] = matrix.shape
    ret = [rows - 1, columns - 1]
    for x in range(rows):
        for y in range(columns):
            if matrix[x, y] != 0 and x + y < ret[0] + ret[1]:
                ret = [x, y]
    return ret


def swap(matrix: np.array, source: list, obj: list) -> np.array:
    """
    Swap the row and column given in source and the ones in obj.
    Args:
        matrix (np.array): target matrix
        source (list): source indexes
        obj (list): objective indexes
    Returns:
        np.array: matrix with the applied swap
    """
    aux = matrix.copy()
    if source[0] != obj[0]:
        aux[obj[0], :], aux[source[0], :] = matrix[source[0], :], matrix[obj[0], :]
    aux2 = aux.copy()
    if source[1] != obj[1]:
        aux[:, obj[1]], aux[:, source[1]] = aux2[:, source[1]], aux2[:, obj[1]]
    return aux


def swap_opp_matrix(rows_opp_matrix: np.array, columns_opp_matrix: np.array, source: list, obj: list) -> \
        tuple[np.array, np.array]:
    """
    Swap the row and column given in source and the ones in obj
    Args:
        rows_opp_matrix (np.array): rows matrix
        columns_opp_matrix (np.array): columns matrix
        source (list): source indexes
        obj (list): objective indexes
    Returns:
        tuple[np.array, np.array]: swapped matrix's
    """
    if source[0] != obj[0]:
        rows_opp_matrix = swap(rows_opp_matrix, [source[0], 0], [obj[0], 0])
    if source[1] != obj[1]:
        columns_opp_matrix = swap(columns_opp_matrix, [0, source[1]], [0, obj[1]])
    return rows_opp_matrix, columns_opp_matrix


def simplify_rows_and_columns(matrix_target: np.array, rows_opp_matrix: np.array, columns_opp_matrix: np.array,
                              group: object) -> tuple[np.array, np.array, np.array]:
    """
    Simplifies the columns of the given matrix.
    Args:
        rows_opp_matrix:
        matrix_target (np.array): target matrix
        columns_opp_matrix (np.array): columns opp matrix
        group (object): group
    Returns:
        tuple[np.array, np.array, np.array]: simplified matrix
    """
    matrix = matrix_target.copy()

    rows, columns = matrix.shape
    for i in range(1, rows):
        if matrix[i, 0] != 0:
            inv = inverse(matrix[i, 0], group)
            matrix[i, :] = (inv * matrix[i, :] - matrix[0, :])
            rows_opp_matrix[i, :] = (inv * rows_opp_matrix[i, :] - rows_opp_matrix[0, :])

    for i in range(1, columns):
        if matrix[0, i] != 0:
            inv = inverse(matrix[0, i], group)
            matrix[:, i] = (inv * matrix[:, i] - matrix[:, 0])
            columns_opp_matrix[:, i] = (inv * columns_opp_matrix[:, i] - columns_opp_matrix[:, 0])

    if group is not None and group != 'Q':
        matrix, rows_opp_matrix, columns_opp_matrix = matrix % group, rows_opp_matrix % group, columns_opp_matrix % group

    return matrix, rows_opp_matrix, columns_opp_matrix


def reduce_rows_columns(matrix: np.array, rows_opp_matrix, columns_opp_matrix) -> np.array:
    """
    Reduces the first column and first row assuming that the element [0, 0] divides
    every element in both, the first column and first row.
    Args:
        columns_opp_matrix:
        rows_opp_matrix:
        matrix (np.array): target matrix to reduce
    Returns:
        np.array: matrix with the applied reduction
    """
    matrix = matrix.copy()
    first_row = matrix[0, :]
    first_col = matrix[:, 0]
    first_elem = matrix[0, 0]
    for i in range(1, len(first_row)):
        if first_row[i] != 0:
            inv = first_row[i] / first_elem
            matrix[:, i] = matrix[:, i] - inv * matrix[:, 0]
            columns_opp_matrix[:, i] = columns_opp_matrix[:, i] - inv * columns_opp_matrix[:, 0]

    for j in range(1, len(first_col)):
        if first_col[j] != 0:
            inv = first_col[j] / first_elem
            matrix[j, :] = matrix[j, :] - inv * matrix[0, :]
            rows_opp_matrix[j, :] = rows_opp_matrix[j, :] - inv * rows_opp_matrix[0, :]

    return matrix, rows_opp_matrix, columns_opp_matrix


def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """
    Computes the extended Euclidean algorithm for integers a and b.
    Returns a tuple (gcd, x, y) such that gcd is the greatest common divisor of a and b,
    and x and y are integers satisfying the equation gcd = ax + by.
    Parameters:
        a (int): The first integer
        b (int): The second integer
    Returns:
        tuple: A tuple (gcd, x, y) where gcd is the GCD of a and b, x and y are integers
    """
    if a == 0:
        return b, 0, 1
    else:
        gcd, x, y = extended_gcd(b % a, a)
        return gcd, y - (b // a) * x, x


def reconstruct(matrix: np.array, aux: np.array) -> np.array:
    """
    Mixes the sub-matrix aux with the matrix.
    Args:
        matrix (np.array): target matrix
        aux (np.array): reconstruction matrix
    Returns:
        np.array: reconstructed matrix
    """
    if len(aux) == 0:
        return matrix
    first_row = matrix[0, 1:]
    first_col = matrix[:, 0]
    matrix_res = np.r_[[first_row], aux]
    matrix_res = np.c_[first_col, matrix_res]
    return matrix_res


def inverse(number: int | Fraction, group: int = None) -> int:
    """
    Computes the modular inverse of a number in a given group.
    The modular inverse of a is the number x such that a*x = 1 (mod group)
    If no modular inverse exists (gcd(number, group) != 1), a value of None is returned.

    Parameters:
        number (int): The number to find the modular inverse of
        group (int): The group to find the modular inverse in

    Returns:
        int: The modular inverse of number in group, or None if no inverse exists
    """
    if group is None:
        return number
    if group == 'Q':
        return Fraction(number.denominator, number.numerator)
    _, x, _ = extended_gcd(number, group)
    return x if x > 0 else x + group


def _build_eye_matrix(rows: int, cols: int, group: object = None) -> tuple[np.array, np.array]:
    """
    Compute the greatest common divisor (gcd) of two integers using the Euclidean algorithm.
    Parameters:
        rows (int): number of rows
        cols (int): number of columns
        group (object): group
    Returns:
        tuple[np.array, np.array]: columns and rows matrix's
    """
    rows_opp_matrix = np.eye(rows, dtype=int)
    columns_opp_matrix = np.eye(cols, dtype=int)
    if group == 'Q':
        rows_opp_matrix = np.array(
            [[Fraction(1) if i == j else Fraction(0) for j in range(rows)] for i in
             range(rows)])
        columns_opp_matrix = np.array(
            [[Fraction(1) if i == j else Fraction(0) for j in range(cols)] for i in
             range(cols)])
    return columns_opp_matrix, rows_opp_matrix


def gcd_euclides(a: int, b: int) -> int:
    """
    Compute the greatest common divisor (gcd) of two integers using the Euclidean algorithm.
    Parameters:
        a (int): the first integer
        b (int): the second integer
    Returns:
        int: the gcd of a and b
    """
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def matrix_gcd(matrix: np.array) -> int:
    """
    Compute the greatest common divisor (gcd) of all elements in a matrix using the Euclidean algorithm
    Parameters:
        matrix (np.array): a 2D list of integers
    Returns:
        int: the gcd of all elements in matrix
    """
    if len(matrix) == 0 or len(matrix[0]) == 0:
        return 0
    gcd_result = matrix[0][0]
    for row in matrix:
        for element in row:
            if element == 0:
                continue
            gcd_result = gcd_euclides(gcd_result % element, element)
    return gcd_result


def min_abs_position(matrix: np.array) -> list[int, int]:
    """
    Find the position of the minimum absolute value in a matrix.
    Parameters:
        matrix (np.array): A 2D list of integers
    Returns:
        list[int, int]: A tuple of integers representing the row and column indices
                         of the minimum absolute value in the matrix.
    """
    min_val = float('inf')
    min_val_pos = [0, 0]
    for i, row in enumerate(matrix):
        for j, element in enumerate(row):
            if element != 0 and abs(element) < min_val:
                min_val = abs(element)
                min_val_pos = [i, j]
    return min_val_pos


def swap_and_sign(matrix: np.array, rows_opp_matrix: np.array, columns_opp_matrix: np.array, source: list, obj: list) -> \
        tuple[np.array, np.array, np.array]:
    """
    Swap the row and column given in source and the ones in obj
    and change row sign if necessary
    Args:
        matrix (np.array): target matrix
        columns_opp_matrix (np.array): columns matrix
        rows_opp_matrix (np.array): rows matrix
        source (list): source indexes
        obj (list): objective indexes
    Returns:
        tuple[np.array,np.array, np.array]: matrix with the applied swap
    """
    matrix = swap(matrix, source, obj)
    rows_opp_matrix, columns_opp_matrix = swap_opp_matrix(rows_opp_matrix, columns_opp_matrix, source, obj)
    if matrix[obj[0], obj[1]] < 0:
        matrix[obj[0], :] *= -1
        rows_opp_matrix[obj[0], :] *= -1

    return matrix, rows_opp_matrix, columns_opp_matrix


def _find_element_with_property(matrix: np.array) -> tuple[int, int]:
    """
    Finds the first element which can not be divided by the first one.
    Args:
        matrix (np.array): matrix to find elements
    Returns:
        tuple[int, int]: coordinates of the element
    """
    rows, cols = matrix.shape
    first = matrix[0, 0]

    for i in range(1, rows):
        if gcd_euclides(first, matrix[i][0]) < first:
            return i, 0
    for j in range(1, cols):
        if gcd_euclides(first, matrix[0][j]) < first:
            return 0, j

    for i in range(1, rows):
        for j in range(1, cols):
            if gcd_euclides(first, matrix[i][j]) < first:
                return i, j


def _process_reduction(matrix: np.array, coord: tuple[int, int], rows_opp_matrix, columns_opp_matrix) -> np.array:
    """
    Returns the matrix with the reduction applied.
    Args:
        matrix (np.array): matrix to find elements
        coord (tuple[int, int]): coordinates where the element is in the matrix
    Returns:
        np.array: reduced matrix
    """
    first = matrix[0, 0]
    elem = matrix[coord[0], coord[1]]
    if coord[0] == 0:
        if matrix[coord[0], coord[1]] < 0:
            matrix[:, coord[1]] *= -1
            columns_opp_matrix[:, coord[1]] *= -1
            elem *= -1
        matrix[:, coord[1]] -= int(elem / first) * matrix[:, 0]
        columns_opp_matrix[:, coord[1]] -= int(elem / first) * columns_opp_matrix[:, 0]
    if coord[1] == 0:
        if matrix[coord[0], coord[1]] < 0:
            matrix[coord[0], :] *= -1
            rows_opp_matrix[coord[0], :] *= -1
            elem *= -1
        matrix[coord[0], :] -= int(elem / first) * matrix[0, :]
        rows_opp_matrix[coord[0], :] -= int(elem / first) * rows_opp_matrix[0, :]
    if coord[0] > 0 and coord[1] > 0:
        matrix[0, :] += matrix[coord[0], :]
        rows_opp_matrix[0, :] += rows_opp_matrix[coord[0], :]
        matrix, rows_opp_matrix, columns_opp_matrix = _process_reduction(matrix, [0, coord[1]], rows_opp_matrix,
                                                                         columns_opp_matrix)
    return matrix, rows_opp_matrix, columns_opp_matrix


def reconstruct_all(matrix: np.array, aux_matrix: np.array, rows_opp_matrix: np.array, aux_rows: np.array,
                    columns_opp_matrix: np.array, aux_columns: np.array) -> tuple[np.array, np.array, np.array]:
    """
    Mixes the sub-matrix aux with the matrix.
    Args:
        matrix (np.array): target matrix
        aux_matrix (np.array): reconstruction matrix
        rows_opp_matrix: target rows matrix
        aux_rows: reconstruction rows matrix
        columns_opp_matrix: target columns matrix
        aux_columns: reconstruction columns matrix
    Returns:
        tuple[np.array, np.array, np.array]: reconstructed matrix's
    """
    matrix = reconstruct(matrix, aux_matrix)

    if len(aux_rows) != 0:
        first_row = rows_opp_matrix[0, :]
        rows_opp_matrix = np.r_[[first_row], aux_rows]
    if len(aux_rows) != 0:
        first_col = columns_opp_matrix[:, 0]
        columns_opp_matrix = np.c_[first_col, aux_columns]

    return matrix, rows_opp_matrix, columns_opp_matrix


def generalized_border_matrix_algorithm(M: list[list[int]]) -> tuple[np.array, list]:
    """
    Reduce the generalized border matrix and computes the lows list.
    Args:
        M (list[list[int]]): dictionary with faces
    Returns:
        tuple[np.array, list]: the reduced generalized border matrix and the lows list
    """
    M = np.array(M)
    rows, cols = M.shape
    lows_list = [-1 for _ in range(cols)]
    for j in range(cols):
        column = M[:, j]
        ones_list = [x for x in range(rows) if column[x] == 1]
        if len(ones_list) == 0:
            continue
        low = max(ones_list)
        lows_list[j] = low
        prev_cols = [x for x in range(cols) if lows_list[x] == low and x != j]
        while len(prev_cols) > 0:
            prev_col = prev_cols[0]
            M[:, j] = (M[:, j] + M[:, prev_col]) % 2
            ones_list = [x for x in range(rows) if M[:, j][x] == 1]
            if len(ones_list) == 0:
                break
            low = max(ones_list)
            lows_list[j] = low
            prev_cols = [x for x in range(cols) if lows_list[x] == low and x != j]
    return M, lows_list