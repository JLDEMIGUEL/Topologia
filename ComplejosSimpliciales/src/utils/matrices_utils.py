import numpy as np


def search_one(matrix):
    """
    Searches a one with lower indexes in the given matrix
    Args:
        matrix: np.matrix

    Returns the indexes of the one

    """
    [rows, columns] = matrix.shape
    ret = [rows - 1, columns - 1]
    for x in range(rows):
        for y in range(columns):
            if matrix[x, y] == 1 and x + y < ret[0] + ret[1]:
                ret = [x, y]
    return ret


def swap(matrix, source, obj):
    """
    Swap the row and column given in source and the ones in obj
    Args:
        matrix: np.matrix
        source: array
        obj: array

    Returns the given matrix with the applied swap

    """
    aux = matrix.copy()
    if source[0] != obj[0]:
        aux[obj[0], :] = matrix[source[0], :]
        aux[source[0], :] = matrix[obj[0], :]
    aux2 = aux.copy()
    if source[1] != obj[1]:
        aux[:, obj[1]] = aux2[:, source[1]]
        aux[:, source[1]] = aux2[:, obj[1]]
    return aux


def simplify_columns(matrix_target):
    """
    Simplifies the columns of the given matrix
    Args:
        matrix_target:  np.matrix

    Returns the simplified matrix

    """
    matrix = matrix_target.copy()
    columns = matrix.shape[1]
    for i in range(columns - 1):
        i += 1
        if matrix[0, i] == 1:
            matrix[:, i] = (matrix[:, i] + matrix[:, 0]) % 2
    return matrix


def simplify_rows(matrix_target):
    """
    Simplifies the rows of the given matrix
    Args:
        matrix_target: np.matrix

    Returns the simplified matrix

    """
    matrix = matrix_target.copy()
    rows = matrix.shape[0]
    for i in range(rows - 1):
        i += 1
        if matrix[i, 0] == 1:
            matrix[i, :] = (matrix[i, :] + matrix[0, :]) % 2
    return matrix


def reconstruct(matrix, aux):
    """
    Mixes the sub-matrix aux with the matrix
    Args:
        matrix: np.matrix
        aux: np.matrix

    Returns the reconstructed matrix

    """
    first_row = matrix[0, :]
    first_row = np.delete(first_row, 0)
    first_column = matrix[:, 0]
    aux = np.insert(aux, 0, first_row, 0)
    aux = np.concatenate([first_column, aux], 1)
    return aux


def smith_normal_form(matrix):
    """
    Smith normal form of the given matrix
    Args:
        matrix: np.matrix

    Returns the smith normal form of the given matrix

    """
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return matrix
    [x, y] = search_one(matrix)
    if matrix[x, y] != 1:
        return matrix
    if [x, y] != [0, 0]:
        matrix = swap(matrix, [x, y], [0, 0])
    matrix = simplify_columns(matrix)
    matrix = simplify_rows(matrix)
    aux = np.delete(matrix, 0, 0)
    aux = np.delete(aux, 0, 1)
    aux = smith_normal_form(aux)
    aux = reconstruct(matrix, aux)
    return aux
