import numpy as np


def search_one(matrix: np.matrix) -> list:
    """
    Searches a one with lower indexes in the given matrix

    Args:
        matrix (np.matrix): target matrix

    Returns:
        list: indexes of the one

    """
    [rows, columns] = matrix.shape
    ret = [rows - 1, columns - 1]
    for x in range(rows):
        for y in range(columns):
            if matrix[x, y] == 1 and x + y < ret[0] + ret[1]:
                ret = [x, y]
    return ret


def swap(matrix: np.matrix, source: list, obj: list) -> np.matrix:
    """
    Swap the row and column given in source and the ones in obj

    Args:
        matrix (np.matrix): target matrix
        source (list): source indexes
        obj (list): objective indexes
    Returns:
        np.matrix: matrix with the applied swap

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


def simplify_columns(matrix_target: np.matrix) -> np.matrix:
    """
    Simplifies the columns of the given matrix

    Args:
        matrix_target (np.matrix): target matrix
    Returns:
        np.matrix: simplified matrix

    """
    matrix = matrix_target.copy()
    columns = matrix.shape[1]
    for i in range(columns - 1):
        i += 1
        if matrix[0, i] == 1:
            matrix[:, i] = (matrix[:, i] + matrix[:, 0]) % 2
    return matrix


def simplify_rows(matrix_target: np.matrix) -> np.matrix:
    """
    Simplifies the rows of the given matrix

    Args:
        matrix_target (np.matrix): target matrix
    Returns:
        np.matrix: simplified matrix
    """
    matrix = matrix_target.copy()
    rows = matrix.shape[0]
    for i in range(rows - 1):
        i += 1
        if matrix[i, 0] == 1:
            matrix[i, :] = (matrix[i, :] + matrix[0, :]) % 2
    return matrix


def reconstruct(matrix: np.matrix, aux: np.matrix) -> np.matrix:
    """
    Mixes the sub-matrix aux with the matrix

    Args:
        matrix (np.matrix): target matrix
        aux (np.matrix): reconstruction matrix
    Returns:
        np.matrix: reconstructed matrix

    """
    first_row = matrix[0, :]
    first_row = np.delete(first_row, 0)
    first_column = matrix[:, 0]
    aux = np.insert(aux, 0, first_row, 0)
    aux = np.concatenate([first_column, aux], 1)
    return aux


def smith_normal_form(matrix: np.matrix) -> np.matrix:
    """
    Smith normal form of the given matrix

    Args:
        matrix (np.matrix): target matrix
    Returns:
        np.matrix: smith normal form of the given matrix
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


def gcd_euclides(a, b):
    pass


def matrix_gcd(matrix):
    pass


def smith_normal_form_z(matrix: np.matrix) -> np.matrix:
    """
    Smith normal form of the given matrix

    Args:
        matrix (np.matrix): target matrix
    Returns:
        np.matrix: smith normal form of the given matrix
    """

    pass


def algoritmo_matriz(M):
    M = np.matrix(M)
    rows, cols = M.shape
    lows_list = [-1 for _ in range(cols)]
    for j in range(cols):
        columna = M[:, j]
        lista = [x for x in range(rows) if columna[x] == 1]
        if len(lista) == 0: continue
        low = max(lista)
        lows_list[j] = low
        prev_cols = [x for x in range(cols) if lows_list[x] == low and x != j]
        while len(prev_cols) > 0:
            prev_col = prev_cols[0]
            M[:, j] = (M[:, j] + M[:, prev_col]) % 2
            lista = [x for x in range(rows) if M[:, j][x] == 1]
            if len(lista) == 0: break
            low = max(lista)
            lows_list[j] = low
            prev_cols = [x for x in range(cols) if lows_list[x] == low and x != j]
    return M, lows_list


def matriz_borde_generalizado(dic):
    faces = sorted(dic.keys(), key=lambda face: (dic[face], len(face), face))
    faces.remove(faces[0])

    M = [[0 for _ in range(len(faces))] for _ in range(len(faces))]
    for i in range(len(faces)):
        for j in range(len(faces)):
            if len(faces[i]) is not len(faces[j]) - 1:
                continue
            bool = False
            for vert in faces[i]:
                if vert not in faces[j]:
                    bool = False
                    break
                bool = True
            if not bool: continue
            M[i][j] = 1
    return M
