import math

import matplotlib.colors
import numpy as np
from matplotlib import pyplot as plt


def radius(tri, points):
    """
    radius
    Args:
        tri (Delaunay): Delaunay triangulation
        points (np.array): Array of points
    Returns the Circumscribed circle radius of the given triangle
    """
    a = tri.points[points[0]]
    b = tri.points[points[1]]
    c = tri.points[points[2]]
    lado_a = math.dist(c, b)
    lado_b = math.dist(a, c)
    lado_c = math.dist(a, b)
    semiperimetro = (lado_a + lado_b + lado_c) * 0.5
    radio = lado_a * lado_b * lado_c * 0.25 / math.sqrt(
        semiperimetro * (semiperimetro - lado_a) * (semiperimetro - lado_b) * (semiperimetro - lado_c))
    return radio


def edges(tri, arista):
    """
    edges
    Args:
        tri (Delaunay): Delaunay triangulation
        arista (tuple): tuple of vertex
    Returns None or radius depending on AlphaComplex algorithm
    """
    v1 = tri.points[arista[0]]
    v2 = tri.points[arista[1]]
    radio = math.dist(v1, v2) * 0.5
    centro = (v1 + v2) * 0.5
    for x in range(len(tri.points)):
        if math.dist(centro, tri.points[x]) < radio:
            if math.dist(tri.points[x], v1) > 0:
                if math.dist(tri.points[x], v2) > 0:
                    return
    return radio


def plottriangles(triangles, tri):
    """
    plottriangles
    Args:
        tri (Delaunay): Delaunay triangulation
        triangles (list): list of triangles
    Plots the given triangles
    """
    if len(triangles) > 0:
        c = np.ones(len(triangles))
        cmap = matplotlib.colors.ListedColormap("limegreen")
        plt.tripcolor(tri.points[:, 0], tri.points[:, 1], triangles, c, edgecolor="k", lw=2, cmap=cmap)


def plotedges(edges, tri):
    """
    plotedges
    Args:
        tri (Delaunay): Delaunay triangulation
        edges (list): list of edges
    Plots the given edges
    """
    for edge in edges:
        x = [tri.points[edge[0], 0], tri.points[edge[1], 0]]
        y = [tri.points[edge[0], 1], tri.points[edge[1], 1]]
        plt.plot(x, y, 'k')


def order(faces):
    """
     order

     Args:
         faces: set of faces of a Simplicial Complex

     Returns the ordered list of faces
     """
    # faces.remove(())
    return sorted(faces, key=lambda a: (a, len(a)))


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


def simplify_columns(matrix):
    """
    Simplifies the columns of the given matrix
    Args:
        matrix: np.matrix

    Returns the simplified matrix

    """
    columns = matrix.shape[1]
    for i in range(columns - 1):
        i += 1
        if matrix[0, i] == 1:
            matrix[:, i] = (matrix[:, i] + matrix[:, 0]) % 2
    return matrix


def simplify_rows(matrix):
    """
    Simplifies the rows of the given matrix
    Args:
        matrix: np.matrix

    Returns the simplified matrix

    """
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


def connected_components(complex: set):
    """
    Returns number of connected components of the SimplicialComplex
    Returns:
        int: number of connected components
    """
    vertex = [x[0] for x in complex if len(x) == 1]
    edges = [x for x in complex if len(x) == 2]
    visitedVertex = dict()
    for x in vertex:
        visitedVertex[x] = False
    components = set()
    for vert in vertex:
        if not visitedVertex[vert]:
            reachableList = sorted(reachable(edges, vert, visitedVertex), key=lambda a: a)
            components.add(tuple(reachableList))
    return len(components)


def reachable(edges: list | set | tuple, vert: int, visitedVertex: dict) -> list:
    """
    Args:
        edges (list | set | tuple): list of edges
        visitedVertex (dict): dict with the visited vertex
        vert (int): entry vertex to get the list of reachable vertex
    Returns:
        list: list of reachable vertex from the given vertex
    """
    reach = [vert]
    visitedVertex[vert] = True
    for edge in edges:
        if vert in edge:
            tup = tuple(x for x in edge if x != vert)
            endVert = tup[0]
            if not visitedVertex[endVert]:
                reach = reach + reachable(edges, endVert, visitedVertex)
    return reach


def num_loops(complex):
    edges = set(face for face in complex if len(face) == 2)
    loops = set()

    for edge1 in edges:
        for edge2 in edges.difference({edge1}):
            for edge3 in edges.difference({edge1, edge2}):
                if len({edge1[0], edge1[1], edge2[0], edge2[1], edge3[0], edge3[1]}) == 3:
                    loop = sorted({edge1[0], edge1[1], edge2[0], edge2[1], edge3[0], edge3[1]}, key=lambda a: a)
                    loops.add(tuple(loop))

    return len(loops)


def num_triang(complex: set):
    return len([x for x in complex if len(x) == 3])
