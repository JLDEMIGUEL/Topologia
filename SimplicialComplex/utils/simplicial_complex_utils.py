import math

import numpy as np
from matplotlib import pyplot as plt


def order(faces: list | set | tuple) -> list:
    """
    Sorts the list of faces following lexicographic and faces length.
    Args:
        faces (list | set | tuple): set of faces of a Simplicial Complex
    Returns:
        list: ordered list of faces
    """
    # faces.remove(())
    return sorted(faces, key=lambda a: (a, len(a)))


def reachable(edges: list | set | tuple, vert: int, visitedVertex: dict) -> list:
    """
    Returns a list with the reachable vertex from the given vertex.
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


def sub_faces(face: list | set | tuple) -> set:
    """
    Computes the sub-faces of the given face.
    Args:
        face (list | set | tuple): tuple of vertex
    Returns:
        set: set with all sub faces
    """
    auxSet = set()
    for vert in face:
        face2 = tuple(x for x in face if x != vert)
        auxSet.add(face2)
        auxSet = auxSet.union(sub_faces(face2))
    return auxSet


def updateDict(dic_target: dict, faces: list | set | tuple, float_value: float) -> dict:
    """
    Update the dictionary of faces with the new given faces.
    Args:
        dic_target (dict): dict
        float_value (float):
        faces (list | set | tuple): list/set of tuples

    Returns:
        dict: faces with the faces given and the value
    """
    dic = dic_target.copy()
    for face in faces:
        if face not in dic:
            dic[face] = float_value
        elif dic[face] > float_value:
            dic[face] = float_value
    return dic


def order_faces(faces: list | set | tuple) -> set:
    """
    Sorts the faces in lexicographic order.
    Args:
        faces (list | set | tuple):
    Returns:
        set: sorted faces set
    """
    sorted_faces = set()
    for x in faces:
        sorted_faces.add(tuple(sorted(list(x), key=lambda a: a)))
    faces = sorted_faces
    return faces


def filter_by_float(dic: dict, value: float) -> set:
    """
    Returns a set of faces which float value is less than the given one.
    Args:
        dic (dict): dict
        value (float): Float value
    Returns:
        set: faces which float value is less than the given value
    """
    keys = dic.keys()
    res = {x for x in keys if dic[x] <= value}
    return res


def check_if_sub_face(sub_face: tuple, super_face: tuple) -> bool:
    """
    Check if a tuple represents a sub-face of another tuple.
    Args:
        super_face (tuple): A tuple representing a face.
        sub_face (tuple): A tuple representing a face.

    Returns:
        bool: A boolean indicating whether `sub_face` is a sub-face of `super_face`.
    """
    if len(sub_face) is not len(super_face) - 1 or len(sub_face) == 0:
        return False
    for vert in sub_face:
        if vert not in super_face:
            return False
    return True


def noise(points: np.array) -> np.array:
    """
    Add noise to an array of points.
    Args:
        points (np.array): An array of points, with shape (n, 2) where n is the number of points
    Returns:
        np.array: A new array of points with shape (n, 2), where each point has been perturbed by noise
    """
    mean = sum([math.sqrt(p[0] ** 2 + p[1] ** 2) for p in points]) / len(points)
    return np.array([np.array(p) + np.random.normal(mean, 0.1, size=2) for p in points])


def connected_components(complex_faces: set) -> int:
    """
    Returns number of connected components of the SimplicialComplex.
    Args:
        complex_faces (set): the faces of the complex
    Returns:
        int: number of connected components
    """
    vertex = [x[0] for x in complex_faces if len(x) == 1]
    edges = [x for x in complex_faces if len(x) == 2]
    # Build a visited vertex dictionary
    visited_vertex = {x: False for x in vertex}
    # For each vertex, compute its component
    components = set()
    for vert in vertex:
        if not visited_vertex[vert]:
            reachableList = sorted(reachable_alg(edges, vert, visited_vertex), key=lambda a: a)
            components.add(tuple(reachableList))
    return len(components)


def reachable_alg(edges: list | set | tuple, vert: int, visitedVertex: dict) -> list:
    """
    Returns a list with the reachable vertex from the given vertex.
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


def num_loops(complex_faces: set) -> int:
    """
    Computes the number of loops in the complex.
    Args:
        complex_faces (set):  the faces of the complex

    Returns:
        int: the number of loops
    """
    edges = set(face for face in complex_faces if len(face) == 2)
    loops = set()

    for edge1 in edges:
        for edge2 in edges.difference({edge1}):
            for edge3 in edges.difference({edge1, edge2}):
                if len({edge1[0], edge1[1], edge2[0], edge2[1], edge3[0], edge3[1]}) == 3:
                    loop = sorted({edge1[0], edge1[1], edge2[0], edge2[1], edge3[0], edge3[1]}, key=lambda a: a)
                    loops.add(tuple(loop))

    return len(loops)


def num_triangles(complex_faces: set) -> int:
    """
    Computes the number of triangles in the complex.
    Args:
        complex_faces (set):  the faces of the complex

    Returns:
        int: the number of triangles
    """
    return len([x for x in complex_faces if len(x) == 3])


def calc_homology(complex_faces: object) -> tuple[int, int, int]:
    """
    Computes the homology of the complex.
    Args:
        complex_faces (set):  the faces of the complex

    Returns:
        tuple[int, int, int]: the number of connected components, number of triangles and number of loops
    """
    return connected_components(complex_faces), num_loops(complex_faces), num_triangles(complex_faces)


colors = ["b", "g", "r", "m", "y", "b", "g", "r", "m", "y"]


def plot_persistence_diagram(points: dict, infinite: int) -> None:
    """
    Plot the persistence diagram of a set of points.
    Args:
        points (dict): A dictionary where the keys are integers representing the dimension of the points,
                       and the values are lists of points
        infinite (int): The maximum value to be plotted on the x and y axis
    Returns:
        None
    """
    # Plot all points of the diagram
    for dim, points_list in points.items():
        points_list = np.array([np.array(point) for point in points_list])
        plt.plot(points_list[:, 0].tolist(), points_list[:, 1].tolist(), colors[dim % len(colors)] + "o")
    # Plot axis
    plt.axis([-0.1 * infinite, infinite * 1.1, -0.1 * infinite, infinite * 1.1])
    plt.plot([-0.1 * infinite, infinite * 1.1], [-0.1 * infinite, infinite * 1.1], "b--")
    plt.plot([-0.1 * infinite, infinite * 1.1], [infinite, infinite], "b--")


def plot_barcode_diagram(points: dict) -> None:
    """
    Plot the barcode diagram of a set of points.
    Args:
        points (dict): A dictionary where the keys are integers representing the dimension of the points,
                       and the values are lists of points
    Returns:
        None
    """
    # Plot all bars of the diagram
    height = 0
    for dim, points_list in points.items():
        for point in points_list:
            if point[0] != point[1]:
                plt.plot([point[0], point[1]], [height, height], colors[dim])
                height += 1
