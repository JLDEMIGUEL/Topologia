import math
from random import random

import numpy as np


def order(faces: list | set | tuple) -> list:
    """
    Args:
        faces (list | set | tuple): set of faces of a Simplicial Complex
    Returns:
        list: ordered list of faces
    """
    # faces.remove(())
    return sorted(faces, key=lambda a: (a, len(a)))


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


def subFaces(face: list | set | tuple) -> set:
    """
    Args:
        face (list | set | tuple): tuple of vertex
    Returns:
        set: set with all sub faces
    """
    auxSet = set()
    for vert in face:
        face2 = tuple(x for x in face if x != vert)
        auxSet.add(face2)
        auxSet = auxSet.union(subFaces(face2))
    return auxSet


def updateDict(dic_target: dict, faces: list | set | tuple, float_value: float) -> dict:
    """
    Args:
        dic_target (dict): dict
        float_value (float):
        faces (list | set | tuple): list/set of tuples

    Returns:
        dict: dic with the faces given and the value
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


def filterByFloat(dic: dict, value: float) -> set:
    """
    Args:
        dic (dict): dict
        value (float): Float value
    Returns:
        set: faces which float value is less than the given value
    """
    keys = dic.keys()
    res = {x for x in keys if dic[x] <= value}
    return res


def noise(points: np.array):
    mean = sum([math.sqrt(p[0] ** 2 + p[1] ** 2) for p in points]) / len(points)
    return np.array([np.array(p) + np.random.normal(mean, 0.1, size=2) for p in points])
