import math

import numpy as np


def calc_radio(points: tuple, data: dict) -> float:
    """
    Calculates vietoris rips value.
    Args:
        data (dict):
        points (tuple): tuple of points
    Returns:
        float: value
    """
    if len(points) <= 1:
        return 0
    tup = tuple([tuple(e) for e in points])
    sub_tuple = tuple([tuple(e) for e in points[slice(len(points)-1)]])
    maximum = data.get(sub_tuple, 0)
    last = points[len(points)-1]
    for y in sub_tuple:
        dist = math.dist(y, last)
        if maximum < dist:
            maximum = dist
    data[tup] = maximum
    return maximum


def all_faces(combinations: set, points: tuple) -> set:
    """
    Computes all the faces of a Vietoris Rips Complex.
    Args:
        combinations (set): set with early faces
        points (tuple): coordinates of every face
    Returns:
        Set: combinations set
    """
    for i in range(len(points)):
        face2 = tuple(j for j in points if i != j)
        previous_size = len(combinations)
        combinations.add(face2)
        if len(combinations) == previous_size:
            return combinations
        all_faces(combinations, face2)
    return combinations


def get_all_radios(faces: set, points: np.array) -> dict:
    """
    Computes all the radius of the faces
    Args:
        faces (set): set of faces
        points (np.array): points coordinates
    Returns:
        dict: dictionary with faces as keys and their respective values
    """
    dic = dict()
    cache = dict()
    for x in sorted(faces, key=lambda a: (len(a), a)):
        dic[x] = calc_radio(tuple(points[j] for j in x), cache)
    return dic
