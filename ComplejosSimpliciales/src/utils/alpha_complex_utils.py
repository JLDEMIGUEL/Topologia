import math

import matplotlib.colors
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import Delaunay


def radius(a: tuple, b: tuple, c: tuple) -> float:
    """
    Computes the radius of the circumference which contains the three points.
    Args:
        a (tuple): first point
        b (tuple): second point
        c (tuple): third point
    Returns:
        float: circumscribed circle radius of the given triangle
    """
    lado_a = math.dist(c, b)
    lado_b = math.dist(a, c)
    lado_c = math.dist(a, b)
    semiperimetro = (lado_a + lado_b + lado_c) * 0.5
    radio = lado_a * lado_b * lado_c * 0.25 / math.sqrt(
        semiperimetro * (semiperimetro - lado_a) * (semiperimetro - lado_b) * (semiperimetro - lado_c))
    return radio


def edges(v1: tuple, v2: tuple, points: np.ndarray) -> float | None:
    """
    Computes the length of the edge if exists.
    Args:
        v1 (tuple): first point
        v2 (tuple): second point
        points (np.ndarray): array of other points
    Returns:
        float | None: None or radius depending on AlphaComplex algorithm
    """
    radio = math.dist(v1, v2) * 0.5
    centro = (v1 + v2) * 0.5
    for x in range(len(points)):
        if math.dist(centro, points[x]) < radio:
            if math.dist(points[x], v1) > 0:
                if math.dist(points[x], v2) > 0:
                    return
    return radio


def plottriangles(triangles: list, tri: Delaunay) -> None:
    """
    Plots the given triangles.
    Args:
        tri (Delaunay): Delaunay triangulation
        triangles (list): list of triangles
    Returns:
        None:
    """
    if len(triangles) > 0:
        c = np.ones(len(triangles))
        cmap = matplotlib.colors.ListedColormap("limegreen")
        plt.tripcolor(tri.points[:, 0], tri.points[:, 1], triangles, c, edgecolor="k", lw=2, cmap=cmap)


def plotedges(edges_list: list, tri: Delaunay) -> None:
    """
    Plots the given edges.
    Args:
        tri (Delaunay): Delaunay triangulation
        edges_list (list): list of edges
    Returns:
        None:
    """
    for edge in edges_list:
        x = [tri.points[edge[0], 0], tri.points[edge[1], 0]]
        y = [tri.points[edge[0], 1], tri.points[edge[1], 1]]
        plt.plot(x, y, 'k')
