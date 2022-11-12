import math

import matplotlib.colors
import numpy as np
from matplotlib import pyplot as plt


def radius(a, b, c):
    """
    radius
    Args:

    Returns the Circumscribed circle radius of the given triangle
    """
    lado_a = math.dist(c, b)
    lado_b = math.dist(a, c)
    lado_c = math.dist(a, b)
    semiperimetro = (lado_a + lado_b + lado_c) * 0.5
    radio = lado_a * lado_b * lado_c * 0.25 / math.sqrt(
        semiperimetro * (semiperimetro - lado_a) * (semiperimetro - lado_b) * (semiperimetro - lado_c))
    return radio


def edges(v1, v2, points):
    """
    edges
    Args:
        v1:
        v2:
        points:
    Returns None or radius depending on AlphaComplex algorithm
    """
    radio = math.dist(v1, v2) * 0.5
    centro = (v1 + v2) * 0.5
    for x in range(len(points)):
        if math.dist(centro, points[x]) < radio:
            if math.dist(points[x], v1) > 0:
                if math.dist(points[x], v2) > 0:
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
