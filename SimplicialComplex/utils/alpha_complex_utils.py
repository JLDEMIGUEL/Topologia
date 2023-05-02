import math
import statistics

import matplotlib.colors
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import Delaunay


def compute_circumference_radius(a: tuple, b: tuple, c: tuple) -> float:
    """
    Computes the radius of the circumference which contains the three points.
    Args:
        a (tuple): first point
        b (tuple): second point
        c (tuple): third point
    Returns:
        float: circumscribed circle radius of the given triangle
    """
    edge_a, edge_b, edge_c = math.dist(c, b), math.dist(a, c), math.dist(a, b)
    semi_perimeter = (edge_a + edge_b + edge_c) * 0.5
    radius = edge_a * edge_b * edge_c * 0.25 / math.sqrt(
        semi_perimeter * (semi_perimeter - edge_a) * (semi_perimeter - edge_b) * (semi_perimeter - edge_c))
    return radius


def compute_edge_value(v1: np.array, v2: np.array, points: np.array) -> float | None:
    """
    Computes the length of the edge if exists.
    Args:
        v1 (np.array): first point
        v2 (np.array): second point
        points (np.array): array of other points
    Returns:
        float | None: None or edge radius depending on AlphaComplex algorithm
    """
    radius = math.dist(v1, v2) * 0.5
    center = (v1 + v2) * 0.5
    for point in points:
        # If the point is in the other point's radius, return
        if math.dist(center, point) < radius and point.tolist() not in (v1.tolist(), v2.tolist()):
            return
    return radius


def plot_triangles(triangles: list, tri: Delaunay) -> None:
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


def plot_edges(edges_list: list, tri: Delaunay) -> None:
    """
    Plots the given edge.
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


def gif_plot_triangles(triangles: list, tri: Delaunay, ax) -> list:
    """
    Plots the given triangles.
    Args:
        tri (Delaunay): Delaunay triangulation
        triangles (list): list of triangles
        ax:
    Returns:
        list: list of plots
    """
    plots = []
    if len(triangles) > 0:
        c = np.ones(len(triangles))
        cmap = matplotlib.colors.ListedColormap("limegreen")
        plots.append(ax.tripcolor(tri.points[:, 0], tri.points[:, 1], triangles, c, edgecolor="k", lw=2, cmap=cmap))
    return plots


def gif_plot_edges(edges_list: list, tri: Delaunay, ax) -> None:
    """
    Plots the given edge.
    Args:
        tri (Delaunay): Delaunay triangulation
        edges_list (list): list of edges
    Returns:
        None:
    """
    plots = []
    for edge in edges_list:
        x = [tri.points[edge[0], 0], tri.points[edge[1], 0]]
        y = [tri.points[edge[0], 1], tri.points[edge[1], 1]]
        plots.append(ax.plot(x, y, 'k'))
    return plots


def filter_faces(dic: dict) -> dict:
    """
    Filter faces in a dictionary by removing faces with high values.
    Args:
        dic (dict): faces dictionary
    Returns:
        dict: The filtered dictionary with faces removed.
    """
    ordered_faces = sorted(dic.keys(), key=lambda face: dic[face])
    while statistics.mean(dic.values()) > 1.25 * statistics.median(dic.values()):
        last = ordered_faces[-1]
        ordered_faces.remove(last)
        dic.pop(last)
    return dic
