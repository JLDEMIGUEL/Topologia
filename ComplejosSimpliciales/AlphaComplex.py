import numpy as np
from SimplicialComplex import SimplicialComplex
import math
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib as mpl
from IPython.core.display_functions import clear_output
import time


class AlphaComplex:

    def __init__(self, points):
        tri = Delaunay(points)
        aux = SimplicialComplex(tuple([tuple(e) for e in tri.simplices]))
        self.sc = SimplicialComplex(aux.n_faces(0))
        for x in aux.n_faces(1):
            self.aristas(tri, x)
        for x in aux.n_faces(2):
            self.sc.add({x}, self.radio(tri, x))


    def radio(self, tri, points):
        a = tri.points[points[0]]
        b = tri.points[points[1]]
        c = tri.points[points[2]]
        lado_a = math.dist(a, b)
        lado_b = math.dist(a, c)
        lado_c = math.dist(c, b)
        semiperimetro = (lado_a + lado_b + lado_c) * 0.5
        radio = lado_a * lado_b * lado_c * 0.25 / math.sqrt(
            semiperimetro * (semiperimetro - lado_a) * (semiperimetro - lado_b) * (semiperimetro - lado_c))
        return radio

    def aristas(self, tri, arista):
        v1 = tri.points[arista[0]]
        v2 = tri.points[arista[1]]
        radio = math.dist(v1, v2) * 0.5
        centro = (v1 + v2) * 0.5
        for x in range(len(tri.points)):
            if math.dist(centro, tri.points[x]) < radio:
                if math.dist(tri.points[x], v1) > 0:
                    if math.dist(tri.points[x], v2) > 0:
                        return
        self.sc.add({arista}, radio)

    def plotalpha(self, points):
        tri = Delaunay(points)
        vor = Voronoi(points)

        for x in self.sc.thresholdvalues():
            clear_output()
            faces = self.sc.filterByFloat(x)
            edges = [list(edge) for edge in faces if len(edge) == 2]
            triangles = [list(triangle) for triangle in faces if len(triangle) == 3]
            fig = voronoi_plot_2d(vor, show_vertices=False, line_width=2, line_colors='blue')
            c = np.ones(len(triangles))
            cmap = matplotlib.colors.ListedColormap("limegreen")
            if len(triangles) > 0:
                tri
                plt.tripcolor(points[:, 0], points[:, 1], triangles, c, edgecolor="k", lw=2, cmap=cmap)
            plt.plot(points[:, 0], points[:, 1], 'ko')
            for edge in edges:
                x = [points[edge[0], 0], points[edge[1], 0]]
                y = [points[edge[0], 1], points[edge[1], 1]]
                plt.plot(x, y, 'k')
            plt.show()
            time.sleep(.5)

