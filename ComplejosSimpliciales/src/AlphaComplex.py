import time

import matplotlib.pyplot as plt
import numpy as np
from IPython.core.display_functions import clear_output
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d

from ComplejosSimpliciales.src.SimplicialComplex import SimplicialComplex
from ComplejosSimpliciales.src.utils.alpha_complex_utils import radius, edges, plottriangles, plotedges
from ComplejosSimpliciales.src.utils.simplicial_complex_utils import filterByFloat


class AlphaComplex(SimplicialComplex):
    """
    Class used to represent a AlphaComplex
    Extends from SimplicialComplex

    Attributes:

    tri (Delaunay): stores Delaunay triangulation for the given points
    attributes inherited from SimplicialComplex

    """

    def __init__(self, points: np.array) -> None:
        """
        Instantiates a new AlphaComplex

        Args:
            points (np.array): array of points

        Returns:
            None: Instantiates a new AlphaComplex
        """

        self.tri = Delaunay(points)
        aux = SimplicialComplex(tuple([tuple(e) for e in self.tri.simplices]))
        super().__init__(aux.n_faces(0))
        for x in aux.n_faces(1):
            r = edges(self.tri.points[x[0]], self.tri.points[x[1]], self.tri.points)
            if r is not None:
                self.add({x}, r)
        for x in aux.n_faces(2):
            self.add({x}, radius(self.tri.points[x[0]], self.tri.points[x[1]], self.tri.points[x[2]]))

    def plotalpha(self) -> None:
        """
        Plots the AlphaComplex iterating the dict values in order

        Returns:
            None:
        """

        vor = Voronoi(self.tri.points)
        for x in self.thresholdvalues():
            clear_output()
            faces = filterByFloat(self.dic, x)
            edges = [list(edge) for edge in faces if len(edge) == 2]
            triangles = [list(triangle) for triangle in faces if len(triangle) == 3]

            voronoi_plot_2d(vor, show_vertices=False, line_width=2, line_colors='blue')
            plt.plot(self.tri.points[:, 0], self.tri.points[:, 1], 'ko')
            plotedges(edges, self.tri)
            plottriangles(triangles, self.tri)

            plt.show()
            time.sleep(.2)
