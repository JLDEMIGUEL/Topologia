import io
import time

import imageio
import matplotlib.pyplot as plt
import numpy as np
from IPython.core.display_functions import clear_output
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d

from SimplicialComplex.SimplicialComplex import SimplicialComplex
from SimplicialComplex.utils.alpha_complex_utils import compute_circumference_radius, compute_edge_value, \
    plot_triangles, plot_edges, filter_faces, gif_plot_edges, gif_plot_triangles
from SimplicialComplex.utils.simplicial_complex_utils import filter_by_float, sort_vertex


class AlphaComplex(SimplicialComplex):
    """
    Class used to represent a AlphaComplex.
    Extends from SimplicialComplex.

    Attributes:

    tri (Delaunay): stores Delaunay triangulation for the given points
    faces (dict): stores a dictionary with faces as keys and float as value

    """

    def __init__(self, points: np.array) -> None:
        """
        Instantiates a new AlphaComplex.
        Args:
            points (np.array): array of points
        Returns:
            None: Instantiates a new AlphaComplex
        """
        self.tri = Delaunay(points)
        # Add all the vertex to the complex
        sorted_faces = sort_vertex([tuple([int(num) for num in tup]) for tup in self.tri.simplices])
        aux = SimplicialComplex(sorted_faces)
        super().__init__(aux.n_faces(0))
        # Add the value of each edge
        for x in aux.n_faces(1):
            r = compute_edge_value(self.tri.points[x[0]], self.tri.points[x[1]], self.tri.points)
            if r is not None:
                self.add({x}, r)
        # Add the value of each triangle
        for x in aux.n_faces(2):
            self.add({x},
                     compute_circumference_radius(self.tri.points[x[0]], self.tri.points[x[1]], self.tri.points[x[2]]))

        if len(self.faces.keys()) > 30:
            self.faces = filter_faces(self.faces)

    def plot_alpha(self, sleep_time=None) -> None:
        """
        Plots the AlphaComplex iterating the dict values in order.
        Returns:
            None:
        """
        if not sleep_time:
            sleep_time = .1
        vor = Voronoi(self.tri.points)

        for x in self.threshold_values():
            clear_output()
            # Compute compute_edge_value and triangles
            faces = filter_by_float(self.faces, x)
            edges_list = [list(edge) for edge in faces if len(edge) == 2]
            triangles = [list(triangle) for triangle in faces if len(triangle) == 3]
            # Plot voronoi, points, compute_edge_value and triangles
            voronoi_plot_2d(vor, show_vertices=False, line_width=2, line_colors='blue')
            plt.plot(self.tri.points[:, 0], self.tri.points[:, 1], 'ko')
            plot_edges(edges_list, self.tri)
            plot_triangles(triangles, self.tri)

            plt.show()
            time.sleep(sleep_time)

    def gif_alpha(self) -> bytes:
        """
        Plots the AlphaComplex iterating the dict values in order.

        Returns:
            bytes: the bytes of the gif
        """
        images = []
        vor = Voronoi(self.tri.points)
        for x in self.threshold_values():
            fig, ax = plt.subplots()
            # Compute compute_edge_value and triangles
            faces = filter_by_float(self.faces, x)
            edges_list = [list(edge) for edge in faces if len(edge) == 2]
            triangles = [list(triangle) for triangle in faces if len(triangle) == 3]
            # Plot voronoi, points, compute_edge_value and triangles
            voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_width=2, line_colors='blue')
            ax.plot(self.tri.points[:, 0], self.tri.points[:, 1], 'ko')
            gif_plot_edges(edges_list, self.tri, ax)
            gif_plot_triangles(triangles, self.tri, ax)
            images.append(fig)
            plt.close()

        with io.BytesIO() as buffer:
            with imageio.get_writer(buffer, mode='I', duration=1000 / 3, format="gif", loop=0) as writer:
                for fig in images:
                    with io.BytesIO() as img_buffer:
                        fig.savefig(img_buffer, format='png')
                        img_buffer.seek(0)
                        image = imageio.imread_v2(img_buffer)
                        writer.append_data(image)
                    plt.close(fig)
            file_content = buffer.getvalue()
        return file_content
