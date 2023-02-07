from unittest import TestCase, mock
from unittest.mock import patch, ANY

import numpy as np
from scipy.spatial import Delaunay

from SimplicialComplex.utils.alpha_complex_utils import compute_circumference_radius, compute_edge_value, plot_edges, plot_triangles


class Test(TestCase):
    def test_radius_1(self):
        a = (1.0, 0.0)
        b = (0.0, 1.0)
        c = (2.0, 1.0)
        expected_radius = round(1, 14)
        self.assertEqual(expected_radius, round(compute_circumference_radius(a, b, c), 14))

    def test_radius_2(self):
        a = (0.38021547, 0.46419202)
        b = (0.79516283, 0.4926363)
        c = (0.56662377, 0.03832562)
        expected_radius = round(0.27011640994552, 6)
        self.assertEqual(expected_radius, round(compute_circumference_radius(a, b, c), 6))

    def test_edges_1(self):
        a = np.array([2.0, 0.0])
        b = np.array([0.0, 0.0])
        points = []
        expected_radius = 1
        self.assertEqual(expected_radius, compute_edge_value(a, b, points), 6)

    def test_edges_2(self):
        a = np.array([4.0, 0.0])
        b = np.array([0.0, 0.0])
        points = [a, b, np.array([1.0, 1.0])]
        expected_radius = None
        self.assertEqual(expected_radius, compute_edge_value(a, b, points), 6)

    def test_plotedges(self):
        with patch('matplotlib.pyplot.plot') as mocked_plot:
            points = [[-3, 0], [0, 1], [3, 0], [-1.7, -1.8], [1.7, -1.8], [0, -4]]
            edges_list = [[0, 1], [1, 3], [2, 4], [1, 2], [0, 3], [1, 4], [4, 5], [3, 5]]
            plot_edges(edges_list, Delaunay(points))

            self.assertEqual(8, mocked_plot.call_count)

            self.assertEqual([
                 mock.call([-3.0, 0.0], [0.0, 1.0], 'k'),
                 mock.call([0.0, -1.7], [1.0, -1.8], 'k'),
                 mock.call([3.0, 1.7], [0.0, -1.8], 'k'),
                 mock.call([0.0, 3.0], [1.0, 0.0], 'k'),
                 mock.call([-3.0, -1.7], [0.0, -1.8], 'k'),
                 mock.call([0.0, 1.7], [1.0, -1.8], 'k'),
                 mock.call([1.7, 0.0], [-1.8, -4.0], 'k'),
                 mock.call([-1.7, 0.0], [-1.8, -4.0], 'k')
            ], mocked_plot.mock_calls)

    def test_plottriangles(self):
        with patch('matplotlib.pyplot.tripcolor') as mocked_tripcolor:
            points = [[-3, 0], [0, 1], [3, 0], [-1.7, -1.8], [1.7, -1.8], [0, -4]]
            triangles = [[[0, 1, 3], [1, 2, 4], [1, 3, 4], [3, 4, 5]]]
            plot_triangles(triangles, Delaunay(points))

            self.assertEqual(1, mocked_tripcolor.call_count)

            mocked_tripcolor.assert_called_with(ANY, ANY, [[[0, 1, 3], [1, 2, 4], [1, 3, 4], [3, 4, 5]]], ANY,
                                                edgecolor='k', lw=2, cmap=ANY)
