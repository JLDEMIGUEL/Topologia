from unittest import TestCase

import numpy as np

from ComplejosSimpliciales.src.utils.alpha_complex_utils import radius, edges


class Test(TestCase):
    def test_radius_1(self):
        a = (1.0, 0.0)
        b = (0.0, 1.0)
        c = (2.0, 1.0)
        expected_radius = round(1, 14)
        self.assertEqual(expected_radius, round(radius(a, b, c), 14))

    def test_radius_2(self):
        a = (0.38021547, 0.46419202)
        b = (0.79516283, 0.4926363)
        c = (0.56662377, 0.03832562)
        expected_radius = round(0.27011640994552, 6)
        self.assertEqual(expected_radius, round(radius(a, b, c), 6))

    def test_edges_1(self):
        a = np.array([2.0, 0.0])
        b = np.array([0.0, 0.0])
        points = []
        expected_radius = 1
        self.assertEqual(expected_radius, edges(a, b, points), 6)

    def test_edges_2(self):
        a = np.array([4.0, 0.0])
        b = np.array([0.0, 0.0])
        points = [a, b, np.array([1.0, 1.0])]
        expected_radius = None
        self.assertEqual(expected_radius, edges(a, b, points), 6)
