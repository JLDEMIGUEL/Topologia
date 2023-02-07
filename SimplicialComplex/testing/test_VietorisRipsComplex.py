from unittest import TestCase

import numpy as np

from SimplicialComplex.VietorisRipsComplex import Vietoris_RipsComplex


class TestVietoris_RipsComplex(TestCase):
    ac1 = Vietoris_RipsComplex(np.array([[0, 3], [-1, 2], [1, 0], [0, -3]]))

    def test_constructor(self):
        expected_combinations = {(), (0,), (0, 1), (0, 1, 2), (0, 1, 3), (0, 2), (0, 2, 3), (0, 3), (1,), (1, 2),
                                 (0, 1, 2, 3), (1, 2, 3), (1, 3), (2,), (2, 3), (3,)}
        expected_dict = {(): 0, (0,): 0, (0, 1): 1.4142135623730951, (0, 1, 2): 3.1622776601683795, (0, 1, 2, 3): 6.0,
                         (0, 1, 3): 6.0, (0, 2): 3.1622776601683795, (0, 2, 3): 6.0, (0, 3): 6.0, (1,): 0,
                         (1, 2): 2.8284271247461903, (1, 2, 3): 5.0990195135927845, (1, 3): 5.0990195135927845, (2,): 0,
                         (2, 3): 3.1622776601683795, (3,): 0}
        self.assertEqual(expected_combinations, self.ac1.faces.keys())
        self.assertEqual(expected_dict, self.ac1.faces)

    def test_dimension(self):
        self.assertEqual(3, self.ac1.dimension())

    def test_connected_components(self):
        self.assertEqual(1, self.ac1.connected_components())

    def test_euler_characteristic(self):
        self.assertEqual(1, self.ac1.euler_characteristic())

    def test_betti_numbers(self):
        self.assertEqual((1, 0, 0, 0), (
            self.ac1.betti_number(0), self.ac1.betti_number(1), self.ac1.betti_number(2), self.ac1.betti_number(3)))
