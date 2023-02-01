from unittest import TestCase

import numpy as np

from SimplicialComplex.utils.vietoris_complex_utils import all_faces, get_all_radios, calc_radio


class Test(TestCase):

    def test_calc_radio_1(self):
        expected = 0
        self.assertEqual(expected, calc_radio(([0, 3],), {}))

    def test_calc_radio_2(self):
        expected = 1.4142135623730951
        self.assertEqual(expected, calc_radio(([0, 3], [-1,  2]), {}))

    def test_calc_radio_3(self):
        expected = 3.1622776601683795
        self.assertEqual(expected, calc_radio(([0, 3], [-1,  2], [1, 0]), {}))

    def test_all_faces_1(self):
        expected = {(), (0,), (0, 1), (0, 1, 2), (0, 1, 2, 3), (0, 1, 3), (0, 2), (0, 2, 3), (0, 3), (1,), (1, 2),
                    (1, 2, 3), (1, 3), (2,), (2, 3), (3,)}
        self.assertEqual(expected, all_faces({(0, 1, 2, 3)}, (0, 1, 2, 3)))

    def test_all_faces_2(self):
        expected = {(0, 1, 2), (0, 1, 2, 3), (0, 1, 3), (0, 2), (0, 2, 3), (0, 3), (1, 2), (1, 2, 3), (1, 3), (2,),
                    (2, 3), (3,)}
        self.assertEqual(expected, all_faces(
            {(1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3), (0, 3), (2, 3), (0, 2), (0, 1, 2, 3), (3,), (0, 1, 2), (1, 3)},
            (0, 2)))

    def test_get_all_radios(self):
        expected = {(): 0, (0,): 0, (0, 1): 1.4142135623730951, (0, 1, 2): 3.1622776601683795, (0, 1, 2, 3): 6.0,
                    (0, 1, 3): 6.0, (0, 2): 3.1622776601683795, (0, 2, 3): 6.0, (0, 3): 6.0, (1,): 0,
                    (1, 2): 2.8284271247461903, (1, 2, 3): 5.0990195135927845, (1, 3): 5.0990195135927845, (2,): 0,
                    (2, 3): 3.1622776601683795, (3,): 0}
        self.assertEqual(expected, get_all_radios(
            {(0, 1), (1, 2), (2,), (0,), (0, 1, 3), (1, 2, 3), (0, 2, 3), (0, 3), (2, 3), (1,), (0, 2), (0, 1, 2, 3),
             (3,), (0, 1, 2), (), (1, 3)}, np.array([[0, 3], [-1, 2], [1, 0], [0, -3]])))