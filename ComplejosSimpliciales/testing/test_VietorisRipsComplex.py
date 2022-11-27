import os
from pathlib import Path
from unittest import TestCase

import numpy as np

from ComplejosSimpliciales.src.VietorisRipsComplex import Vietoris_RipsComplex

ROOT_DIR = os.path.abspath(Path(__file__).parent.parent.parent)


class TestVietoris_RipsComplex(TestCase):
    ac1 = Vietoris_RipsComplex(np.array([[0, 3], [-1, 2], [1, 0], [0, -3]]))

    def test_constructor(self):
        expected_combinations = {(), (0,), (0, 1), (0, 1, 2), (0, 1, 3), (0, 2), (0, 2, 3), (0, 3), (1,), (1, 2),
                                 (0, 1, 2, 3), (1, 2, 3), (1, 3), (2,), (2, 3), (3,)}
        expected_dict = {(): 0, (0,): 0, (0, 1): 1.4142135623730951, (0, 1, 2): 3.1622776601683795, (0, 1, 2, 3): 6.0,
                         (0, 1, 3): 6.0, (0, 2): 3.1622776601683795, (0, 2, 3): 6.0, (0, 3): 6.0, (1,): 0,
                         (1, 2): 2.8284271247461903, (1, 2, 3): 5.0990195135927845, (1, 3): 5.0990195135927845, (2,): 0,
                         (2, 3): 3.1622776601683795, (3,): 0}
        self.assertEqual(expected_combinations, self.ac1.combinations)
        self.assertEqual(expected_dict, self.ac1.dic)
