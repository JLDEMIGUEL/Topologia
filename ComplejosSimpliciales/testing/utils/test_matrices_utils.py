from unittest import TestCase

import numpy as np

from ComplejosSimpliciales.src.utils.matrices_utils import search_one, swap, simplify_rows, simplify_columns, \
    reconstruct, smith_normal_form, gcd_euclides, matrix_gcd


class Test(TestCase):
    m1 = np.matrix([[1, 1, 1, 0, 0, 0],
                    [1, 0, 0, 1, 1, 0],
                    [0, 1, 0, 1, 0, 1],
                    [0, 0, 1, 0, 1, 1]])

    m2 = np.matrix([[0, 0, 0],
                    [0, 0, 0],
                    [0, 1, 1],
                    [0, 0, 1]])

    m3 = np.matrix([[0],
                    [0],
                    [1],
                    [1]])

    m4 = np.matrix([0, 1, 1, 1])

    def test_search_one_1(self):
        ret = [0, 0]
        self.assertEqual(ret, search_one(self.m1))

    def test_search_one_2(self):
        ret = [2, 1]
        self.assertEqual(ret, search_one(self.m2))

    def test_search_one_3(self):
        ret = [2, 0]
        self.assertEqual(ret, search_one(self.m3))

    def test_search_one_4(self):
        ret = [0, 1]
        self.assertEqual(ret, search_one(self.m4))

    def test_swap_1(self):
        source = [1, 0]
        obj = [0, 0]
        expected_matrix = np.matrix([[1, 0, 0, 1, 1, 0],
                                     [1, 1, 1, 0, 0, 0],
                                     [0, 1, 0, 1, 0, 1],
                                     [0, 0, 1, 0, 1, 1]])
        swapped_matrix = swap(self.m1, source, obj)
        self.assertTrue((expected_matrix == swapped_matrix).all())

    def test_swap_2(self):
        source = [1, 0]
        obj = [2, 0]
        expected_matrix = np.matrix([[0],
                                     [1],
                                     [0],
                                     [1]])
        swapped_matrix = swap(self.m3, source, obj)
        self.assertTrue((expected_matrix == swapped_matrix).all())

    def test_simplify_columns(self):
        expected_matrix = np.matrix([[1, 0, 0, 0, 0, 0],
                                     [1, 1, 1, 1, 1, 0],
                                     [0, 1, 0, 1, 0, 1],
                                     [0, 0, 1, 0, 1, 1]])
        simplified_matrix = simplify_columns(self.m1)
        self.assertTrue((expected_matrix == simplified_matrix).all())

    def test_simplify_rows(self):
        expected_matrix = np.matrix([[1, 1, 1, 0, 0, 0],
                                     [0, 1, 1, 1, 1, 0],
                                     [0, 1, 0, 1, 0, 1],
                                     [0, 0, 1, 0, 1, 1]])
        simplified_matrix = simplify_rows(self.m1)
        self.assertTrue((expected_matrix == simplified_matrix).all())

    def test_reconstruct_1(self):
        matrix = np.matrix([[1, 0, 0, 0],
                            [0, 0, 0, 0]])
        aux = np.matrix([[0, 0, 0]])
        expected_matrix = np.matrix([[1, 0, 0, 0],
                                     [0, 0, 0, 0]])
        reconstructed = reconstruct(matrix, aux)
        self.assertTrue((expected_matrix == reconstructed).all())

    def test_reconstruct_2(self):
        matrix = np.matrix([[1, 0, 0, 0, 0],
                            [0, 1, 0, 1, 1],
                            [0, 1, 0, 1, 1]])
        aux = np.matrix([[1, 0, 0, 0],
                         [0, 0, 0, 0]])
        expected_matrix = np.matrix([[1, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 0]])
        reconstructed = reconstruct(matrix, aux)
        self.assertTrue((expected_matrix == reconstructed).all())

    def test_smith_normal_form_1(self):
        expected_matrix = np.matrix([[1, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0],
                                     [0, 0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0]])
        smf = smith_normal_form(self.m1)
        self.assertTrue((expected_matrix == smf).all())

    def test_smith_normal_form_2(self):
        expected_matrix = np.matrix([[1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 0],
                                     [0, 0, 0]])
        smf = smith_normal_form(self.m2)
        self.assertTrue((expected_matrix == smf).all())

    def test_smith_normal_form_3(self):
        expected_matrix = np.matrix([[1],
                                     [0],
                                     [0],
                                     [0]])
        smf = smith_normal_form(self.m3)
        self.assertTrue((expected_matrix == smf).all())

    def test_smith_normal_form_4(self):
        expected_matrix = np.matrix([[1, 0, 0, 0]])
        smf = smith_normal_form(self.m4)
        self.assertTrue((expected_matrix == smf).all())

    def test_gcd_euclides_1(self):
        expected = 2
        returned = gcd_euclides(2, 4)
        self.assertEqual(expected, returned)

    def test_gcd_euclides_2(self):
        expected = 36
        returned = gcd_euclides(180, 324)
        self.assertEqual(expected, returned)

    def test_matrix_gcd_1(self):
        matrix = [[2, 4, 6], [8, 10, 12], [14, 16, 18]]
        expected = 2
        returned = matrix_gcd(matrix)
        self.assertEqual(expected, returned)

    def test_matrix_gcd_2(self):
        matrix = [[45, 30, 15], [150, 120, 75]]
        expected = 15
        returned = matrix_gcd(matrix)
        self.assertEqual(expected, returned)
