from unittest import TestCase

import numpy as np
from fractions import Fraction

from SimplicialComplex.AlphaComplex import AlphaComplex
from SimplicialComplex.utils.matrices_utils import search_non_zero_elem, swap, reconstruct, smith_normal_form, \
    gcd_euclides, matrix_gcd, min_abs_position, smith_normal_form_z, generalized_border_matrix, \
    generalized_border_matrix_algorithm, extended_gcd


class Test(TestCase):
    m1 = np.array([[1, 1, 1, 0, 0, 0],
                   [1, 0, 0, 1, 1, 0],
                   [0, 1, 0, 1, 0, 1],
                   [0, 0, 1, 0, 1, 1]])

    m2 = np.array([[0, 0, 0],
                   [0, 0, 0],
                   [0, 1, 1],
                   [0, 0, 1]])

    m3 = np.array([[0],
                   [0],
                   [1],
                   [1]])

    m4 = np.array([[0, 1, 1, 1]])

    def test_search_one_1(self):
        ret = [0, 0]
        self.assertEqual(ret, search_non_zero_elem(self.m1))

    def test_search_one_2(self):
        ret = [2, 1]
        self.assertEqual(ret, search_non_zero_elem(self.m2))

    def test_search_one_3(self):
        ret = [2, 0]
        self.assertEqual(ret, search_non_zero_elem(self.m3))

    def test_search_one_4(self):
        ret = [0, 1]
        self.assertEqual(ret, search_non_zero_elem(self.m4))

    def test_swap_1(self):
        source = [1, 0]
        obj = [0, 0]
        expected_matrix = np.array([[1, 0, 0, 1, 1, 0],
                                    [1, 1, 1, 0, 0, 0],
                                    [0, 1, 0, 1, 0, 1],
                                    [0, 0, 1, 0, 1, 1]])
        swapped_matrix = swap(self.m1, source, obj)
        self.assertListEqual(expected_matrix.tolist(), swapped_matrix.tolist())

    def test_swap_2(self):
        source = [1, 0]
        obj = [2, 0]
        expected_matrix = np.array([[0],
                                    [1],
                                    [0],
                                    [1]])
        swapped_matrix = swap(self.m3, source, obj)
        self.assertListEqual(expected_matrix.tolist(), swapped_matrix.tolist())

    def test_extended_gcd_1(self):
        a = 60
        b = 48
        gcd, x, y = extended_gcd(a, b)
        self.assertEqual(12, gcd)
        self.assertEqual(12, (a * x) + (b * y))

    def test_extended_gcd_2(self):
        a = 35
        b = 14
        gcd, x, y = extended_gcd(a, b)
        self.assertEqual(7, gcd)
        self.assertEqual(7, (a * x) + (b * y))

    def test_extended_gcd_3(self):
        a = 24
        b = 60
        gcd, x, y = extended_gcd(a, b)
        self.assertEqual(12, gcd)
        self.assertEqual(12, (a * x) + (b * y))

    def test_reconstruct_1(self):
        matrix = np.array([[1, 0, 0, 0],
                           [0, 0, 0, 0]])
        aux = np.array([[0, 0, 0]])
        expected_matrix = np.array([[1, 0, 0, 0],
                                    [0, 0, 0, 0]])
        reconstructed = reconstruct(matrix, aux)
        self.assertListEqual(expected_matrix.tolist(), reconstructed.tolist())

    def test_reconstruct_2(self):
        matrix = np.array([[1, 0, 0, 0, 0],
                           [0, 1, 0, 1, 1],
                           [0, 1, 0, 1, 1]])
        aux = np.array([[1, 0, 0, 0],
                        [0, 0, 0, 0]])
        expected_matrix = np.array([[1, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0]])
        reconstructed = reconstruct(matrix, aux)
        self.assertListEqual(expected_matrix.tolist(), reconstructed.tolist())

    def test_smith_normal_form_Z2_1(self):
        expected_matrix = np.array([[1, 0, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0]])
        expected_rows_matrix = np.array([[1, 0, 0, 0],
                                         [1, 1, 0, 0],
                                         [1, 1, 1, 0],
                                         [1, 1, 1, 1]])
        expected_cols_matrix = np.array([[1, 1, 0, 1, 1, 0],
                                         [0, 1, 1, 1, 0, 1],
                                         [0, 0, 1, 0, 1, 1],
                                         [0, 0, 0, 1, 0, 0],
                                         [0, 0, 0, 0, 1, 0],
                                         [0, 0, 0, 0, 0, 1]])
        smf, rows_matrix, columns_matrix = smith_normal_form(self.m1)

        self.assertListEqual(expected_matrix.tolist(), smf.tolist())
        self.assertEqual(expected_rows_matrix.tolist(), rows_matrix.tolist())
        self.assertEqual(expected_cols_matrix.tolist(), columns_matrix.tolist())
        self.assertEqual(np.matrix(expected_matrix).tolist(),
                         ((np.matrix(expected_rows_matrix) @ np.matrix(self.m1) @
                           np.matrix(expected_cols_matrix)) % 2).tolist())

    def test_smith_normal_form_Z2_2(self):
        expected_matrix = np.array([[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 0],
                                    [0, 0, 0]])
        expected_rows_matrix = np.array([[0, 0, 1, 0],
                                         [0, 0, 0, 1],
                                         [1, 0, 0, 0],
                                         [0, 1, 0, 0]])
        expected_cols_matrix = np.array([[0, 0, 1],
                                         [1, 1, 0],
                                         [0, 1, 0]])

        smf, rows_matrix, columns_matrix = smith_normal_form(self.m2)

        self.assertListEqual(expected_matrix.tolist(), smf.tolist())
        self.assertEqual(expected_rows_matrix.tolist(), rows_matrix.tolist())
        self.assertEqual(expected_cols_matrix.tolist(), columns_matrix.tolist())
        self.assertEqual(np.matrix(expected_matrix).tolist(),
                         ((np.matrix(expected_rows_matrix) @ np.matrix(self.m2) @
                           np.matrix(expected_cols_matrix)) % 2).tolist())

    def test_smith_normal_form_Z2_3(self):
        expected_matrix = np.array([[1],
                                    [0],
                                    [0],
                                    [0]])

        expected_rows_matrix = np.array([[0, 0, 1, 0],
                                         [0, 1, 0, 0],
                                         [1, 0, 0, 0],
                                         [0, 0, 1, 1]])
        expected_cols_matrix = np.array([[1]])

        smf, rows_matrix, columns_matrix = smith_normal_form(self.m3)

        self.assertListEqual(expected_matrix.tolist(), smf.tolist())
        self.assertEqual(expected_rows_matrix.tolist(), rows_matrix.tolist())
        self.assertEqual(expected_cols_matrix.tolist(), columns_matrix.tolist())
        self.assertEqual(np.matrix(expected_matrix).tolist(),
                         ((np.matrix(expected_rows_matrix) @ np.matrix(self.m3) @
                           np.matrix(expected_cols_matrix)) % 2).tolist())

    def test_smith_normal_form_Z2_4(self):
        expected_matrix = np.array([[1, 0, 0, 0]])
        expected_rows_matrix = np.array([[1]])
        expected_cols_matrix = np.array([[0, 1, 0, 0],
                                         [1, 0, 1, 1],
                                         [0, 0, 1, 0],
                                         [0, 0, 0, 1]])

        smf, rows_matrix, columns_matrix = smith_normal_form(self.m4)

        self.assertListEqual(expected_matrix.tolist(), smf.tolist())
        self.assertEqual(expected_rows_matrix.tolist(), rows_matrix.tolist())
        self.assertEqual(expected_cols_matrix.tolist(), columns_matrix.tolist())
        self.assertEqual(np.matrix(expected_matrix).tolist(),
                         ((np.matrix(expected_rows_matrix) @ np.matrix(self.m4) @
                           np.matrix(expected_cols_matrix)) % 2).tolist())

    def test_smith_normal_form_Z7_1(self):
        matrix = np.array([[1, 2, 3],
                           [4, 5, 6]])
        expected_matrix = np.array([[1, 0, 0],
                                    [0, 1, 0]])
        expected_rows_matrix = np.array([[1, 0],
                                         [6, 2]])
        expected_cols_matrix = np.array([[1, 5, 4],
                                         [0, 1, 6],
                                         [0, 0, 4]])

        group = 7
        smf, rows_matrix, columns_matrix = smith_normal_form(matrix, group=group)

        self.assertListEqual(expected_matrix.tolist(), smf.tolist())
        self.assertEqual(expected_rows_matrix.tolist(), rows_matrix.tolist())
        self.assertEqual(expected_cols_matrix.tolist(), columns_matrix.tolist())
        self.assertEqual(np.matrix(expected_matrix).tolist(),
                         ((np.matrix(expected_rows_matrix) @ np.matrix(matrix) @
                           np.matrix(expected_cols_matrix)) % group).tolist())

    def test_smith_normal_form_Z11_1(self):
        matrix = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9],
                           [10, 0, 1]])
        expected_matrix = np.array([[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 0],
                                    [0, 0, 0]])
        expected_rows_matrix = np.array([[1, 0, 0, 0],
                                         [10, 3, 0, 0],
                                         [7, 8, 7, 0],
                                         [2, 8, 0, 1]])
        expected_cols_matrix = np.array([[1, 10, 3],
                                         [0, 6, 5],
                                         [0, 0, 3]])

        group = 11
        smf, rows_matrix, columns_matrix = smith_normal_form(matrix, group=group)

        self.assertListEqual(expected_matrix.tolist(), smf.tolist())
        self.assertEqual(expected_rows_matrix.tolist(), rows_matrix.tolist())
        self.assertEqual(expected_cols_matrix.tolist(), columns_matrix.tolist())
        self.assertEqual(np.matrix(expected_matrix).tolist(),
                         ((np.matrix(expected_rows_matrix) @ np.matrix(matrix) @
                           np.matrix(expected_cols_matrix)) % group).tolist())

    def test_smith_normal_form_Q_1(self):
        matrix = np.array([[Fraction(1, 1), Fraction(2, 1), Fraction(3, 1)],
                           [Fraction(4, 1), Fraction(5, 1), Fraction(6, 1)]])
        expected_matrix = np.array([[Fraction(1, 1), Fraction(0, 1), Fraction(0, 1)],
                                    [Fraction(0, 1), Fraction(1, 1), Fraction(0, 1)]])
        expected_rows_matrix = np.array([[Fraction(1, 1), Fraction(0, 1)],
                                         [Fraction(-1, 1), Fraction(1, 4)]])
        expected_cols_matrix = np.array([[Fraction(1, 1), Fraction(8, 3), Fraction(-2, 3)],
                                         [Fraction(0, 1), Fraction(-4, 3), Fraction(4, 3)],
                                         [Fraction(0, 1), Fraction(0, 1), Fraction(-2, 3)]])

        group = 'Q'
        smf, rows_matrix, columns_matrix = smith_normal_form(matrix, group=group)

        self.assertListEqual(expected_matrix.tolist(), smf.tolist())
        self.assertEqual(expected_rows_matrix.tolist(), rows_matrix.tolist())
        self.assertEqual(expected_cols_matrix.tolist(), columns_matrix.tolist())
        self.assertEqual(np.matrix(expected_matrix).tolist(),
                         ((np.matrix(expected_rows_matrix) @ np.matrix(matrix) @
                           np.matrix(expected_cols_matrix))).tolist())

    def test_smith_normal_form_Q_2(self):
        matrix = np.array([[Fraction(3, 1), Fraction(2, 1)],
                           [Fraction(3, 1), Fraction(5, 1)],
                           [Fraction(4, 1), Fraction(1, 1)]])
        expected_matrix = np.array([[Fraction(1, 1), Fraction(0, 1)],
                                    [Fraction(0, 1), Fraction(1, 1)],
                                    [Fraction(0, 1), Fraction(0, 1)]])
        expected_rows_matrix = np.array([[Fraction(1, 1), Fraction(0, 1), Fraction(0, 1)],
                                         [Fraction(-1, 1), Fraction(1, 1), Fraction(0, 1)],
                                         [Fraction(17, 5), Fraction(-1, 1), Fraction(-9, 5)]])
        expected_cols_matrix = np.array([[Fraction(1, 3), Fraction(-2, 9)],
                                         [Fraction(0, 1), Fraction(1, 3)]])

        group = 'Q'
        smf, rows_matrix, columns_matrix = smith_normal_form(matrix, group=group)

        self.assertListEqual(expected_matrix.tolist(), smf.tolist())
        self.assertEqual(expected_rows_matrix.tolist(), rows_matrix.tolist())
        self.assertEqual(expected_cols_matrix.tolist(), columns_matrix.tolist())
        self.assertEqual(np.matrix(expected_matrix).tolist(),
                         ((np.matrix(expected_rows_matrix) @ np.matrix(matrix) @
                           np.matrix(expected_cols_matrix))).tolist())

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

    def test_matrix_gcd_3(self):
        matrix = [[0, 0, 0], [0, 0, 0]]
        expected = 0
        returned = matrix_gcd(matrix)
        self.assertEqual(expected, returned)

    def test_min_abs_position_1(self):
        matrix = [[0, 30, 45], [150, 120, 0]]
        expected = [0, 1]
        returned = min_abs_position(matrix)
        self.assertEqual(expected, returned)

    def test_min_abs_position_2(self):
        matrix = [[-2, 4, -6], [-8, 0, 12], [0, -16, 18]]
        expected = [0, 0]
        returned = min_abs_position(matrix)
        self.assertEqual(expected, returned)

    def test_smith_normal_form_z_0(self):
        matrix = np.array([[3, 2, 3],
                           [0, 2, 0],
                           [2, 2, 2]])
        expected_matrix = np.array([[1, 0, 0],
                                    [0, 2, 0],
                                    [0, 0, 0]])
        expected_rows_matrix = np.array([[1, 0, 0],
                                         [0, 0, 1],
                                         [2, 1, -3]])
        expected_columns_matrix = np.array([[1, -2, -1],
                                            [-1, 3, 0],
                                            [0, 0, 1]])

        reduced_matrix, rows_opp_matrix, columns_opp_matrix = smith_normal_form_z(matrix)

        self.assertEqual(expected_matrix.tolist(), reduced_matrix.tolist())
        self.assertEqual(expected_rows_matrix.tolist(), rows_opp_matrix.tolist())
        self.assertEqual(expected_columns_matrix.tolist(), columns_opp_matrix.tolist())
        self.assertEqual(np.matrix(expected_matrix).tolist(),
                         (np.matrix(expected_rows_matrix) @ np.matrix(matrix) @
                          np.matrix(expected_columns_matrix)).tolist())

    def test_smith_normal_form_z_00(self):
        matrix = np.array([[1, 1],
                           [1, -1],
                           [-1, 1]])
        expected_matrix = np.array([[1, 0],
                                    [0, 2],
                                    [0, 0]])
        expected_rows_matrix = np.array([[1, 0, 0],
                                         [1, -1, 0],
                                         [0, 1, 1]])
        expected_columns_matrix = np.array([[1, -1],
                                            [0, 1]])

        reduced_matrix, rows_opp_matrix, columns_opp_matrix = smith_normal_form_z(matrix)

        self.assertEqual(expected_matrix.tolist(), reduced_matrix.tolist())
        self.assertEqual(expected_rows_matrix.tolist(), rows_opp_matrix.tolist())
        self.assertEqual(expected_columns_matrix.tolist(), columns_opp_matrix.tolist())
        self.assertEqual(np.matrix(expected_matrix).tolist(),
                         (np.matrix(expected_rows_matrix) @ np.matrix(matrix) @
                          np.matrix(expected_columns_matrix)).tolist())

    def test_smith_normal_form_z_1(self):
        matrix = np.array([[1, 2, 3],
                           [4, 5, 6]])
        expected_matrix = np.array([[1, 0, 0],
                                    [0, 3, 0]])
        expected_rows_matrix = np.array([[1, 0],
                                         [4, -1]])

        expected_columns_matrix = np.array([[1, -2, 1],
                                            [0, 1, -2],
                                            [0, 0, 1]])

        reduced_matrix, rows_opp_matrix, columns_opp_matrix = smith_normal_form_z(matrix)

        self.assertEqual(expected_matrix.tolist(), reduced_matrix.tolist())
        self.assertEqual(expected_rows_matrix.tolist(), rows_opp_matrix.tolist())
        self.assertEqual(expected_columns_matrix.tolist(), columns_opp_matrix.tolist())
        self.assertEqual(np.matrix(expected_matrix).tolist(),
                         (np.matrix(expected_rows_matrix) @ np.matrix(matrix) @
                          np.matrix(expected_columns_matrix)).tolist())

    def test_smith_normal_form_z_2(self):
        matrix = np.array([[2, 4, 4],
                           [-6, 6, 12],
                           [10, -4, -16]])
        expected_matrix = np.array([[2, 0, 0],
                                    [0, 6, 0],
                                    [0, 0, 12]])
        expected_rows_matrix = np.array([[1, 0, 0],
                                         [2, -1, -1],
                                         [3, -4, -3]])
        expected_columns_matrix = np.array([[1, -2, 2],
                                            [0, 1, -2],
                                            [0, 0, 1]])

        reduced_matrix, rows_opp_matrix, columns_opp_matrix = smith_normal_form_z(matrix)

        self.assertEqual(expected_matrix.tolist(), reduced_matrix.tolist())
        self.assertEqual(expected_rows_matrix.tolist(), rows_opp_matrix.tolist())
        self.assertEqual(expected_columns_matrix.tolist(), columns_opp_matrix.tolist())
        self.assertEqual(np.matrix(expected_matrix).tolist(),
                         (np.matrix(expected_rows_matrix) @ np.matrix(matrix) @
                          np.matrix(expected_columns_matrix)).tolist())

    def test_smith_normal_form_z_3(self):
        matrix = np.array([[2, 4, 4],
                           [-6, 6, 12],
                           [10, 4, 16]])
        expected_matrix = np.array([[2, 0, 0],
                                    [0, 2, 0],
                                    [0, 0, 156]])
        expected_rows_matrix = np.array([[1, 0, 0],
                                         [8, 1, -1],
                                         [219, 28, -27]])
        expected_columns_matrix = np.array([[1, 2, -6],
                                            [0, 5, -14],
                                            [0, -6, 17]])

        reduced_matrix, rows_opp_matrix, columns_opp_matrix = smith_normal_form_z(matrix)

        self.assertEqual(expected_matrix.tolist(), reduced_matrix.tolist())
        self.assertEqual(expected_rows_matrix.tolist(), rows_opp_matrix.tolist())
        self.assertEqual(expected_columns_matrix.tolist(), columns_opp_matrix.tolist())
        self.assertEqual(np.matrix(expected_matrix).tolist(),
                         (np.matrix(expected_rows_matrix) @ np.matrix(matrix) @
                          np.matrix(expected_columns_matrix)).tolist())

    def test_smith_normal_form_z_4(self):
        matrix = np.array([[-6, 111, -36, 6],
                           [5, -672, 210, 74],
                           [0, -255, 81, 24],
                           [-7, 255, -81, -10]])
        expected_matrix = np.array([[1, 0, 0, 0],
                                    [0, 3, 0, 0],
                                    [0, 0, 21, 0],
                                    [0, 0, 0, 0]])
        expected_rows_matrix = np.array([[-1, -1, 0, 0],
                                         [-5, -6, 20, 0],
                                         [-85, -130, 419, -20],
                                         [-98, 21, 1, 99]])
        expected_columns_matrix = np.array([[1, -21039, 7026, 2],
                                            [0, -1, 0, 2],
                                            [0, 0, -1, 6],
                                            [0, -270, 90, 1]])

        reduced_matrix, rows_opp_matrix, columns_opp_matrix = smith_normal_form_z(matrix)

        self.assertEqual(expected_matrix.tolist(), reduced_matrix.tolist())
        self.assertEqual(expected_rows_matrix.tolist(), rows_opp_matrix.tolist())
        self.assertEqual(expected_columns_matrix.tolist(), columns_opp_matrix.tolist())
        self.assertEqual(np.matrix(expected_matrix).tolist(),
                         (np.matrix(expected_rows_matrix) @ np.matrix(matrix) @
                          np.matrix(expected_columns_matrix)).tolist())

    def test_smith_normal_form_z_5(self):
        matrix = np.array([[6, -6],
                           [-6, -12],
                           [4, -8]])
        expected_matrix = np.array([[2, 0],
                                    [0, 6],
                                    [0, 0]])
        expected_rows_matrix = np.array([[0, -1, -1],
                                         [-3, -1, 3],
                                         [8, 2, -9]])
        expected_columns_matrix = np.array([[1, -10],
                                            [0, 1]])

        reduced_matrix, rows_opp_matrix, columns_opp_matrix = smith_normal_form_z(matrix)

        self.assertEqual(expected_matrix.tolist(), reduced_matrix.tolist())
        self.assertEqual(expected_rows_matrix.tolist(), rows_opp_matrix.tolist())
        self.assertEqual(expected_columns_matrix.tolist(), columns_opp_matrix.tolist())
        self.assertEqual(np.matrix(expected_matrix).tolist(),
                         (np.matrix(expected_rows_matrix) @ np.matrix(matrix) @
                          np.matrix(expected_columns_matrix)).tolist())

    def test_smith_normal_form_z_6(self):
        matrix = np.array([[1, 2, 1, 1],
                           [2, 0, 1, 2],
                           [3, 2, 2, 3]])
        expected_matrix = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 0, 0]])
        expected_rows_matrix = np.array([[1, 0, 0],
                                         [2, -1, 0],
                                         [-1, -1, 1]])
        expected_columns_matrix = np.array([[1, -1, 2, -1],
                                            [0, 0, 1, 0],
                                            [0, 1, -4, 0],
                                            [0, 0, 0, 1]])

        reduced_matrix, rows_opp_matrix, columns_opp_matrix = smith_normal_form_z(matrix)

        self.assertEqual(expected_matrix.tolist(), reduced_matrix.tolist())
        self.assertEqual(expected_rows_matrix.tolist(), rows_opp_matrix.tolist())
        self.assertEqual(expected_columns_matrix.tolist(), columns_opp_matrix.tolist())
        self.assertEqual(np.matrix(expected_matrix).tolist(),
                         (np.matrix(expected_rows_matrix) @ np.matrix(matrix) @
                          np.matrix(expected_columns_matrix)).tolist())

    def test_smith_normal_form_z_7(self):
        matrix = np.array([[1, 2, 3],
                           [3, -2, 1],
                           [1, 2, 3]])
        expected_matrix = np.array([[1, 0, 0],
                                    [0, 8, 0],
                                    [0, 0, 0]])
        expected_rows_matrix = np.array([[1, 0, 0],
                                         [3, -1, 0],
                                         [-1, 0, 1]])
        expected_columns_matrix = np.array([[1, -2, -1],
                                            [0, 1, -1],
                                            [0, 0, 1]])

        reduced_matrix, rows_opp_matrix, columns_opp_matrix = smith_normal_form_z(matrix)

        self.assertEqual(expected_matrix.tolist(), reduced_matrix.tolist())
        self.assertEqual(expected_rows_matrix.tolist(), rows_opp_matrix.tolist())
        self.assertEqual(expected_columns_matrix.tolist(), columns_opp_matrix.tolist())
        self.assertEqual(np.matrix(expected_matrix).tolist(),
                         (np.matrix(expected_rows_matrix) @ np.matrix(matrix) @
                          np.matrix(expected_columns_matrix)).tolist())

    def test_smith_normal_form_z_8(self):
        matrix = np.array([[1, 0, -1],
                           [4, 3, -1],
                           [0, 9, 3],
                           [3, 12, 3]])
        expected_matrix = np.array([[1, 0, 0],
                                    [0, 3, 0],
                                    [0, 0, 6],
                                    [0, 0, 0]])
        expected_rows_matrix = np.array([[1, 0, 0, 0],
                                         [-4, 1, 0, 0],
                                         [-12, 3, -1, 0],
                                         [1, -1, -1, 1]])
        expected_columns_matrix = np.array([[1, 0, 1],
                                            [0, 1, -1],
                                            [0, 0, 1]])

        reduced_matrix, rows_opp_matrix, columns_opp_matrix = smith_normal_form_z(matrix)

        self.assertEqual(expected_matrix.tolist(), reduced_matrix.tolist())
        self.assertEqual(expected_rows_matrix.tolist(), rows_opp_matrix.tolist())
        self.assertEqual(expected_columns_matrix.tolist(), columns_opp_matrix.tolist())
        self.assertEqual(np.matrix(expected_matrix).tolist(),
                         (np.matrix(expected_rows_matrix) @ np.matrix(matrix) @
                          np.matrix(expected_columns_matrix)).tolist())

    def test_smith_normal_form_z_9(self):
        matrix = np.array([[8, 4, 8],
                           [4, 8, 4]])
        expected_matrix = np.array([[4, 0, 0],
                                    [0, 12, 0]])
        expected_rows_matrix = np.array([[1, 0],
                                         [2, -1]])
        expected_columns_matrix = np.array([[0, 1, -1],
                                            [1, -2, 0],
                                            [0, 0, 1]])

        reduced_matrix, rows_opp_matrix, columns_opp_matrix = smith_normal_form_z(matrix)

        self.assertEqual(expected_matrix.tolist(), reduced_matrix.tolist())
        self.assertEqual(expected_rows_matrix.tolist(), rows_opp_matrix.tolist())
        self.assertEqual(expected_columns_matrix.tolist(), columns_opp_matrix.tolist())
        self.assertEqual(np.matrix(expected_matrix).tolist(),
                         (np.matrix(expected_rows_matrix) @ np.matrix(matrix) @
                          np.matrix(expected_columns_matrix)).tolist())

    def test_smith_normal_form_z_10(self):
        matrix = np.array([[2, 1, -3],
                           [1, -1, 2]])
        expected_matrix = np.array([[1, 0, 0],
                                    [0, 1, 0]])
        expected_rows_matrix = np.array([[1, 0],
                                         [-1, -1]])
        expected_columns_matrix = np.array([[0, 0, 1],
                                            [1, 3, 7],
                                            [0, 1, 3]])

        reduced_matrix, rows_opp_matrix, columns_opp_matrix = smith_normal_form_z(matrix)

        self.assertEqual(expected_matrix.tolist(), reduced_matrix.tolist())
        self.assertEqual(expected_rows_matrix.tolist(), rows_opp_matrix.tolist())
        self.assertEqual(expected_columns_matrix.tolist(), columns_opp_matrix.tolist())
        self.assertEqual(np.matrix(expected_matrix).tolist(),
                         (np.matrix(expected_rows_matrix) @ np.matrix(matrix) @
                          np.matrix(expected_columns_matrix)).tolist())

    def test_smith_normal_form_z_11(self):
        matrix = np.array([[2, 10, 6],
                           [-4, -6, -12],
                           [-2, 4, -6]])
        expected_matrix = np.array([[2, 0, 0],
                                    [0, 14, 0],
                                    [0, 0, 0]])
        expected_rows_matrix = np.array([[1, 0, 0],
                                         [2, 1, 0],
                                         [-1, -1, 1]])
        expected_columns_matrix = np.array([[1, -5, -3],
                                            [0, 1, 0],
                                            [0, 0, 1]])

        reduced_matrix, rows_opp_matrix, columns_opp_matrix = smith_normal_form_z(matrix)

        self.assertEqual(expected_matrix.tolist(), reduced_matrix.tolist())
        self.assertEqual(expected_rows_matrix.tolist(), rows_opp_matrix.tolist())
        self.assertEqual(expected_columns_matrix.tolist(), columns_opp_matrix.tolist())
        self.assertEqual(np.matrix(expected_matrix).tolist(),
                         (np.matrix(expected_rows_matrix) @ np.matrix(matrix) @
                          np.matrix(expected_columns_matrix)).tolist())

    def test_smith_normal_form_z_12(self):
        matrix = np.array([[2, 6, -8],
                           [12, 14, 6],
                           [4, -4, 8]])
        expected_matrix = np.array([[2, 0, 0],
                                    [0, 2, 0],
                                    [0, 0, 168]])
        expected_rows_matrix = np.array([[1, 0, 0],
                                         [10, -3, 4],
                                         [-26, 8, -11]])
        expected_columns_matrix = np.array([[1, -3, -95],
                                            [0, 1, 33],
                                            [0, 0, 1]])

        reduced_matrix, rows_opp_matrix, columns_opp_matrix = smith_normal_form_z(matrix)

        self.assertEqual(expected_matrix.tolist(), reduced_matrix.tolist())
        self.assertEqual(expected_rows_matrix.tolist(), rows_opp_matrix.tolist())
        self.assertEqual(expected_columns_matrix.tolist(), columns_opp_matrix.tolist())
        self.assertEqual(np.matrix(expected_matrix).tolist(),
                         (np.matrix(expected_rows_matrix) @ np.matrix(matrix) @
                          np.matrix(expected_columns_matrix)).tolist())

    def test_smith_normal_form_z_13(self):
        matrix = np.array([[1, -1, 1],
                           [1, 0, 2]])
        expected_matrix = np.array([[1, 0, 0],
                                    [0, 1, 0]])
        expected_rows_matrix = np.array([[1, 0],
                                         [-1, 1]])
        expected_columns_matrix = np.array([[1, 1, -2],
                                            [0, 1, -1],
                                            [0, 0, 1]])

        reduced_matrix, rows_opp_matrix, columns_opp_matrix = smith_normal_form_z(matrix)

        self.assertEqual(expected_matrix.tolist(), reduced_matrix.tolist())
        self.assertEqual(expected_rows_matrix.tolist(), rows_opp_matrix.tolist())
        self.assertEqual(expected_columns_matrix.tolist(), columns_opp_matrix.tolist())
        self.assertEqual(np.matrix(expected_matrix).tolist(),
                         (np.matrix(expected_rows_matrix) @ np.matrix(matrix) @
                          np.matrix(expected_columns_matrix)).tolist())

    def test_smith_normal_form_z_14(self):
        matrix = np.array([[1, 0, -3],
                           [1, 2, 5]])
        expected_matrix = np.array([[1, 0, 0],
                                    [0, 2, 0]])
        expected_rows_matrix = np.array([[1, 0],
                                         [-1, 1]])
        expected_columns_matrix = np.array([[1, 0, 3],
                                            [0, 1, -4],
                                            [0, 0, 1]])

        reduced_matrix, rows_opp_matrix, columns_opp_matrix = smith_normal_form_z(matrix)

        self.assertEqual(expected_matrix.tolist(), reduced_matrix.tolist())
        self.assertEqual(expected_rows_matrix.tolist(), rows_opp_matrix.tolist())
        self.assertEqual(expected_columns_matrix.tolist(), columns_opp_matrix.tolist())
        self.assertEqual(np.matrix(expected_matrix).tolist(),
                         (np.matrix(expected_rows_matrix) @ np.matrix(matrix) @
                          np.matrix(expected_columns_matrix)).tolist())

    def test_smith_normal_form_z_15(self):
        matrix = np.array([[2, -1, 0],
                           [1, -3, 0],
                           [1, 1, 1]])
        expected_matrix = np.array([[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 5]])
        expected_rows_matrix = np.array([[-1, 0, 0],
                                         [1, 0, 1],
                                         [3, -1, 0]])
        expected_columns_matrix = np.array([[0, 0, 1],
                                            [1, 0, 2],
                                            [0, 1, -3]])

        reduced_matrix, rows_opp_matrix, columns_opp_matrix = smith_normal_form_z(matrix)

        self.assertEqual(expected_matrix.tolist(), reduced_matrix.tolist())
        self.assertEqual(expected_rows_matrix.tolist(), rows_opp_matrix.tolist())
        self.assertEqual(expected_columns_matrix.tolist(), columns_opp_matrix.tolist())
        self.assertEqual(np.matrix(expected_matrix).tolist(),
                         (np.matrix(expected_rows_matrix) @ np.matrix(matrix) @
                          np.matrix(expected_columns_matrix)).tolist())

    def test_smith_normal_form_z_16(self):
        matrix = np.array([[1, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0]])
        expected_matrix = np.array([[1, 0, 0, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0]])
        expected_rows_matrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 1, 0, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 1, 0, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 1, 0, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 1, 0],
                                         [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        expected_columns_matrix = np.array([[1, 0, 0, 0, 0, 0, 0],
                                            [0, 1, 0, 0, 0, 0, 0],
                                            [0, 0, 1, 0, 0, 0, 0],
                                            [0, 0, 0, 1, 0, 0, 0],
                                            [0, 0, 0, 0, 1, 0, 0],
                                            [0, 0, 0, 0, 0, 1, 0],
                                            [0, 0, 0, 0, 0, 0, 1]])

        reduced_matrix, rows_opp_matrix, columns_opp_matrix = smith_normal_form_z(matrix)

        self.assertEqual(expected_matrix.tolist(), reduced_matrix.tolist())
        self.assertEqual(expected_rows_matrix.tolist(), rows_opp_matrix.tolist())
        self.assertEqual(expected_columns_matrix.tolist(), columns_opp_matrix.tolist())
        self.assertEqual(np.matrix(expected_matrix).tolist(),
                         (np.matrix(expected_rows_matrix) @ np.matrix(matrix) @
                          np.matrix(expected_columns_matrix)).tolist())

    def test_generalized_border_matrix_algorithm(self):
        simple_alpha = AlphaComplex([[-3, 0], [0, 1], [3, 0], [-1.7, -1.8], [1.7, -1.8], [0, -4]])
        matrix = generalized_border_matrix(simple_alpha.faces)
        reduced_matrix, lows_list = generalized_border_matrix_algorithm(matrix)

        self.assertEqual([[0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], reduced_matrix.tolist())

        self.assertEqual([-1, -1, -1, -1, -1, -1, 3, 4, 5, 2, 1, 1, 1, 1, 2, 12, 13, 14, 11], lows_list)

    def test_generalized_border_matrix(self):
        simple_alpha = AlphaComplex([[-3, 0], [0, 1], [3, 0], [-1.7, -1.8], [1.7, -1.8], [0, -4]])
        matrix = generalized_border_matrix(simple_alpha.faces)

        self.assertListEqual([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], matrix)
