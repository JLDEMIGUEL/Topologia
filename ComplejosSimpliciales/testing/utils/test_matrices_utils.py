from unittest import TestCase

import numpy as np

from ComplejosSimpliciales.src.AlphaComplex import AlphaComplex
from ComplejosSimpliciales.src.utils.matrices_utils import search_one, swap, simplify_rows, simplify_columns, \
    reconstruct, smith_normal_form, gcd_euclides, matrix_gcd, min_abs_position, swap_and_sign, reduce_rows_columns, \
    smith_normal_form_z


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

    def test_simplify_columns(self):
        expected_matrix = np.array([[1, 0, 0, 0, 0, 0],
                                    [1, 1, 1, 1, 1, 0],
                                    [0, 1, 0, 1, 0, 1],
                                    [0, 0, 1, 0, 1, 1]])
        simplified_matrix = simplify_columns(self.m1)
        self.assertListEqual(expected_matrix.tolist(), simplified_matrix.tolist())

    def test_simplify_rows(self):
        expected_matrix = np.array([[1, 1, 1, 0, 0, 0],
                                    [0, 1, 1, 1, 1, 0],
                                    [0, 1, 0, 1, 0, 1],
                                    [0, 0, 1, 0, 1, 1]])
        simplified_matrix = simplify_rows(self.m1)
        self.assertListEqual(expected_matrix.tolist(), simplified_matrix.tolist())

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

    def test_smith_normal_form_1(self):
        expected_matrix = np.array([[1, 0, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0]])
        smf = smith_normal_form(self.m1)
        self.assertListEqual(expected_matrix.tolist(), smf.tolist())

    def test_smith_normal_form_2(self):
        expected_matrix = np.array([[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 0],
                                    [0, 0, 0]])
        smf = smith_normal_form(self.m2)
        self.assertListEqual(expected_matrix.tolist(), smf.tolist())

    def test_smith_normal_form_3(self):
        expected_matrix = np.array([[1],
                                    [0],
                                    [0],
                                    [0]])
        smf = smith_normal_form(self.m3)
        self.assertListEqual(expected_matrix.tolist(), smf.tolist())

    def test_smith_normal_form_4(self):
        expected_matrix = np.array([[1, 0, 0, 0]])
        smf = smith_normal_form(self.m4)
        self.assertListEqual(expected_matrix.tolist(), smf.tolist())

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

    def test_swap_and_sign_1(self):
        source = [1, 0]
        obj = [0, 0]
        matrix = np.array([[1, 1, 1, 0, 0, 0],
                           [-1, 0, 0, 1, 1, 0],
                           [0, 1, 0, 1, 0, 1],
                           [0, 0, 1, 0, 1, 1]])
        expected_matrix = np.array([[1, 0, 0, -1, -1, 0],
                                    [1, 1, 1, 0, 0, 0],
                                    [0, 1, 0, 1, 0, 1],
                                    [0, 0, 1, 0, 1, 1]])
        swapped_matrix = swap_and_sign(matrix, source, obj)
        self.assertListEqual(expected_matrix.tolist(), swapped_matrix.tolist())

    def test_swap_and_sign_2(self):
        source = [1, 0]
        obj = [0, 0]
        matrix = np.array([[1, 1, 1, 0, 0, 0],
                           [1, 0, 0, 1, 1, 0],
                           [0, 1, 0, 1, 0, 1],
                           [0, 0, 1, 0, 1, 1]])
        expected_matrix = np.array([[1, 0, 0, 1, 1, 0],
                                    [1, 1, 1, 0, 0, 0],
                                    [0, 1, 0, 1, 0, 1],
                                    [0, 0, 1, 0, 1, 1]])
        swapped_matrix = swap_and_sign(matrix, source, obj)
        self.assertListEqual(expected_matrix.tolist(), swapped_matrix.tolist())

    def test_reduce_rows_columns(self):
        matrix = np.array([[-2, 4, -6], [-8, 0, 12], [0, -16, 18]])
        expected_matrix = np.array([[-2, 0, 0], [0, -16, 36], [0, -16, 18]])

        reduced_matrix = reduce_rows_columns(matrix)
        self.assertListEqual(expected_matrix.tolist(), reduced_matrix.tolist())

    def test_smith_normal_form_z_1(self):
        matrix = np.array([[1, 2, 3],
                           [4, 5, 6]])
        expected_matrix = np.array([[1, 0, 0],
                                    [0, 3, 0]])

        reduced_matrix = smith_normal_form_z(matrix)
        self.assertEqual(expected_matrix.tolist(), reduced_matrix.tolist())

    def test_smith_normal_form_z_2(self):
        matrix = np.array([[2, 4, 4],
                           [-6, 6, 12],
                           [10, -4, -16]])
        expected_matrix = np.array([[2, 0, 0],
                                    [0, 6, 0],
                                    [0, 0, 12]])

        reduced_matrix = smith_normal_form_z(matrix)
        self.assertEqual(expected_matrix.tolist(), reduced_matrix.tolist())

    def test_smith_normal_form_z_3(self):
        matrix = np.array([[2, 4, 4],
                           [-6, 6, 12],
                           [10, 4, 16]])
        expected_matrix = np.array([[2, 0, 0],
                                    [0, 2, 0],
                                    [0, 0, 156]])

        reduced_matrix = smith_normal_form_z(matrix)
        self.assertEqual(expected_matrix.tolist(), reduced_matrix.tolist())

    def test_smith_normal_form_z_4(self):
        matrix = np.array([[-6, 111, -36, 6],
                           [5, -672, 210, 74],
                           [0, -255, 81, 24],
                           [-7, 255, -81, -10]])
        expected_matrix = np.array([[1, 0, 0, 0],
                                    [0, 3, 0, 0],
                                    [0, 0, 21, 0],
                                    [0, 0, 0, 0]])

        reduced_matrix = smith_normal_form_z(matrix)
        self.assertEqual(expected_matrix.tolist(), reduced_matrix.tolist())

    def test_smith_normal_form_z_5(self):
        matrix = np.array([[6, -6],
                           [-6, -12],
                           [4, -8]])
        expected_matrix = np.array([[2, 0],
                                    [0, 6],
                                    [0, 0]])

        reduced_matrix = smith_normal_form_z(matrix)
        self.assertEqual(expected_matrix.tolist(), reduced_matrix.tolist())

    def test_smith_normal_form_z_6(self):
        matrix = np.array([[1, 2, 1, 1],
                           [2, 0, 1, 2],
                           [3, 2, 2, 3]])
        expected_matrix = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 0, 0]])

        reduced_matrix = smith_normal_form_z(matrix)
        self.assertEqual(expected_matrix.tolist(), reduced_matrix.tolist())

    def test_smith_normal_form_z_7(self):
        matrix = np.array([[1, 2, 3],
                           [3, -2, 1],
                           [1, 2, 3]])
        expected_matrix = np.array([[1, 0, 0],
                                    [0, 8, 0],
                                    [0, 0, 0]])

        reduced_matrix = smith_normal_form_z(matrix)
        self.assertEqual(expected_matrix.tolist(), reduced_matrix.tolist())

    def test_smith_normal_form_z_8(self):
        matrix = np.array([[1, 0, -1],
                           [4, 3, -1],
                           [0, 9, 3],
                           [3, 12, 3]])
        expected_matrix = np.array([[1, 0, 0],
                                    [0, 3, 0],
                                    [0, 0, 6],
                                    [0, 0, 0]])

        reduced_matrix = smith_normal_form_z(matrix)
        self.assertEqual(expected_matrix.tolist(), reduced_matrix.tolist())

    def test_smith_normal_form_z_9(self):
        matrix = np.array([[8, 4, 8],
                           [4, 8, 4]])
        expected_matrix = np.array([[4, 0, 0],
                                    [0, 12, 0]])

        reduced_matrix = smith_normal_form_z(matrix)
        self.assertEqual(expected_matrix.tolist(), reduced_matrix.tolist())

    def test_smith_normal_form_z_10(self):
        matrix = np.array([[2, 1, -3],
                           [1, -1, 2]])
        expected_matrix = np.array([[1, 0, 0],
                                    [0, 1, 0]])

        reduced_matrix = smith_normal_form_z(matrix)
        self.assertEqual(expected_matrix.tolist(), reduced_matrix.tolist())

    def test_smith_normal_form_z_11(self):
        matrix = np.array([[2, 10, 6],
                           [-4, -6, -12],
                           [-2, 4, -6]])
        expected_matrix = np.array([[2, 0, 0],
                                    [0, 14, 0],
                                    [0, 0, 0]])

        reduced_matrix = smith_normal_form_z(matrix)
        self.assertEqual(expected_matrix.tolist(), reduced_matrix.tolist())

    def test_smith_normal_form_z_12(self):
        matrix = np.array([[2, 6, -8],
                           [12, 14, 6],
                           [4, -4, 8]])
        expected_matrix = np.array([[2, 0, 0],
                                    [0, 2, 0],
                                    [0, 0, 168]])

        reduced_matrix = smith_normal_form_z(matrix)
        self.assertEqual(expected_matrix.tolist(), reduced_matrix.tolist())

    def test_smith_normal_form_z_13(self):
        matrix = np.array([[1, -1, 1],
                           [1, 0, 2]])
        expected_matrix = np.array([[1, 0, 0],
                                    [0, 1, 0]])

        reduced_matrix = smith_normal_form_z(matrix)
        self.assertEqual(expected_matrix.tolist(), reduced_matrix.tolist())

    def test_smith_normal_form_z_14(self):
        matrix = np.array([[1, 0, -3],
                           [1, 2, 5]])
        expected_matrix = np.array([[1, 0, 0],
                                    [0, 2, 0]])

        reduced_matrix = smith_normal_form_z(matrix)
        self.assertEqual(expected_matrix.tolist(), reduced_matrix.tolist())

    def test_smith_normal_form_z_15(self):
        matrix = np.array([[2, -1, 0],
                           [1, -3, 0],
                           [1, 1, 1]])
        expected_matrix = np.array([[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 5]])

        reduced_matrix = smith_normal_form_z(matrix)
        self.assertEqual(expected_matrix.tolist(), reduced_matrix.tolist())

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

        reduced_matrix = smith_normal_form_z(matrix)
        self.assertListEqual(expected_matrix.tolist(), reduced_matrix.tolist())

    def test_generalized_border_matrix_algorithm(self):
        simple_alpha = AlphaComplex([[-3, 0], [0, 1], [3, 0], [-1.7, -1.8], [1.7, -1.8], [0, -4]])
        matrix = generalized_border_matrix(simple_alpha.dic)
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
        matrix = generalized_border_matrix(simple_alpha.dic)

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
