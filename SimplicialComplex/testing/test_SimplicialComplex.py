import json
import os
from pathlib import Path
from unittest import TestCase, mock
from unittest.mock import patch

import numpy as np

from SimplicialComplex.AlphaComplex import AlphaComplex
from SimplicialComplex.SimplicialComplex import SimplicialComplex
from SimplicialComplex.utils.constants import tetraedro, toro, plano_proyectivo, botella_klein

CLOUD_PATH = os.path.join(os.path.abspath(Path(__file__).parent.parent.parent), "docs", "clouds.json")


class TestSimplicialComplex(TestCase):
    # Figura
    sc2 = SimplicialComplex([(0, 1), (1, 2, 3, 4), (4, 5), (5, 6), (4, 6), (6, 7, 8), (8, 9)])

    # Figura no conexa
    sc3 = SimplicialComplex([(0,), (1,), (2, 3), (4, 5), (5, 6), (4, 6), (6, 7, 8, 9)])

    # AlphaComplex
    ac4 = AlphaComplex(np.array(json.load(open(CLOUD_PATH))['alpha1']))

    # Plano Proyectivo
    plano_proyectivo = plano_proyectivo

    # Botella Klein
    botella_klein = botella_klein

    # Simple AlphaComplex
    simple_alpha = AlphaComplex([[-3, 0], [0, 1], [3, 0], [-1.7, -1.8], [1.7, -1.8], [0, -4]])

    def test_add_1(self):
        sc = SimplicialComplex(tetraedro.faces_list())
        sc.add([(9, 10), (10, 11)], 0)
        expected_faces = [(), (0,), (0, 1), (0, 1, 2), (0, 1, 2, 3), (0, 1, 3), (0, 2), (0, 2, 3), (0, 3), (1,), (1, 2),
                          (1, 2, 3), (1, 3), (2,), (2, 3), (3,), (9,), (9, 10), (10,), (10, 11), (11,)]
        self.assertEqual(expected_faces, sc.faces_list())

    def test_add_2(self):
        sc = SimplicialComplex(self.sc2.faces_list())
        sc.add([(10, 11), (11, 12)], 0)
        sc.add([(20, 21)], 0)
        expected_faces = [(), (0,), (0, 1), (1,), (1, 2), (1, 2, 3), (1, 2, 3, 4), (1, 2, 4), (1, 3), (1, 3, 4), (1, 4),
                          (2,), (2, 3), (2, 3, 4), (2, 4), (3,), (3, 4), (4,), (4, 5), (4, 6), (5,), (5, 6), (6,),
                          (6, 7), (6, 7, 8), (6, 8), (7,), (7, 8), (8,), (8, 9), (9,), (10,), (10, 11), (11,), (11, 12),
                          (12,), (20,), (20, 21), (21,)]
        self.assertEqual(expected_faces, sc.faces_list())

    def test_face_set_1(self):
        expected_faces = [(), (0,), (0, 1), (0, 1, 2), (0, 1, 2, 3), (0, 1, 3), (0, 2), (0, 2, 3), (0, 3), (1,), (1, 2),
                          (1, 2, 3), (1, 3), (2,), (2, 3), (3,)]
        self.assertEqual(expected_faces, tetraedro.faces_list())

    def test_face_set_2(self):
        expected_faces = [(), (0,), (0, 1), (1,), (1, 2), (1, 2, 3), (1, 2, 3, 4), (1, 2, 4), (1, 3), (1, 3, 4), (1, 4),
                          (2,), (2, 3), (2, 3, 4), (2, 4), (3,), (3, 4), (4,), (4, 5), (4, 6), (5,), (5, 6), (6,),
                          (6, 7), (6, 7, 8), (6, 8), (7,), (7, 8), (8,), (8, 9), (9,)]
        self.assertEqual(expected_faces, self.sc2.faces_list())

    def test_face_set_3(self):
        expected_faces = [(), (0,), (1,), (2,), (2, 3), (3,), (4,), (4, 5), (4, 6), (5,), (5, 6), (6,), (6, 7),
                          (6, 7, 8), (6, 7, 8, 9), (6, 7, 9), (6, 8), (6, 8, 9), (6, 9), (7,), (7, 8), (7, 8, 9),
                          (7, 9), (8,), (8, 9), (9,)]
        self.assertEqual(expected_faces, self.sc3.faces_list())

    def test_dimension_1(self):
        dimension = tetraedro.dimension()
        self.assertEqual(3, dimension)

    def test_dimension_2(self):
        dimension = self.sc2.dimension()
        self.assertEqual(3, dimension)

    def test_dimension_3(self):
        dimension = self.sc3.dimension()
        self.assertEqual(3, dimension)

    def test_n_faces_1(self):
        nfaces_0 = [(0,), (1,), (2,), (3,)]
        self.assertEqual(nfaces_0, tetraedro.n_faces(0))
        nfaces_1 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        self.assertEqual(nfaces_1, tetraedro.n_faces(1))
        nfaces_2 = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        self.assertEqual(nfaces_2, tetraedro.n_faces(2))
        nfaces_3 = [(0, 1, 2, 3)]
        self.assertEqual(nfaces_3, tetraedro.n_faces(3))

    def test_n_faces_2(self):
        nfaces_0 = [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,)]
        self.assertEqual(nfaces_0, self.sc2.n_faces(0))
        nfaces_1 = [(0, 1), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4), (4, 5), (4, 6), (5, 6), (6, 7), (6, 8),
                    (7, 8), (8, 9)]
        self.assertEqual(nfaces_1, self.sc2.n_faces(1))
        nfaces_2 = [(1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4), (6, 7, 8)]
        self.assertEqual(nfaces_2, self.sc2.n_faces(2))
        nfaces_3 = [(1, 2, 3, 4)]
        self.assertEqual(nfaces_3, self.sc2.n_faces(3))

    def test_star_1(self):
        self.assertEqual([(0, 1), (0, 1, 2), (0, 1, 2, 3), (0, 1, 3)], tetraedro.star((0, 1)))

    def test_star_2(self):
        expected_star = [(1, 2, 3, 4), (1, 2, 4), (1, 3, 4), (1, 4), (2, 3, 4), (2, 4), (3, 4), (4,), (4, 5), (4, 6)]
        self.assertEqual(expected_star, self.sc2.star((4,)))

    def test_closed_star_1(self):
        expectedStar = [(), (0,), (0, 1), (0, 1, 2), (0, 1, 2, 3), (0, 1, 3), (0, 2), (0, 2, 3), (0, 3), (1,), (1, 2),
                        (1, 2, 3), (1, 3), (2,), (2, 3), (3,)]
        self.assertEqual(expectedStar, tetraedro.closed_star((0, 1)))

    def test_closed_star_2(self):
        expectedStar = [(), (1,), (1, 2), (1, 2, 3), (1, 2, 3, 4), (1, 2, 4), (1, 3), (1, 3, 4), (1, 4), (2,), (2, 3),
                        (2, 3, 4), (2, 4), (3,), (3, 4), (4,), (4, 5), (4, 6), (5,), (6,)]
        self.assertEqual(expectedStar, self.sc2.closed_star((4,)))

    def test_link_1(self):
        expected_link = [(), (2,), (2, 3), (3,)]
        self.assertEqual(expected_link, tetraedro.link((0, 1)))

    def test_link_2(self):
        expected_link = [(), (1,), (1, 2), (1, 2, 3), (1, 3), (2,), (2, 3), (3,), (5,), (6,)]
        self.assertEqual(expected_link, self.sc2.link((4,)))

    def test_skeleton_1(self):
        expected_sk_0 = [(), (0,), (1,), (2,), (3,)]
        self.assertEqual(expected_sk_0, tetraedro.skeleton(0))
        expected_sk_1 = [(), (0,), (0, 1), (0, 2), (0, 3), (1,), (1, 2), (1, 3), (2,), (2, 3), (3,)]
        self.assertEqual(expected_sk_1, tetraedro.skeleton(1))
        expected_sk_2 = [(), (0,), (0, 1), (0, 1, 2), (0, 1, 3), (0, 2), (0, 2, 3), (0, 3), (1,), (1, 2), (1, 2, 3),
                         (1, 3), (2,), (2, 3), (3,)]
        self.assertEqual(expected_sk_2, tetraedro.skeleton(2))
        expected_sk_3 = tetraedro.faces_list()
        self.assertEqual(expected_sk_3, tetraedro.skeleton(3))

    def test_skeleton_2(self):
        expected_sk_0 = [(), (0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,)]
        self.assertEqual(expected_sk_0, self.sc2.skeleton(0))
        expected_sk_1 = [(), (0,), (0, 1), (1,), (1, 2), (1, 3), (1, 4), (2,), (2, 3), (2, 4), (3,), (3, 4), (4,),
                         (4, 5), (4, 6), (5,), (5, 6), (6,), (6, 7), (6, 8), (7,), (7, 8), (8,), (8, 9), (9,)]
        self.assertEqual(expected_sk_1, self.sc2.skeleton(1))
        expected_sk_2 = [(), (0,), (0, 1), (1,), (1, 2), (1, 2, 3), (1, 2, 4), (1, 3), (1, 3, 4), (1, 4), (2,), (2, 3),
                         (2, 3, 4), (2, 4), (3,), (3, 4), (4,), (4, 5), (4, 6), (5,), (5, 6), (6,), (6, 7), (6, 7, 8),
                         (6, 8), (7,), (7, 8), (8,), (8, 9), (9,)]
        self.assertEqual(expected_sk_2, self.sc2.skeleton(2))
        expected_sk_3 = self.sc2.faces_list()
        self.assertEqual(expected_sk_3, self.sc2.skeleton(3))

    def test_euler_characteristic_1(self):
        self.assertEqual(1, tetraedro.euler_characteristic())

    def test_euler_characteristic_2(self):
        self.assertEqual(0, self.sc2.euler_characteristic())

    def test_euler_characteristic_3(self):
        self.assertEqual(3, self.sc3.euler_characteristic())

    def test_euler_characteristic_4(self):
        self.assertEqual(1, self.ac4.euler_characteristic())

    def test_connected_components_1(self):
        self.assertEqual(1, tetraedro.connected_components())

    def test_connected_components_2(self):
        self.assertEqual(1, self.sc2.connected_components())

    def test_connected_components_3(self):
        self.assertEqual(4, self.sc3.connected_components())

    def test_connected_components_4(self):
        self.assertEqual(1, self.ac4.connected_components())

    def test_boundarymatrix_1(self):
        expected_bm_0 = [[0, 0, 0, 0]]
        self.assertTrue((expected_bm_0 == tetraedro.boundary_matrix(0, group=2)).all())
        expected_bm_1 = [[1, 1, 1, 0, 0, 0],
                         [1, 0, 0, 1, 1, 0],
                         [0, 1, 0, 1, 0, 1],
                         [0, 0, 1, 0, 1, 1]]
        self.assertTrue((expected_bm_1 == tetraedro.boundary_matrix(1, group=2)).all())
        expected_bm_2 = [[1, 1, 0, 0],
                         [1, 0, 1, 0],
                         [0, 1, 1, 0],
                         [1, 0, 0, 1],
                         [0, 1, 0, 1],
                         [0, 0, 1, 1]]
        self.assertTrue((expected_bm_2 == tetraedro.boundary_matrix(2, group=2)).all())
        expected_bm_3 = [[1],
                         [1],
                         [1],
                         [1]]
        self.assertTrue((expected_bm_3 == tetraedro.boundary_matrix(3, group=2)).all())
        self.assertTrue(([[]] == tetraedro.boundary_matrix(4)).all())

    def test_boundarymatrix_2(self):
        expected_bm_0 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        self.assertTrue((expected_bm_0 == self.sc2.boundary_matrix(0, group=2)).all())
        expected_bm_1 = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
        self.assertTrue((expected_bm_1 == self.sc2.boundary_matrix(1, group=2)).all())
        expected_bm_2 = [[0, 0, 0, 0, 0],
                         [1, 1, 0, 0, 0],
                         [1, 0, 1, 0, 0],
                         [0, 1, 1, 0, 0],
                         [1, 0, 0, 1, 0],
                         [0, 1, 0, 1, 0],
                         [0, 0, 1, 1, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 0]]
        self.assertTrue((expected_bm_2 == self.sc2.boundary_matrix(2, group=2)).all())
        expected_bm_3 = [[1],
                         [1],
                         [1],
                         [1],
                         [0]]
        self.assertTrue((expected_bm_3 == self.sc2.boundary_matrix(3, group=2)).all())
        self.assertTrue(([[]] == self.sc2.boundary_matrix(4)).all())

    def test_generalized_border_matrix(self):
        simple_alpha = AlphaComplex([[-3, 0], [0, 1], [3, 0], [-1.7, -1.8], [1.7, -1.8], [0, -4]])
        matrix = simple_alpha.generalized_boundary_matrix()

        expected_matrix = [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        self.assertTrue((np.array(expected_matrix) == matrix).all())

    def test_boundary_matrix_1(self):
        expected = [[-1, -1, -1, 0, 0, 0],
                    [1, 0, 0, -1, -1, 0],
                    [0, 1, 0, 1, 0, -1],
                    [0, 0, 1, 0, 1, 1]]
        self.assertTrue((np.array(expected) == tetraedro.boundary_matrix(1)).all())

    def test_boundary_matrix_2(self):
        expected = [[1, 1, 0, 0],
                    [-1, 0, 1, 0],
                    [0, -1, -1, 0],
                    [1, 0, 0, 1],
                    [0, 1, 0, -1],
                    [0, 0, 1, 1]]
        self.assertTrue((np.array(expected) == tetraedro.boundary_matrix(2)).all())

    def test_boundary_matrix_3(self):
        expected = [[-1], [1], [-1], [1]]
        self.assertTrue((np.array(expected) == tetraedro.boundary_matrix(3)).all())

    def test_boundary_matrix_4(self):
        sc = SimplicialComplex([(0, 1, 2, 3, 4)])
        expected = [[-1, -1, 0, 0, 0],
                    [1, 0, -1, 0, 0],
                    [0, 1, 1, 0, 0],
                    [-1, 0, 0, -1, 0],
                    [0, -1, 0, 1, 0],
                    [0, 0, -1, -1, 0],
                    [1, 0, 0, 0, -1],
                    [0, 1, 0, 0, 1],
                    [0, 0, 1, 0, -1],
                    [0, 0, 0, 1, 1]]
        self.assertTrue((np.array(expected) == sc.boundary_matrix(3)).all())

    def test_all_betti_numbers_1(self):
        self.assertEqual([1, 0, 0], tetraedro.all_betti_numbers())
        self.assertEqual([1, 0, 0], tetraedro.all_betti_numbers(group=2))
        self.assertEqual([1, 0, 0], tetraedro.all_betti_numbers(group=7))
        self.assertEqual([1, 0, 0], tetraedro.all_betti_numbers(group=11))
        self.assertEqual([1, 0, 0], tetraedro.all_betti_numbers(group='Q'))

    def test_all_betti_numbers_2(self):
        self.assertEqual([1, 1, 0], self.sc2.all_betti_numbers())
        self.assertEqual([1, 1, 0], self.sc2.all_betti_numbers(group=2))
        self.assertEqual([1, 1, 0], self.sc2.all_betti_numbers(group=7))
        self.assertEqual([1, 1, 0], self.sc2.all_betti_numbers(group=11))
        self.assertEqual([1, 1, 0], self.sc2.all_betti_numbers(group='Q'))

    def test_all_betti_numbers_3(self):
        self.assertEqual([1, 0], self.ac4.all_betti_numbers())
        self.assertEqual([1, 0], self.ac4.all_betti_numbers(group=2))
        self.assertEqual([1, 0], self.ac4.all_betti_numbers(group=7))
        self.assertEqual([1, 0], self.ac4.all_betti_numbers(group=11))
        self.assertEqual([1, 0], self.ac4.all_betti_numbers(group='Q'))

    def test_all_betti_numbers_4(self):
        self.assertEqual([1, 2], toro.all_betti_numbers())
        self.assertEqual([1, 2], toro.all_betti_numbers(group=2))
        self.assertEqual([1, 2], toro.all_betti_numbers(group=7))
        self.assertEqual([1, 2], toro.all_betti_numbers(group=11))
        self.assertEqual([1, 2], toro.all_betti_numbers(group='Q'))

    def test_all_betti_numbers_5(self):
        self.assertEqual([1, 0], plano_proyectivo.all_betti_numbers())
        self.assertEqual([1, 1], plano_proyectivo.all_betti_numbers(group=2))
        self.assertEqual([1, 0], plano_proyectivo.all_betti_numbers(group=7))
        self.assertEqual([1, 0], plano_proyectivo.all_betti_numbers(group=11))
        self.assertEqual([1, 0], plano_proyectivo.all_betti_numbers(group='Q'))

    def test_all_betti_numbers_6(self):
        self.assertEqual([1, 1], botella_klein.all_betti_numbers())
        self.assertEqual([1, 2], botella_klein.all_betti_numbers(group=2))
        self.assertEqual([1, 1], botella_klein.all_betti_numbers(group=7))
        self.assertEqual([1, 1], botella_klein.all_betti_numbers(group=11))
        self.assertEqual([1, 1], botella_klein.all_betti_numbers(group='Q'))

    def test_euler_char_and_betti_char_1(self):
        euler = 0
        for i in range(tetraedro.dimension() + 1):
            euler += (-1) ** i * tetraedro.betti_number(i, group=2)
        self.assertEqual(tetraedro.euler_characteristic(), euler)

    def test_euler_char_and_betti_char_2(self):
        euler = 0
        for i in range(self.sc2.dimension() + 1):
            euler += (-1) ** i * self.sc2.betti_number(i, group=2)
        self.assertEqual(self.sc2.euler_characteristic(), euler)

    def test_euler_char_and_betti_char_3(self):
        euler = 0
        for i in range(self.sc3.dimension() + 1):
            euler += (-1) ** i * self.sc3.betti_number(i, group=2)
        self.assertEqual(self.sc3.euler_characteristic(), euler)

    def test_euler_char_and_betti_char_4(self):
        euler = 0
        for i in range(self.ac4.dimension() + 1):
            euler += (-1) ** i * self.ac4.betti_number(i, group=2)
        self.assertEqual(self.ac4.euler_characteristic(), euler)

    def test_euler_char_and_betti_char_5(self):
        euler = 0
        for i in range(toro.dimension() + 1):
            euler += (-1) ** i * toro.betti_number(i, group=2)
        self.assertEqual(toro.euler_characteristic(), euler)

    def test_euler_char_and_betti_char_6(self):
        euler = 0
        for i in range(plano_proyectivo.dimension() + 1):
            euler += (-1) ** i * plano_proyectivo.betti_number(i, group=2)
        self.assertEqual(plano_proyectivo.euler_characteristic(), euler)

    def test_euler_char_and_betti_char_7(self):
        euler = 0
        for i in range(botella_klein.dimension() + 1):
            euler += (-1) ** i * botella_klein.betti_number(i, group=2)
        self.assertEqual(botella_klein.euler_characteristic(), euler)

    def test_homology_cohomology_tetraedro_Z(self):
        self.assertEqual("Z", tetraedro.homology(0))
        self.assertEqual("0", tetraedro.homology(1))
        self.assertEqual("0", tetraedro.homology(2))

        self.assertEqual("Z", tetraedro.cohomology(0))
        self.assertEqual("0", tetraedro.cohomology(1))
        self.assertEqual("0", tetraedro.cohomology(2))

    def test_homology_cohomology_tetraedro_Z2(self):
        self.assertEqual("Z2", tetraedro.homology(0, group=2))
        self.assertEqual("0", tetraedro.homology(1, group=2))
        self.assertEqual("0", tetraedro.homology(2, group=2))

        self.assertEqual("Z2", tetraedro.cohomology(0, group=2))
        self.assertEqual("0", tetraedro.cohomology(1, group=2))
        self.assertEqual("0", tetraedro.cohomology(2, group=2))

    def test_homology_cohomology_tetraedro_Z7(self):
        self.assertEqual("Z7", tetraedro.homology(0, group=7))
        self.assertEqual("0", tetraedro.homology(1, group=7))
        self.assertEqual("0", tetraedro.homology(2, group=7))

        self.assertEqual("Z7", tetraedro.cohomology(0, group=7))
        self.assertEqual("0", tetraedro.cohomology(1, group=7))
        self.assertEqual("0", tetraedro.cohomology(2, group=7))

    def test_homology_cohomology_tetraedro_Q(self):
        self.assertEqual("Q", tetraedro.homology(0, group='Q'))
        self.assertEqual("0", tetraedro.homology(1, group='Q'))
        self.assertEqual("0", tetraedro.homology(2, group='Q'))

        self.assertEqual("Q", tetraedro.cohomology(0, group='Q'))
        self.assertEqual("0", tetraedro.cohomology(1, group='Q'))
        self.assertEqual("0", tetraedro.cohomology(2, group='Q'))

    def test_homology_cohomology_toro_Z(self):
        self.assertEqual("Z", toro.homology(0))
        self.assertEqual("Z²", toro.homology(1))
        self.assertEqual("Z", toro.homology(2))

        self.assertEqual("Z", toro.cohomology(0))
        self.assertEqual("Z²", toro.cohomology(1))
        self.assertEqual("Z", toro.cohomology(2))

    def test_homology_cohomology_toro_Z2(self):
        self.assertEqual("Z2", toro.homology(0, group=2))
        self.assertEqual("Z2²", toro.homology(1, group=2))
        self.assertEqual("Z2", toro.homology(2, group=2))

        self.assertEqual("Z2", toro.cohomology(0, group=2))
        self.assertEqual("Z2²", toro.cohomology(1, group=2))
        self.assertEqual("Z2", toro.cohomology(2, group=2))

    def test_homology_cohomology_toro_Z7(self):
        self.assertEqual("Z7", toro.homology(0, group=7))
        self.assertEqual("Z7²", toro.homology(1, group=7))
        self.assertEqual("Z7", toro.homology(2, group=7))

        self.assertEqual("Z7", toro.cohomology(0, group=7))
        self.assertEqual("Z7²", toro.cohomology(1, group=7))
        self.assertEqual("Z7", toro.cohomology(2, group=7))

    def test_homology_cohomology_toro_Q(self):
        self.assertEqual("Q", toro.homology(0, group='Q'))
        self.assertEqual("Q²", toro.homology(1, group='Q'))
        self.assertEqual("Q", toro.homology(2, group='Q'))

        self.assertEqual("Q", toro.cohomology(0, group='Q'))
        self.assertEqual("Q²", toro.cohomology(1, group='Q'))
        self.assertEqual("Q", toro.cohomology(2, group='Q'))

    def test_homology_cohomology_botella_klein_Z(self):
        self.assertEqual("Z", botella_klein.homology(0))
        self.assertEqual("ZxZ2", botella_klein.homology(1))
        self.assertEqual("0", botella_klein.homology(2))

        self.assertEqual("Z", botella_klein.cohomology(0))
        self.assertEqual("Z", botella_klein.cohomology(1))
        self.assertEqual("Z2", botella_klein.cohomology(2))

    def test_homology_cohomology_botella_klein_Z2(self):
        self.assertEqual("Z2", botella_klein.homology(0, group=2))
        self.assertEqual("Z2²", botella_klein.homology(1, group=2))
        self.assertEqual("Z2", botella_klein.homology(2, group=2))

        self.assertEqual("Z2", botella_klein.cohomology(0, group=2))
        self.assertEqual("Z2²", botella_klein.cohomology(1, group=2))
        self.assertEqual("Z2", botella_klein.cohomology(2, group=2))

    def test_homology_cohomology_botella_klein_Z7(self):
        self.assertEqual("Z7", botella_klein.homology(0, group=7))
        self.assertEqual("Z7", botella_klein.homology(1, group=7))
        self.assertEqual("0", botella_klein.homology(2, group=7))

        self.assertEqual("Z7", botella_klein.cohomology(0, group=7))
        self.assertEqual("Z7", botella_klein.cohomology(1, group=7))
        self.assertEqual("0", botella_klein.cohomology(2, group=7))

    def test_homology_cohomology_botella_klein_Q(self):
        self.assertEqual("Q", botella_klein.homology(0, group='Q'))
        self.assertEqual("Q", botella_klein.homology(1, group='Q'))
        self.assertEqual("0", botella_klein.homology(2, group='Q'))

        self.assertEqual("Q", botella_klein.cohomology(0, group='Q'))
        self.assertEqual("Q", botella_klein.cohomology(1, group='Q'))
        self.assertEqual("0", botella_klein.cohomology(2, group='Q'))

    def test_homology_cohomology_plano_proyectivo_Z(self):
        self.assertEqual("Z", plano_proyectivo.homology(0))
        self.assertEqual("Z2", plano_proyectivo.homology(1))
        self.assertEqual("0", plano_proyectivo.homology(2))

        self.assertEqual("Z", plano_proyectivo.cohomology(0))
        self.assertEqual("0", plano_proyectivo.cohomology(1))
        self.assertEqual("Z2", plano_proyectivo.cohomology(2))

    def test_homology_cohomology_plano_proyectivo_Z2(self):
        self.assertEqual("Z2", plano_proyectivo.homology(0, group=2))
        self.assertEqual("Z2", plano_proyectivo.homology(1, group=2))
        self.assertEqual("Z2", plano_proyectivo.homology(2, group=2))

        self.assertEqual("Z2", plano_proyectivo.cohomology(0, group=2))
        self.assertEqual("Z2", plano_proyectivo.cohomology(1, group=2))
        self.assertEqual("Z2", plano_proyectivo.cohomology(2, group=2))

    def test_homology_cohomology_plano_proyectivo_Z7(self):
        self.assertEqual("Z7", plano_proyectivo.homology(0, group=7))
        self.assertEqual("0", plano_proyectivo.homology(1, group=7))
        self.assertEqual("0", plano_proyectivo.homology(2, group=7))

        self.assertEqual("Z7", plano_proyectivo.cohomology(0, group=7))
        self.assertEqual("0", plano_proyectivo.cohomology(1, group=7))
        self.assertEqual("0", plano_proyectivo.cohomology(2, group=7))

    def test_homology_cohomology_plano_proyectivo_Q(self):
        self.assertEqual("Q", plano_proyectivo.homology(0, group='Q'))
        self.assertEqual("0", plano_proyectivo.homology(1, group='Q'))
        self.assertEqual("0", plano_proyectivo.homology(2, group='Q'))

        self.assertEqual("Q", plano_proyectivo.cohomology(0, group='Q'))
        self.assertEqual("0", plano_proyectivo.cohomology(1, group='Q'))
        self.assertEqual("0", plano_proyectivo.cohomology(2, group='Q'))

    def test_incremental_algth(self):
        sc = SimplicialComplex([(0, 1, 2), (2, 3), (3, 4)])
        self.assertEqual([1, 0], sc.incremental_algth())
        sc = SimplicialComplex([(0, 1), (1, 2), (0, 2), (2, 3), (3, 4)])
        self.assertEqual([1, 1], sc.incremental_algth())
        sc = SimplicialComplex([(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (2, 4)])
        self.assertEqual([1, 2], sc.incremental_algth())

    def test_persistence_diagram(self):
        with patch('matplotlib.pyplot.show') as mocked_show, \
                patch('matplotlib.pyplot.plot') as mocked_plot:
            self.simple_alpha.persistence_diagram()
            self.assertEqual([
                mock.call([0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [1.110180165558726, 1.3901438774457844, 1.5811388300841898, 1.63783393541592, 1.7,
                           2.874107142857142], 'go'),
                mock.call([1.7, 1.63783393541592, 1.5811388300841898],
                          [1.7568181818181818, 1.7164000648001554, 1.916071428571428], 'ro'),
                mock.call([-0.28741071428571424, 3.1615178571428566], [-0.28741071428571424, 3.1615178571428566],
                          'b--'),
                mock.call([-0.28741071428571424, 3.1615178571428566], [2.874107142857142, 2.874107142857142], 'b--')
            ], mocked_plot.mock_calls)

            mocked_show.assert_called_once()

    def test_barcode_diagram(self):
        with patch('matplotlib.pyplot.show') as mocked_show, \
                patch('matplotlib.pyplot.plot') as mocked_plot:
            self.simple_alpha.barcode_diagram()

            self.assertEqual([
                mock.call([0.0, 1.110180165558726], [0, 0], 'g'),
                mock.call([0.0, 1.3901438774457844], [1, 1], 'g'),
                mock.call([0.0, 1.5811388300841898], [2, 2], 'g'),
                mock.call([0.0, 1.63783393541592], [3, 3], 'g'),
                mock.call([0.0, 1.7], [4, 4], 'g'),
                mock.call([0.0, 2.874107142857142], [5, 5], 'g'),
                mock.call([1.7, 1.7568181818181818], [6, 6], 'r'),
                mock.call([1.63783393541592, 1.7164000648001554], [7, 7], 'r'),
                mock.call([1.5811388300841898, 1.916071428571428], [8, 8], 'r')
            ], mocked_plot.mock_calls)

            mocked_show.assert_called_once()

    def test_process_diagram(self):
        infinite, points = self.simple_alpha._compute_diagrams_points()

        self.assertEqual(2.874107142857142, infinite)
        self.assertEqual([[0.0, 1.110180165558726],
                          [0.0, 1.3901438774457844],
                          [0.0, 1.5811388300841898],
                          [0.0, 1.63783393541592],
                          [0.0, 1.7],
                          [0.0, 2.874107142857142]], points[1].tolist())
        self.assertEqual([[1.7, 1.7568181818181818],
                          [1.63783393541592, 1.7164000648001554],
                          [1.5811388300841898, 1.916071428571428]], points[2].tolist())
