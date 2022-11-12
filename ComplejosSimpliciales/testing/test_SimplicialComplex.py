from unittest import TestCase

import numpy as np

from ComplejosSimpliciales.src.AlphaComplex import AlphaComplex
from ComplejosSimpliciales.src.SimplicialComplex import SimplicialComplex


class TestSimplicialComplex(TestCase):
    # Tetraedro
    sc1 = SimplicialComplex([(0, 1, 2, 3)])

    # Figura
    sc2 = SimplicialComplex([(0, 1), (1, 2, 3, 4), (4, 5), (5, 6), (4, 6), (6, 7, 8), (8, 9)])

    # Figura no conexa
    sc3 = SimplicialComplex([(0,), (1,), (2, 3), (4, 5), (5, 6), (4, 6), (6, 7, 8, 9)])

    # AlphaComplex
    ac4 = AlphaComplex(np.array([(0.38021546727456423, 0.46419202339598786), (0.7951628297672293, 0.49263630135869474),
                                 (0.566623772375203, 0.038325621649018426), (0.3369306814864865, 0.7103735061134965),
                                 (0.08272837815822842, 0.2263273314352896), (0.5180166301873989, 0.6271769943824689),
                                 (0.33691411899985035, 0.8402045183219995), (0.33244488399729255, 0.4524636520475205),
                                 (0.11778991601260325, 0.6657734204021165), (0.9384303415747769, 0.2313873874340855)]))

    def test_basic_1(self):
        aux = SimplicialComplex(self.sc1.face_set())
        self.assertEqual(aux.faces, aux.dic.keys())
        aux.add([(1000, 1001, 1002), (2000,)], 0)
        self.assertEqual(aux.faces, aux.dic.keys())
        aux.add([(3000, 3001), (4000, 4001)], 0)
        self.assertEqual(aux.faces, aux.dic.keys())
        aux.add([], 0)
        self.assertEqual(aux.faces, aux.dic.keys())

    def test_basic_2(self):
        aux = SimplicialComplex(self.sc2.face_set())
        self.assertEqual(aux.faces, aux.dic.keys())
        aux.add([(1000, 1001, 1002), (2000,)], 0)
        self.assertEqual(aux.faces, aux.dic.keys())
        aux.add([(3000, 3001), (4000, 4001)], 0)
        self.assertEqual(aux.faces, aux.dic.keys())
        aux.add([], 0)
        self.assertEqual(aux.faces, aux.dic.keys())

    def test_basic_3(self):
        aux = SimplicialComplex(self.sc3.face_set())
        self.assertEqual(aux.faces, aux.dic.keys())
        aux.add([(1000, 1001, 1002), (2000,)], 0)
        self.assertEqual(aux.faces, aux.dic.keys())
        aux.add([(3000, 3001), (4000, 4001)], 0)
        self.assertEqual(aux.faces, aux.dic.keys())
        aux.add([], 0)
        self.assertEqual(aux.faces, aux.dic.keys())

    def test_basic_4(self):
        aux = AlphaComplex(self.ac4.tri.points)
        self.assertEqual(aux.faces, aux.dic.keys())
        aux.add([(1000, 1001, 1002), (2000,)], 0)
        self.assertEqual(aux.faces, aux.dic.keys())
        aux.add([(3000, 3001), (4000, 4001)], 0)
        self.assertEqual(aux.faces, aux.dic.keys())
        aux.add([], 0)
        self.assertEqual(aux.faces, aux.dic.keys())

    def test_add_1(self):
        sc = SimplicialComplex(self.sc1.face_set())
        sc.add([(9, 10), (10, 11)], 0)
        expected_faces = [(), (0,), (0, 1), (0, 1, 2), (0, 1, 2, 3), (0, 1, 3), (0, 2), (0, 2, 3), (0, 3), (1,), (1, 2),
                          (1, 2, 3), (1, 3), (2,), (2, 3), (3,), (9,), (9, 10), (10,), (10, 11), (11,)]
        self.assertEqual(expected_faces, sc.face_set())

    def test_add_2(self):
        sc = SimplicialComplex(self.sc2.face_set())
        sc.add([(10, 11), (11, 12)], 0)
        sc.add([(20, 21)], 0)
        expected_faces = [(), (0,), (0, 1), (1,), (1, 2), (1, 2, 3), (1, 2, 3, 4), (1, 2, 4), (1, 3), (1, 3, 4), (1, 4),
                          (2,), (2, 3), (2, 3, 4), (2, 4), (3,), (3, 4), (4,), (4, 5), (4, 6), (5,), (5, 6), (6,),
                          (6, 7), (6, 7, 8), (6, 8), (7,), (7, 8), (8,), (8, 9), (9,), (10,), (10, 11), (11,), (11, 12),
                          (12,), (20,), (20, 21), (21,)]
        self.assertEqual(expected_faces, sc.face_set())

    def test_face_set_1(self):
        expected_faces = [(), (0,), (0, 1), (0, 1, 2), (0, 1, 2, 3), (0, 1, 3), (0, 2), (0, 2, 3), (0, 3), (1,), (1, 2),
                          (1, 2, 3), (1, 3), (2,), (2, 3), (3,)]
        self.assertEqual(expected_faces, self.sc1.face_set())

    def test_face_set_2(self):
        expected_faces = [(), (0,), (0, 1), (1,), (1, 2), (1, 2, 3), (1, 2, 3, 4), (1, 2, 4), (1, 3), (1, 3, 4), (1, 4),
                          (2,), (2, 3), (2, 3, 4), (2, 4), (3,), (3, 4), (4,), (4, 5), (4, 6), (5,), (5, 6), (6,),
                          (6, 7), (6, 7, 8), (6, 8), (7,), (7, 8), (8,), (8, 9), (9,)]
        self.assertEquals(expected_faces, self.sc2.face_set())

    def test_face_set_3(self):
        expected_faces = [(), (0,), (1,), (2,), (2, 3), (3,), (4,), (4, 5), (4, 6), (5,), (5, 6), (6,), (6, 7),
                          (6, 7, 8), (6, 7, 8, 9), (6, 7, 9), (6, 8), (6, 8, 9), (6, 9), (7,), (7, 8), (7, 8, 9),
                          (7, 9), (8,), (8, 9), (9,)]
        self.assertEquals(expected_faces, self.sc3.face_set())

    def test_dimension_1(self):
        dimension = self.sc1.dimension()
        self.assertEqual(3, dimension)

    def test_dimension_2(self):
        dimension = self.sc2.dimension()
        self.assertEqual(3, dimension)

    def test_dimension_3(self):
        dimension = self.sc3.dimension()
        self.assertEqual(3, dimension)

    def test_n_faces_1(self):
        nfaces_0 = [(0,), (1,), (2,), (3,)]
        self.assertEqual(nfaces_0, self.sc1.n_faces(0))
        nfaces_1 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        self.assertEqual(nfaces_1, self.sc1.n_faces(1))
        nfaces_2 = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        self.assertEqual(nfaces_2, self.sc1.n_faces(2))
        nfaces_3 = [(0, 1, 2, 3)]
        self.assertEqual(nfaces_3, self.sc1.n_faces(3))

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
        self.assertEqual([(0, 1), (0, 1, 2), (0, 1, 2, 3), (0, 1, 3)], self.sc1.star((0, 1)))

    def test_star_2(self):
        expected_star = [(1, 2, 3, 4), (1, 2, 4), (1, 3, 4), (1, 4), (2, 3, 4), (2, 4), (3, 4), (4,), (4, 5), (4, 6)]
        self.assertEqual(expected_star, self.sc2.star((4,)))

    def test_closed_star_1(self):
        expectedStar = [(), (0,), (0, 1), (0, 1, 2), (0, 1, 2, 3), (0, 1, 3), (0, 2), (0, 2, 3), (0, 3), (1,), (1, 2),
                        (1, 2, 3), (1, 3), (2,), (2, 3), (3,)]
        self.assertEqual(expectedStar, self.sc1.closedStar((0, 1)))

    def test_closed_star_2(self):
        expectedStar = [(), (1,), (1, 2), (1, 2, 3), (1, 2, 3, 4), (1, 2, 4), (1, 3), (1, 3, 4), (1, 4), (2,), (2, 3),
                        (2, 3, 4), (2, 4), (3,), (3, 4), (4,), (4, 5), (4, 6), (5,), (6,)]
        self.assertEqual(expectedStar, self.sc2.closedStar((4,)))

    def test_link_1(self):
        expected_link = [(), (2,), (2, 3), (3,)]
        self.assertEqual(expected_link, self.sc1.link((0, 1)))

    def test_link_2(self):
        expected_link = [(), (1,), (1, 2), (1, 2, 3), (1, 3), (2,), (2, 3), (3,), (5,), (6,)]
        self.assertEqual(expected_link, self.sc2.link((4,)))

    def test_skeleton_1(self):
        expected_sk_0 = [(), (0,), (1,), (2,), (3,)]
        self.assertEqual(expected_sk_0, self.sc1.skeleton(0))
        expected_sk_1 = [(), (0,), (0, 1), (0, 2), (0, 3), (1,), (1, 2), (1, 3), (2,), (2, 3), (3,)]
        self.assertEqual(expected_sk_1, self.sc1.skeleton(1))
        expected_sk_2 = [(), (0,), (0, 1), (0, 1, 2), (0, 1, 3), (0, 2), (0, 2, 3), (0, 3), (1,), (1, 2), (1, 2, 3),
                         (1, 3), (2,), (2, 3), (3,)]
        self.assertEqual(expected_sk_2, self.sc1.skeleton(2))
        expected_sk_3 = self.sc1.face_set()
        self.assertEqual(expected_sk_3, self.sc1.skeleton(3))

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
        expected_sk_3 = self.sc2.face_set()
        self.assertEqual(expected_sk_3, self.sc2.skeleton(3))

    def test_euler_characteristic_1(self):
        self.assertEqual(1, self.sc1.euler_characteristic())

    def test_euler_characteristic_2(self):
        self.assertEqual(0, self.sc2.euler_characteristic())

    def test_euler_characteristic_3(self):
        self.assertEqual(3, self.sc3.euler_characteristic())

    def test_euler_characteristic_4(self):
        self.assertEqual(1, self.ac4.euler_characteristic())

    def test_connected_components_1(self):
        self.assertEqual(1, self.sc1.connected_components())

    def test_connected_components_2(self):
        self.assertEqual(1, self.sc2.connected_components())

    def test_connected_components_3(self):
        self.assertEqual(4, self.sc3.connected_components())

    def test_connected_components_4(self):
        self.assertEqual(1, self.ac4.connected_components())

    def test_boundarymatrix_1(self):
        expected_bm_0 = [[0, 0, 0, 0]]
        self.assertTrue((expected_bm_0 == self.sc1.boundarymatrix(0)).all())
        expected_bm_1 = [[1, 1, 1, 0, 0, 0],
                         [1, 0, 0, 1, 1, 0],
                         [0, 1, 0, 1, 0, 1],
                         [0, 0, 1, 0, 1, 1]]
        self.assertTrue((expected_bm_1 == self.sc1.boundarymatrix(1)).all())
        expected_bm_2 = [[1, 1, 0, 0],
                         [1, 0, 1, 0],
                         [0, 1, 1, 0],
                         [1, 0, 0, 1],
                         [0, 1, 0, 1],
                         [0, 0, 1, 1]]
        self.assertTrue((expected_bm_2 == self.sc1.boundarymatrix(2)).all())
        expected_bm_3 = [[1],
                         [1],
                         [1],
                         [1]]
        self.assertTrue((expected_bm_3 == self.sc1.boundarymatrix(3)).all())
        self.assertTrue(([[]] == self.sc1.boundarymatrix(4)).all())

    def test_boundarymatrix_2(self):
        expected_bm_0 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        self.assertTrue((expected_bm_0 == self.sc2.boundarymatrix(0)).all())
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
        self.assertTrue((expected_bm_1 == self.sc2.boundarymatrix(1)).all())
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
        self.assertTrue((expected_bm_2 == self.sc2.boundarymatrix(2)).all())
        expected_bm_3 = [[1],
                         [1],
                         [1],
                         [1],
                         [0]]
        self.assertTrue((expected_bm_3 == self.sc2.boundarymatrix(3)).all())
        self.assertTrue(([[]] == self.sc2.boundarymatrix(4)).all())

    def test_betti_number_1(self):
        self.assertEqual(1, self.sc1.betti_number(0))
        self.assertEqual(0, self.sc1.betti_number(1))
        self.assertEqual(0, self.sc1.betti_number(2))
        self.assertEqual(0, self.sc1.betti_number(3))

    def test_betti_number_2(self):
        self.assertEqual(1, self.sc2.betti_number(0))
        self.assertEqual(1, self.sc2.betti_number(1))
        self.assertEqual(0, self.sc2.betti_number(2))
        self.assertEqual(0, self.sc2.betti_number(3))
