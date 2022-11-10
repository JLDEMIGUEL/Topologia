from unittest import TestCase

from ComplejosSimpliciales.src.SimplicialComplex import SimplicialComplex


class TestSimplicialComplex(TestCase):
    sc1 = SimplicialComplex([(0, 1, 2, 3)])

    def test_add_1(self):
        sc = SimplicialComplex(self.sc1.face_set())
        sc.add([(9, 10), (10, 11)], 0)
        expected_faces = [(), (0,), (0, 1), (0, 1, 2), (0, 1, 2, 3), (0, 1, 3), (0, 2), (0, 2, 3), (0, 3), (1,), (1, 2),
                          (1, 2, 3), (1, 3), (2,), (2, 3), (3,), (9,), (9, 10), (10,), (10, 11), (11,)]
        self.assertEqual(expected_faces, sc.face_set())

    def test_face_set_1(self):
        expected_faces = [(), (0,), (0, 1), (0, 1, 2), (0, 1, 2, 3), (0, 1, 3), (0, 2), (0, 2, 3), (0, 3), (1,), (1, 2),
                          (1, 2, 3), (1, 3), (2,), (2, 3), (3,)]
        self.assertEquals(expected_faces, self.sc1.face_set())

    def test_dimension_1(self):
        dimension = self.sc1.dimension()
        self.assertEqual(3, dimension, f"Dimension should be 3, not {dimension}")

    def test_n_faces_1(self):
        nfaces_0 = self.sc1.n_faces(0)
        self.assertEqual([(0,), (1,), (2,), (3,)], nfaces_0)
        nfaces_1 = self.sc1.n_faces(1)
        self.assertEqual([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], nfaces_1)
        nfaces_2 = self.sc1.n_faces(2)
        self.assertEqual([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)], nfaces_2)
        nfaces_3 = self.sc1.n_faces(3)
        self.assertEqual([(0, 1, 2, 3)], nfaces_3)

    def test_star_1(self):
        self.assertEqual([(0, 1), (0, 1, 2), (0, 1, 2, 3), (0, 1, 3)], self.sc1.star((0, 1)))

    def test_closed_star_1(self):
        expectedStar = [(), (0,), (0, 1), (0, 1, 2), (0, 1, 2, 3), (0, 1, 3), (0, 2), (0, 2, 3), (0, 3), (1,), (1, 2),
                        (1, 2, 3), (1, 3), (2,), (2, 3), (3,)]
        self.assertEqual(expectedStar, self.sc1.closedStar((0, 1)))

    def test_link_1(self):
        self.assertEqual([(), (2,), (2, 3), (3,)], self.sc1.link((0, 1)))

    def test_skeleton(self):
        expected_sk_0 = [(), (0,), (1,), (2,), (3,)]
        self.assertEqual(expected_sk_0, self.sc1.skeleton(0))
        expected_sk_1 = [(), (0,), (0, 1), (0, 2), (0, 3), (1,), (1, 2), (1, 3), (2,), (2, 3), (3,)]
        self.assertEqual(expected_sk_1, self.sc1.skeleton(1))
        expected_sk_2 = [(), (0,), (0, 1), (0, 1, 2), (0, 1, 3), (0, 2), (0, 2, 3), (0, 3), (1,), (1, 2), (1, 2, 3),
                         (1, 3), (2,), (2, 3), (3,)]
        self.assertEqual(expected_sk_2, self.sc1.skeleton(2))
        expected_sk_3 = self.sc1.face_set()
        self.assertEqual(expected_sk_3, self.sc1.skeleton(3))

    def test_euler_characteristic_1(self):
        self.assertEqual(1, self.sc1.euler_characteristic())

    def test_connected_components_1(self):
        self.assertEqual(1, self.sc1.connected_components())

    def test_boundarymatrix_1(self):
        expected_bm_0 = [[0, 0, 0, 0]]
        self.assertEqual(expected_bm_0, self.sc1.boundarymatrix(0))
        expected_bm_1 = [[1, 1, 1, 0, 0, 0],
                         [1, 0, 0, 1, 1, 0],
                         [0, 1, 0, 1, 0, 1],
                         [0, 0, 1, 0, 1, 1]]
        self.assertEqual(expected_bm_1, self.sc1.boundarymatrix(1))
        expected_bm_2 = [[1, 1, 0, 0],
                         [1, 0, 1, 0],
                         [0, 1, 1, 0],
                         [1, 0, 0, 1],
                         [0, 1, 0, 1],
                         [0, 0, 1, 1]]
        self.assertEqual(expected_bm_2, self.sc1.boundarymatrix(2))
        expected_bm_3 = [[1],
                         [1],
                         [1],
                         [1]]
        self.assertEqual(expected_bm_3, self.sc1.boundarymatrix(3))
        self.assertEqual([[]], self.sc1.boundarymatrix(4))

    def test_betti_number(self):
        self.assertEqual(1, self.sc1.betti_number(0))
        self.assertEqual(0, self.sc1.betti_number(1))
        self.assertEqual(0, self.sc1.betti_number(2))
        self.assertEqual(0, self.sc1.betti_number(3))
