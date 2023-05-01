import math
from unittest import TestCase, mock
from unittest.mock import patch

import numpy as np

from SimplicialComplex.AlphaComplex import AlphaComplex
from SimplicialComplex.utils.simplicial_complex_utils import order, reachable, sub_faces, updateDict, \
    order_faces, filter_by_float, noise, connected_components, reachable_alg, num_loops, calc_homology, num_triangles, \
    check_if_sub_face, boundary_operator


class Test(TestCase):
    def test_order_1(self):
        input_list = [(0, 1), (0,), (100, 1000)]
        expected_output = [(0,), (0, 1), (100, 1000)]
        self.assertEqual(expected_output, order(input_list))

    def test_order_2(self):
        input_list = [(0, 100), (100,), (0, 99)]
        expected_output = [(0, 99), (0, 100), (100,)]
        self.assertEqual(expected_output, order(input_list))

    def test_reachable_1(self):
        edges = [(0, 1), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4), (4, 5), (4, 6), (5, 6), (6, 7), (6, 8), (7, 8),
                 (8, 9)]
        vert = 0
        visitedVertex = {0: False, 1: False, 2: False, 3: False, 4: False, 5: False, 6: False, 7: False, 8: False,
                         9: False}
        reachable_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.assertEqual(reachable_list, reachable(edges, vert, visitedVertex))

    def test_reachable_2(self):
        edges = [(2, 3), (4, 5), (4, 6), (5, 6), (6, 7), (6, 8), (6, 9), (7, 8), (7, 9), (8, 9)]
        vert = 0
        visitedVertex = {0: False, 1: False, 2: False, 3: False, 4: False, 5: False, 6: False, 7: False, 8: False,
                         9: False}
        reachable_list = [0]
        self.assertEqual(reachable_list, reachable(edges, vert, visitedVertex))

    def test_reachable_3(self):
        edges = [(2, 3), (4, 5), (4, 6), (5, 6), (6, 7), (6, 8), (6, 9), (7, 8), (7, 9), (8, 9)]
        vert = 4
        visitedVertex = {0: True, 1: True, 2: True, 3: True, 4: False, 5: False, 6: False, 7: False, 8: False, 9: False}
        reachable_list = [4, 5, 6, 7, 8, 9]
        self.assertEqual(reachable_list, reachable(edges, vert, visitedVertex))

    def test_sub_faces_1(self):
        expected_subfaces = {(), (0,), (0, 1), (0, 1, 2), (0, 1, 3), (0, 2), (0, 2, 3), (0, 3), (1,), (1, 2), (1, 2, 3),
                             (1, 3), (2,), (2, 3), (3,)}
        self.assertEqual(expected_subfaces, sub_faces((0, 1, 2, 3)))

    def test_update_dict_2(self):
        dic = {(6,): 0, (2,): 0, (5,): 0, (8,): 0, (4,): 0, (1,): 0, (7,): 0, (0,): 0, (): 0, (3,): 0, (9,): 0,
               (0, 2): 0.23243829737556373, (0, 3): 0.12497889333174499, (0, 5): 0.1067160499498478,
               (0, 7): 0.024594630110749274, (1, 2): 0.25427755231828075, (1, 5): 0.15403831830169779,
               (1, 9): 0.14897699064479192, (2, 4): 0.259566655903143, (2, 9): 0.20947134070559964,
               (3, 5): 0.09964158316325628, (3, 6): 0.06491550663246884, (3, 8): 0.11181663867459748,
               (4, 7): 0.16844581383081997, (4, 8): 0.2204212878472461, (7, 8): 0.151309131049649,
               (0, 1): 0.2146289409772831, (0, 1, 2): 0.27011640994552, (0, 1, 5): 0.2146289409772831,
               (0, 2, 7): 0.24143718415372575, (2, 7): 0.24143718415372575, (0, 3, 5): 0.129720050879878}
        faces = [(), (0,), (0, 3), (0, 3, 7), (0, 7), (3,), (3, 7), (7,)]
        float_value = 0.12926191918239155
        expected_dic = {(6,): 0, (2,): 0, (5,): 0, (8,): 0, (4,): 0, (1,): 0, (7,): 0, (0,): 0, (): 0, (3,): 0, (9,): 0,
                        (0, 2): 0.23243829737556373, (0, 3): 0.12497889333174499, (0, 5): 0.1067160499498478,
                        (0, 7): 0.024594630110749274, (1, 2): 0.25427755231828075, (1, 5): 0.15403831830169779,
                        (1, 9): 0.14897699064479192, (2, 4): 0.259566655903143, (2, 9): 0.20947134070559964,
                        (3, 5): 0.09964158316325628, (3, 6): 0.06491550663246884, (3, 8): 0.11181663867459748,
                        (4, 7): 0.16844581383081997, (4, 8): 0.2204212878472461, (7, 8): 0.151309131049649,
                        (0, 1): 0.2146289409772831, (0, 1, 2): 0.27011640994552, (0, 1, 5): 0.2146289409772831,
                        (0, 2, 7): 0.24143718415372575, (2, 7): 0.24143718415372575, (0, 3, 5): 0.129720050879878,
                        (0, 3, 7): 0.12926191918239155, (3, 7): 0.12926191918239155}
        self.assertEqual(expected_dic, updateDict(dic, faces, float_value))

    def test_order_faces(self):
        unsorted_faces = {(3, 2, 1), (4, 6, 5), (2, 1), (1,), (8, 9)}
        sorted_faces = {(1, 2, 3), (4, 5, 6), (1, 2), (1,), (8, 9)}
        self.assertEqual(sorted_faces, order_faces(unsorted_faces))

    def test_filter_by_float(self):
        dic = {(2,): 0, (5,): 0, (11,): 0, (8,): 0, (14,): 0, (): 0, (17,): 0, (0,): 0, (3,): 0, (9,): 0, (6,): 0,
               (12,): 0, (18,): 0, (15,): 0, (4,): 0, (1,): 0, (7,): 0, (10,): 0, (16,): 0, (13,): 0, (19,): 0,
               (0, 7): 2.655928521333648, (0, 17): 1.7305797293871168, (1, 9): 0.3594476020513678,
               (1, 15): 0.6579110269255046, (1, 19): 3.075420599920237, (2, 11): 0.8984512866459233,
               (2, 13): 1.018845498134008, (2, 19): 1.0663342997829408, (3, 11): 1.7631900830583442,
               (3, 12): 1.8739871984415204, (3, 19): 1.7105016392899866, (4, 9): 1.0926162823459045,
               (4, 12): 0.9182977742772253, (5, 14): 0.7725554426347773, (5, 16): 1.3897471697859505,
               (5, 18): 1.3421291156269257, (6, 8): 1.0835012924474037, (6, 10): 0.6337392347297187,
               (6, 15): 0.7473705160649131, (7, 16): 0.10565789180691254, (8, 14): 1.3441831248867206,
               (8, 15): 1.0085045640495482, (8, 18): 1.6849626346923738, (9, 15): 0.6811155450226897,
               (11, 19): 1.0097528017787816, (13, 14): 3.81894308488309, (13, 17): 0.512709492652287,
               (13, 19): 1.474790175830562, (0, 7, 13): 3.3474070826562006, (0, 13): 2.0820257648936034,
               (7, 13): 3.3474070826562006, (0, 7, 16): 36.756608113059094, (0, 16): 36.756608113059094,
               (0, 13, 17): 2.0820257648936034, (1, 4): 1.4771154526024748, (1, 4, 9): 1.4771154526024748,
               (1, 4, 12): 2.1335293172038012, (1, 12): 2.1335293172038012, (1, 9, 15): 0.6962666627950436,
               (1, 12, 19): 3.0897352223305905, (12, 19): 2.6101619429572755, (1, 13): 3.8800339027875133,
               (1, 13, 14): 3.9262271123430863, (1, 14): 2.146759732709587, (1, 13, 19): 3.8800339027875133,
               (1, 14, 15): 2.146759732709587, (14, 15): 1.7001972603943867, (2, 11, 19): 1.152885248417531,
               (2, 13, 17): 1.1571660776834516, (2, 17): 1.1571660776834516, (2, 13, 19): 1.4747901776836403,
               (3, 11, 19): 1.8172881307434794, (3, 12, 19): 2.6101619429572755, (4, 6): 2.281499755801347,
               (4, 6, 9): 2.281499755801347, (6, 9): 1.032263578111594, (4, 6, 10): 2.9851322460962346,
               (4, 10): 2.9851322460962346, (5, 14, 16): 1.9880461307921204, (14, 16): 1.9880461307921204,
               (5, 14, 18): 1.6174706622300976, (14, 18): 1.6174706622300976, (5, 16, 18): 5.611766756773166,
               (16, 18): 5.611766756773166, (6, 8, 10): 1.3374542848518491, (8, 10): 1.3374542848518491,
               (6, 8, 15): 1.1241421353709862, (6, 9, 15): 1.032263578111594, (7, 13, 14): 3.8192688178009524,
               (7, 14): 2.134657042046954, (7, 14, 16): 2.134657042046954, (8, 10, 18): 3.9903330121627274,
               (10, 18): 3.9903330121627274, (8, 14, 15): 1.7001972603943867, (8, 14, 18): 1.8064597228087895}
        value = 0.6811155450226897
        expected_faces = {(2,), (5,), (11,), (8,), (14,), (17,), (1, 9), (1, 15), (13, 17), (7, 16), (4,), (1,), (7,),
                          (10,), (16,), (13,), (19,), (6, 10), (0,), (3,), (9,), (6,), (12,), (18,), (15,), (9, 15), ()}
        self.assertEqual(expected_faces, filter_by_float(dic, value))

    def test_check_if_sub_face(self):
        self.assertTrue(check_if_sub_face((1, 2, 3), (1, 2, 3, 4)))
        self.assertTrue(check_if_sub_face((1, 2), (1, 2, 3)))
        self.assertFalse(check_if_sub_face((1, 2, 3), (1, 2)))
        self.assertFalse(check_if_sub_face((1, 2, 3), (1, 2, 4)))
        self.assertFalse(check_if_sub_face((1, 2), (3, 4)))

    def test_check_if_directed_sub_face(self):
        self.assertEqual(0, boundary_operator((1, 2), (2, 3)))
        self.assertEqual(1, boundary_operator((1, 2), (1, 2, 3)))
        self.assertEqual(-1, boundary_operator((1, 3), (1, 2, 3)))
        self.assertEqual(-1, boundary_operator((1, 2), (1, 3, 2)))
        self.assertEqual(1, boundary_operator((1, 3), (1, 3, 2)))

    def test_noise(self):
        points = np.array([[1, 2], [3, 4], [5, 6]])
        perturbed_points = noise(points)
        self.assertTrue(np.any(points != perturbed_points))
        self.assertEqual(points.shape, perturbed_points.shape)

        mean = sum([math.sqrt(p[0] ** 2 + p[1] ** 2) for p in points]) / len(points)
        mean_noise = sum([math.sqrt((p[0] - point[0]) ** 2 + (p[1] - point[1]) ** 2) for p, point in
                          zip(points, perturbed_points)]) / len(points)
        self.assertAlmostEqual(mean, mean_noise, delta=3)

        std_dev = 0.1
        std_dev_noise = np.std(
            [math.sqrt((p[0] - point[0]) ** 2 + (p[1] - point[1]) ** 2) for p, point in zip(points, perturbed_points)])
        self.assertAlmostEqual(std_dev, std_dev_noise, delta=3)

    def test_connected_components(self):
        faces = [(0,), (1,), (2,), (3,), (4,), (0, 1), (0, 2), (1, 2), (2, 3), (3, 4)]
        self.assertEqual(1, connected_components(faces))
        faces = [(0,), (1,), (2,), (3,), (4,), (1, 2), (2, 3), (3, 4)]
        self.assertEqual(2, connected_components(faces))
        faces = [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (0, 1), (2, 3), (3, 6), (2, 4), (2, 5), (4, 5)]
        self.assertEqual(2, connected_components(faces))

    def test_reachable_alg(self):
        edges = [(0, 1), (2, 3), (3, 6), (2, 4), (2, 5), (4, 5)]
        self.assertEqual([0, 1], reachable_alg(edges, 0,
                                               {0: False, 1: False, 2: False, 3: False, 4: False, 5: False, 6: False}))
        self.assertEqual({2, 3, 4, 5, 6}, set(reachable_alg(edges, 2,
                                                            {0: False, 1: False, 2: False, 3: False, 4: False, 5: False,
                                                             6: False})))

    def test_num_loops(self):
        faces = [(0, 1), (2, 3), (3, 6), (2, 4), (2, 5), (2, 6), (4, 7)]
        self.assertEqual(1, num_loops(faces))
        faces = [(0, 1), (2, 3), (3, 6), (2, 4), (2, 5), (2, 6), (4, 5)]
        self.assertEqual(2, num_loops(faces))

    def test_num_triangles(self):
        faces = [(0, 1, 2), (0, 1, 3), (0, 1, 4)]
        self.assertEqual(3, num_triangles(faces))

    def test_calc_homology(self):
        faces = [(0,), (1,), (2,), (3,), (4,), (0, 1), (0, 2), (1, 2), (2, 3), (3, 4)]
        self.assertEqual((1, 1, 0), calc_homology(faces))
        faces = [(0,), (1,), (2,), (3,), (4,), (1, 2), (2, 3), (3, 4)]
        self.assertEqual((2, 0, 0), calc_homology(faces))
        faces = [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (0, 1), (2, 3), (3, 6), (2, 4), (2, 5), (4, 5)]
        self.assertEqual((2, 1, 0), calc_homology(faces))

    def test_plot_persistence_diagram(self):
        simple_alpha = AlphaComplex([[-3, 0], [0, 1], [3, 0], [-1.7, -1.8], [1.7, -1.8], [0, -4]])
        with patch('matplotlib.pyplot.show') as mocked_show, \
                patch('matplotlib.pyplot.plot') as mocked_plot:
            simple_alpha.persistence_diagram()
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

    def test_plot_barcode_diagram(self):
        simple_alpha = AlphaComplex([[-3, 0], [0, 1], [3, 0], [-1.7, -1.8], [1.7, -1.8], [0, -4]])
        with patch('matplotlib.pyplot.show') as mocked_show, \
                patch('matplotlib.pyplot.plot') as mocked_plot:
            simple_alpha.barcode_diagram()

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
