import json
import os
from pathlib import Path
from unittest import TestCase, mock
from unittest.mock import patch

import numpy as np

from ComplejosSimpliciales.src.AlphaComplex import AlphaComplex

CLOUD_PATH = os.path.join(os.path.abspath(Path(__file__).parent.parent.parent), "docs", "clouds.json")


class TestAlphaComplex(TestCase):
    ac1 = AlphaComplex(np.array(json.load(open(CLOUD_PATH))['alpha1']))

    ac2 = AlphaComplex(np.array(json.load(open(CLOUD_PATH))['alpha2']))

    ac3 = AlphaComplex(np.array(json.load(open(CLOUD_PATH))['alpha3']))

    ac4 = AlphaComplex(np.array(json.load(open(CLOUD_PATH))['alpha4']))

    def test_filtration_order_1(self):
        expected_order = [(), (0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (0, 7), (3, 6), (3, 5),
                          (0, 5), (3, 8), (0, 3), (3, 7), (0, 3, 7), (0, 3, 5), (6, 8), (3, 6, 8), (1, 9), (7, 8),
                          (5, 6), (3, 5, 6), (1, 5), (3, 7, 8), (4, 7), (2, 9), (0, 1), (0, 1, 5), (4, 8), (4, 7, 8),
                          (0, 2), (2, 7), (0, 2, 7), (1, 2), (1, 2, 9), (2, 4), (2, 4, 7), (0, 1, 2), (1, 6), (1, 5, 6)]
        self.assertEqual(expected_order, self.ac1.filtration_order())

    def test_filtration_order_2(self):
        expected_order = [(), (0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (4, 7), (4, 6), (4, 9),
                          (7, 9), (4, 7, 9), (6, 9), (4, 6, 9), (1, 7), (3, 5), (4, 8), (7, 8), (4, 7, 8), (6, 8),
                          (4, 6, 8), (0, 5), (1, 9), (1, 7, 9), (3, 8), (0, 2), (1, 8), (1, 7, 8), (3, 6), (3, 6, 8),
                          (2, 6), (0, 3), (0, 3, 5), (0, 6), (0, 2, 6), (0, 3, 6), (2, 9), (2, 6, 9)]
        self.assertEqual(expected_order, self.ac2.filtration_order())

    def test_filtration_order_3(self):
        expected_order = [(), (0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11,), (12,), (13,),
                          (14,), (15,), (16,), (17,), (18,), (19,), (9, 16), (7, 8), (2, 18), (0, 4), (12, 14),
                          (17, 18), (14, 19), (12, 19), (2, 17), (2, 17, 18), (12, 14, 19), (6, 7), (0, 8), (14, 15),
                          (9, 17), (16, 17), (9, 16, 17), (6, 10), (3, 7), (12, 15), (12, 14, 15), (3, 8), (3, 13),
                          (3, 7, 8), (0, 18), (1, 3), (7, 10), (6, 7, 10), (10, 11), (6, 9), (4, 18), (0, 4, 18),
                          (6, 8), (6, 7, 8), (1, 13), (1, 3, 13), (9, 18), (9, 17, 18), (1, 8), (1, 3, 8), (0, 6),
                          (0, 6, 8), (16, 19), (5, 11), (2, 4), (2, 4, 18), (1, 4), (5, 15), (6, 18), (6, 9, 18),
                          (0, 1), (0, 1, 8), (0, 6, 18), (0, 1, 4), (5, 14), (5, 14, 15), (3, 10), (3, 7, 10), (9, 19),
                          (9, 16, 19), (5, 10), (5, 10, 11), (10, 19), (6, 19), (6, 9, 19), (10, 14), (10, 14, 19),
                          (6, 10, 19), (3, 11), (3, 10, 11), (5, 10, 14), (11, 13), (3, 11, 13), (12, 16), (12, 16, 19),
                          (12, 17), (12, 16, 17)]
        self.assertEqual(expected_order, self.ac3.filtration_order())

    def test_filtration_order_4(self):
        expected_order = [(), (0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11,), (12,), (13,),
                          (14,), (15,), (16,), (17,), (18,), (19,), (7, 16), (1, 9), (13, 17), (6, 10), (1, 15),
                          (9, 15), (1, 9, 15), (6, 15), (5, 14), (2, 11), (4, 12), (8, 15), (11, 19), (2, 13), (6, 9),
                          (6, 9, 15), (2, 19), (6, 8), (4, 9), (6, 8, 15), (2, 11, 19), (2, 17), (2, 13, 17), (8, 10),
                          (6, 8, 10), (5, 18), (8, 14), (5, 16), (13, 19), (2, 13, 19), (1, 4), (1, 4, 9), (14, 18),
                          (5, 14, 18), (8, 18), (14, 15), (8, 14, 15), (3, 19), (0, 17), (3, 11), (8, 14, 18),
                          (3, 11, 19), (3, 12), (14, 16), (5, 14, 16), (0, 13), (0, 13, 17), (1, 12), (1, 4, 12),
                          (7, 14), (7, 14, 16), (1, 14), (1, 14, 15), (4, 6), (4, 6, 9), (12, 19), (3, 12, 19), (0, 7),
                          (4, 10), (4, 6, 10), (1, 19), (1, 12, 19), (7, 13), (0, 7, 13), (13, 14), (7, 13, 14),
                          (1, 13), (1, 13, 19), (1, 13, 14), (10, 18), (8, 10, 18), (16, 18), (5, 16, 18)]
        self.assertEqual(expected_order, self.ac4.filtration_order())

    def test_thresholdvalues_1(self):
        expected_values = [0.0, 0.024594630110749274, 0.06491550663246884, 0.09964158316325628, 0.1067160499498478,
                           0.11181663867459748, 0.12497889333174499, 0.12926191918239155, 0.129720050879878,
                           0.1429042928872187, 0.14897699064479192, 0.151309131049649, 0.15386007896068227,
                           0.15403831830169779, 0.15498302512711037, 0.16844581383081997, 0.20947134070559964,
                           0.2146289409772831, 0.2204212878472461, 0.2207272203518557, 0.23243829737556373,
                           0.24143718415372575, 0.25427755231828075, 0.25434295608514496, 0.259566655903143,
                           0.2660519202667225, 0.27011640994552, 0.7144105889522725]
        self.assertEqual(expected_values, self.ac1.thresholdvalues())

    def test_thresholdvalues_2(self):
        expected_values = [0.0, 0.036163698601824126, 0.0687469369692617, 0.08658875829741433, 0.1004969808973941,
                           0.10945407826835805, 0.10947698997829014, 0.11055084225965854, 0.13347642016530828,
                           0.13798083982835635, 0.13982338915839349, 0.140270227857948, 0.150173692446446,
                           0.1504814515923522, 0.1609810430983044, 0.17127297005159184, 0.17549368909767854,
                           0.2075916243129811, 0.23832327791636146, 0.26398454833358354, 0.26768993299910265,
                           0.27072941952967905, 0.3372151209564786, 0.33722983315924704, 0.33907631848107095,
                           0.36210637012732005]
        self.assertEqual(expected_values, self.ac2.thresholdvalues())

    def test_thresholdvalues_3(self):
        expected_values = [0, 0.013372492334073725, 0.04142029475612175, 0.04400202037538318, 0.04911455963807197,
                           0.05750211401213061, 0.05979415174994811, 0.05987782127673453, 0.0662971567981893,
                           0.07025569794627363, 0.07067990072251529, 0.07113675651754954, 0.0718171312896538,
                           0.07284655008370443, 0.07461111964572599, 0.07759791600323905, 0.08646416102371152,
                           0.08813500028678328, 0.08977646358179074, 0.09116980315881915, 0.09136555055470967,
                           0.09144332484158733, 0.09273738722293785, 0.09314123008885489, 0.10320292465591177,
                           0.10800424198696701, 0.11356902673810587, 0.1135692957755203, 0.11391056703882392,
                           0.11500638260260287, 0.11527046280465009, 0.11970598498904288, 0.1287773303569409,
                           0.1310088071059722, 0.13251628383735517, 0.13494860646400642, 0.13552236642253898,
                           0.13591668163570716, 0.14103911018811138, 0.14753151622725796, 0.15066162768072872,
                           0.15878817794096886, 0.15924339251957695, 0.16008936460106923, 0.16032124553800148,
                           0.16058515378127616, 0.16100919728848842, 0.16122628081660784, 0.17077201079486737,
                           0.17124353257525468, 0.1719033698035921, 0.17493267442945287, 0.17788243762217146,
                           0.17864341126113067, 0.1790867412845234, 0.18017134649655983, 0.18079179079051558,
                           0.18342630211201155, 0.18423901400199627, 0.18530372758297417, 0.1860185439096894,
                           0.18794891774059622, 0.2050574349332995, 0.20903538737809907, 0.39184153947179895,
                           1.0558267620678874]
        self.assertEqual(expected_values, self.ac3.thresholdvalues())

    def test_thresholdvalues_4(self):
        expected_values = [0, 0.10565789180691254, 0.3594476020513678, 0.512709492652287, 0.6337392347297187,
                           0.6579110269255046, 0.6811155450226897, 0.6962666627950436, 0.7473705160649131,
                           0.7725554426347773, 0.8984512866459233, 0.9182977742772253, 1.0085045640495482,
                           1.0097528017787816, 1.018845498134008, 1.032263578111594, 1.0663342997829408,
                           1.0835012924474037, 1.0926162823459045, 1.1241421353709862, 1.152885248417531,
                           1.1571660776834516, 1.3374542848518491, 1.3421291156269257, 1.3441831248867206,
                           1.3897471697859505, 1.474790175830562, 1.4747901776836403, 1.4771154526024748,
                           1.6174706622300976, 1.6849626346923738, 1.7001972603943867, 1.7105016392899866,
                           1.7305797293871168, 1.7631900830583442, 1.8064597228087895, 1.8172881307434794,
                           1.8739871984415204, 1.9880461307921204, 2.0820257648936034, 2.1335293172038012,
                           2.134657042046954, 2.146759732709587, 2.281499755801347, 2.6101619429572755,
                           2.655928521333648, 2.9851322460962346, 3.075420599920237, 3.0897352223305905,
                           3.3474070826562006, 3.81894308488309, 3.8192688178009524, 3.8800339027875133,
                           3.9262271123430863, 3.9903330121627274, 5.611766756773166]
        self.assertEqual(expected_values, self.ac4.thresholdvalues())

    def test_plotalpha(self):
        alpha = AlphaComplex([[-3, 0], [0, 1], [3, 0], [-1.7, -1.8], [1.7, -1.8], [0, -4]])
        with patch('matplotlib.pyplot.show') as mocked_show,\
                patch('matplotlib.pyplot.plot') as mocked_plot,\
                patch('matplotlib.pyplot.tripcolor') as mocked_tripcolor:
            alpha.plotalpha()
            calls_list = mocked_plot.mock_calls
            calls_list.reverse()

            self.assertEqual([
                 mock.call([0.0, 1.7], [1.0, -1.8], 'k'),
                 mock.call([-3.0, -1.7], [0.0, -1.8], 'k'),
                 mock.call([-1.7, 0.0], [-1.8, -4.0], 'k'),
                 mock.call([0.0, 3.0], [1.0, 0.0], 'k'),
                 mock.call([3.0, 1.7], [0.0, -1.8], 'k'),
                 mock.call([-3.0, 0.0], [0.0, 1.0], 'k'),
                 mock.call([1.7, 0.0], [-1.8, -4.0], 'k'),
                 mock.call([0.0, -1.7], [1.0, -1.8], 'k'),
                 mock.call([-1.7, 1.7], [-1.8, -1.8], 'k')
            ], calls_list[0:9])
            self.assertEqual(65, mocked_plot.call_count)

            self.assertEqual(3, mocked_tripcolor.call_count)

            mocked_show.assert_called()
            self.assertEqual(9, mocked_show.call_count)


