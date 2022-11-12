from unittest import TestCase

import pytest


class Test(TestCase):

    @pytest.mark.skip()
    def test_calc_radio(self):
        self.fail()
