import os
import sys
import unittest

import torch


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS_DIR = os.path.join(ROOT_DIR, 'utils')

if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

import util


class UtilTests(unittest.TestCase):
    def test_calculate_regression_error_returns_numeric_value(self):
        error = util.calculateRegressionError(
            torch.tensor([1.5, 2.5]),
            torch.tensor([1.0, 2.0]),
        )

        self.assertAlmostEqual(error, 0.5)


if __name__ == '__main__':
    unittest.main()
