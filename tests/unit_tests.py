import sys
import os
import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from generate_features import (
    calculate_norm_range,
    log_transform,
    multiply_columns,
    standard_scale,
    sqrt_transform,
)


class TestGenerateFeatures(unittest.TestCase):
    """
    Test case for the functions in the `generate_features` module. This class inherits from the `unittest.TestCase`
    class and defines a set of test methods for each of the functions in the module.

    The class includes a `setUp` method that creates a pandas DataFrame for use in the tests.
    The test methods cover both the happy and unhappy paths for each function, where the happy path tests include
    valid input scenarios, and the unhappy path tests cover scenarios with invalid input, where an error is
    expected to occur.

    Each test method in the class uses the `assert_called_with` method from the `unittest.mock` module to check
    whether the logger method was called with the correct parameters in case an error occurred.

    Attributes:
        None

    Methods:
        setUp(self): Sets up the test fixture by creating a pandas DataFrame for use in the tests.

        test_calculate_norm_range_happy(self): Test method for the `calculate_norm_range` function with valid input.

        test_log_transform_happy(self): Test method for the `log_transform` function with valid input.

        test_multiply_columns_happy(self): Test method for the `multiply_columns` function with valid input.

        test_standard_scale_happy(self): Test method for the `standard_scale` function with valid input.

        test_sqrt_transform_happy(self): Test method for the `sqrt_transform` function with valid input.

        test_calculate_norm_range_unhappy(self): Test method for the `calculate_norm_range` function with invalid input.

        test_log_transform_unhappy(self): Test method for the `log_transform` function with invalid input.

        test_multiply_columns_unhappy(self): Test method for the `multiply_columns` function with invalid input.

        test_standard_scale_unhappy(self): Test method for the `standard_scale` function with invalid input.

        test_sqrt_transform_unhappy(self): Test method for the `sqrt_transform` function with invalid input.

    """

    def setUp(self):
        self.df = pd.DataFrame(
            {
                "A": [1, 2, 3],
                "B": [4, 5, 6],
                "C": [7, 8, 9],
            }
        )

    # Happy path tests

    def test_calculate_norm_range_happy(self):
        config = {"norm_A": {"min_col": "A", "max_col": "B", "mean_col": "C"}}
        expected = self.df.copy()
        expected["norm_A"] = (expected["B"] - expected["A"]) / expected["C"]
        result = calculate_norm_range(self.df.copy(), config)
        pd.testing.assert_frame_equal(result, expected)

    def test_log_transform_happy(self):
        config = {"log_A": "A"}
        expected = self.df.copy()
        expected["log_A"] = np.log1p(expected["A"])
        result = log_transform(self.df.copy(), config)
        pd.testing.assert_frame_equal(result, expected)

    def test_multiply_columns_happy(self):
        config = {"A_times_B": {"col_a": "A", "col_b": "B"}}
        expected = self.df.copy()
        expected["A_times_B"] = expected["A"] * expected["B"]
        result = multiply_columns(self.df.copy(), config)
        pd.testing.assert_frame_equal(result, expected)

    def test_standard_scale_happy(self):
        config = {"A"}
        expected = self.df.copy()
        expected["A"] = (expected["A"] - expected["A"].mean()) / expected["A"].std()
        result = standard_scale(self.df.copy(), config)
        pd.testing.assert_frame_equal(result, expected)

    def test_sqrt_transform_happy(self):
        config = {"sqrt_A": "A"}
        expected = self.df.copy()
        expected["sqrt_A"] = np.sqrt(expected["A"])
        result = sqrt_transform(self.df.copy(), config)
        pd.testing.assert_frame_equal(result, expected)

    # Unhappy path tests
    @patch("generate_features.logger")
    def test_calculate_norm_range_unhappy(self, mock_logger):
        config = {"norm_A": {"min_col": "X", "max_col": "Y", "mean_col": "Z"}}
        calculate_norm_range(self.df.copy(), config)
        mock_logger.error.assert_called_with(
            "Error during 'calculate_norm_range' transformation for '%s': %s",
            "norm_A",
            "'Y'",
        )

    @patch("generate_features.logger")
    def test_log_transform_unhappy(self, mock_logger):
        config = {"log_A": "X"}
        log_transform(self.df.copy(), config)
        mock_logger.error.assert_called_with(
            "Error during 'log_transform' transformation for '%s': %s", "log_A", "'X'"
        )

    @patch("generate_features.logger")
    def test_multiply_columns_unhappy(self, mock_logger):
        config = {"A_times_B": {"col_a": "X", "col_b": "Y"}}
        multiply_columns(self.df.copy(), config)
        mock_logger.error.assert_called_with(
            "Error during 'multiply_columns' transformation for '%s': %s",
            "A_times_B",
            "'X'",
        )

    @patch("generate_features.logger")
    def test_standard_scale_unhappy(self, mock_logger):
        config = {"X"}
        standard_scale(self.df.copy(), config)
        mock_logger.error.assert_called_with(
            "Error during 'standard_scale' transformation for '%s': %s", "X", "'X'"
        )

    @patch("generate_features.logger")
    def test_sqrt_transform_unhappy(self, mock_logger):
        config = {"sqrt_A": "X"}
        sqrt_transform(self.df.copy(), config)
        mock_logger.error.assert_called_with(
            "Error during 'sqrt_transform' transformation for '%s': %s", "X", "'X'"
        )


if __name__ == "__main__":
    unittest.main()
