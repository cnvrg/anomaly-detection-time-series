import pandas as pd
import unittest
from sklearn.preprocessing import StandardScaler
from tvsplit import ValidationSizeError, validate_arguments
from tvsplit import split_dataset, normalize_data


class TestTVSplit(unittest.TestCase):
    def setUp(self):
        """Overrides setUp from unittest to create a toy pandas dataframe for testing"""
        # Toy Dataframe
        self.df_data = [
            [
                "2020-03-01 17:03:55",
                1.942,
                229.569,
                86.92906,
                -0.07645979999999995,
                1.0,
                0.0,
                0.0,
            ],
            ["2020-03-01 17:03:56", 1.60694, 230.345, 86.7442, 0.054711, 1.0, 0.0, 0.0],
            [
                "2020-03-01 17:03:57",
                1.13916,
                218.121,
                86.8673,
                -0.07645979999999995,
                1.0,
                0.0,
                0.0,
            ],
            [
                "2020-03-01 17:03:58",
                1.57836,
                231.304,
                86.9175,
                -0.20763059999999997,
                0.0,
                1.0,
                0.0,
            ],
            [
                "2020-03-01 17:03:59",
                1.84093,
                242.328,
                86.7062,
                -0.2732159999999999,
                1.0,
                0.0,
                0.0,
            ],
            [
                "2020-03-01 17:04:00",
                1.5707385313807534,
                241.67,
                86.97130000000001,
                0.054711,
                1.0,
                0.0,
                0.0,
            ],
            ["2020-03-01 17:04:01", 1.32774, 232.003, 87.0097, 0.054711, 0.0, 1.0, 0.0],
            [
                "2020-03-01 17:04:03",
                1.5707385313807534,
                248.81,
                87.00912000000001,
                -0.010874399999999979,
                0.0,
                1.0,
                0.0,
            ],
            [
                "2020-03-01 17:04:04",
                1.15871,
                229.105,
                87.01916,
                -0.601143,
                1.0,
                0.0,
                0.0,
            ],
            [
                "2020-03-01 17:04:07",
                1.66831,
                206.299,
                87.00912000000001,
                -0.2732159999999999,
                1.0,
                0.0,
                0.0,
            ],
        ]
        self.df_columns = [
            "datetime",
            "Current",
            "Voltage",
            "Temperature",
            "Pressure",
            "x0_active",
            "x0_inactive",
            "anomaly",
        ]
        self.df = pd.DataFrame(self.df_data, columns=self.df_columns)

        # Define dummy arguments
        self.valid_size = "0.6"


class TestValidateArguments(TestTVSplit):
    def test_validation_size(self):
        """Checks for ValidtionSizeError if validation size is invalid"""
        with self.assertRaises(ValidationSizeError):
            validate_arguments(self)


class TestSplitDataset(TestTVSplit):
    def test_return_type(self):
        """Checks if the function returns two pandas dataframes"""
        self.assertIsInstance(split_dataset(self.df, 0.2)[0], pd.core.frame.DataFrame)
        self.assertIsInstance(split_dataset(self.df, 0.2)[1], pd.core.frame.DataFrame)

    def test_split_size(self):
        """Checks if the sizes of the training and validation sets are correct"""
        self.assertEqual(split_dataset(self.df, 0.2)[0].shape[0], 8)
        self.assertEqual(split_dataset(self.df, 0.2)[1].shape[0], 2)


class TestNormalizeData(TestTVSplit):
    def test_return_type(self):
        """Checks if the function returns a StandardScaler object"""
        self.assertIsInstance(normalize_data(self.df), StandardScaler)
