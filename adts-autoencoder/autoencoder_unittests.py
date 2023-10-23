import pandas as pd
import unittest
from autoencoder import NoCSVError, ContaminationValueError
from autoencoder import (
    validate_arguments,
    get_contamination,
    split_features_labels,
    map_labels,
)


class TestAutoEncoder(unittest.TestCase):
    def setUp(self):
        """Overrides setUp from unittest to create a toy pandas dataframe for testing"""
        # Here anomalies and non-anomalies are represented by -1 and 1 respectively.
        self.train_data = [
            [
                "2020-03-01 17:03:55",
                1.942,
                229.569,
                86.92906,
                -0.07645979999999995,
                1.0,
                0.0,
                1,
            ],
            ["2020-03-01 17:03:56", 1.60694, 230.345, 86.7442, 0.054711, 1.0, 0.0, 1],
            [
                "2020-03-01 17:03:57",
                1.13916,
                218.121,
                86.8673,
                -0.07645979999999995,
                1.0,
                0.0,
                1,
            ],
            [
                "2020-03-01 17:03:58",
                1.57836,
                231.304,
                86.9175,
                -0.20763059999999997,
                0.0,
                1.0,
                1,
            ],
            [
                "2020-03-01 17:03:59",
                1.84093,
                242.328,
                86.7062,
                -0.2732159999999999,
                1.0,
                0.0,
                -1,
            ],
            [
                "2020-03-01 17:04:00",
                1.5707385313807534,
                241.67,
                86.97130000000001,
                0.054711,
                1.0,
                0.0,
                -1,
            ],
            ["2020-03-01 17:04:01", 1.32774, 232.003, 87.0097, 0.054711, 0.0, 1.0, 1],
        ]
        self.valid_data = [
            [
                "2020-03-01 17:04:03",
                1.5707385313807534,
                248.81,
                87.00912000000001,
                -0.010874399999999979,
                0.0,
                1.0,
                1,
            ],
            [
                "2020-03-01 17:04:04",
                1.15871,
                229.105,
                87.01916,
                -0.601143,
                1.0,
                0.0,
                -1,
            ],
            [
                "2020-03-01 17:04:07",
                1.66831,
                206.299,
                87.00912000000001,
                -0.2732159999999999,
                1.0,
                0.0,
                -1,
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
        self.train_df = pd.DataFrame(self.train_data, columns=self.df_columns)
        self.valid_df = pd.DataFrame(self.valid_data, columns=self.df_columns)

        # Define dummy arguments
        self.train_path = "train.csv"
        self.valid_path = "valid.xls"

        # Define lists for testing map_labels
        self.trainy = [1, 1, 1, 1, -1, -1, 1]
        self.validy = [1, -1, -1]

        # Define +/- labels
        self.positive_label = "-1"
        self.negative_label = "1"

        # Expected values
        self.contamination_ratio = 0.2857142857142857
        self.train_length = 7
        self.valid_length = 3
        self.train_mapped = [0, 0, 0, 0, 1, 1, 0]
        self.valid_mapped = [0, 1, 1]


class TestValidateArguments(TestAutoEncoder):
    def test_data_paths(self):
        """Checks for NoCSVError when invalid dataset paths are provided"""
        with self.assertRaises(NoCSVError):
            validate_arguments(self)


class TestGetContamination(TestAutoEncoder):
    def test_return_type(self):
        """Checks if the function returns a float value"""
        self.assertIsInstance(
            get_contamination(self.train_df, self.positive_label), float
        )

    def test_return_value(self):
        """Checks if the function returns the expected contamination ratio"""
        self.assertAlmostEqual(
            get_contamination(self.train_df, self.positive_label),
            self.contamination_ratio,
        )

    def test_invalid_contamination(self):
        """Checks for ContaminationValueError when contamination value is out of range"""
        with self.assertRaises(ContaminationValueError):
            get_contamination(self.valid_df, self.positive_label)


class TestSplitFeaturesLabels(TestAutoEncoder):
    def test_return_type(self):
        """Checks if the function returns four lists"""
        returned_lists = split_features_labels(self.train_df, self.valid_df)
        self.assertIsInstance(returned_lists[0], list)
        self.assertIsInstance(returned_lists[1], list)
        self.assertIsInstance(returned_lists[2], list)
        self.assertIsInstance(returned_lists[3], list)

    def test_list_lengths(self):
        """Checks if the returned lists are of expected lengths"""
        returned_lists = split_features_labels(self.train_df, self.valid_df)
        self.assertEqual(len(returned_lists[0]), self.train_length)
        self.assertEqual(len(returned_lists[1]), self.train_length)
        self.assertEqual(len(returned_lists[2]), self.valid_length)
        self.assertEqual(len(returned_lists[3]), self.valid_length)


class TestMapLabels(TestAutoEncoder):
    def test_return_type(self):
        """Checks if the function returns lists"""
        returned_lists = map_labels(
            self.trainy, self.validy, self.positive_label, self.negative_label
        )
        self.assertIsInstance(returned_lists[0], list)
        self.assertIsInstance(returned_lists[1], list)

    def test_list_values(self):
        """Checks if function returns list with just 1's and 0's"""
        returned_lists = map_labels(
            self.trainy, self.validy, self.positive_label, self.negative_label
        )
        self.assertEqual(returned_lists[0], self.train_mapped)
        self.assertEqual(returned_lists[1], self.valid_mapped)
