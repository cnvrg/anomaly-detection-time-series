import numpy as np
import pandas as pd
import unittest
from batchpredict import InsufficientColumnsError, ColumnDoesNotExistError
from batchpredict import (
    validate_dataset,
    get_numerical_columns,
    filter_dataset,
    remove_missing_timestamps,
    dataframe_to_list,
)


class TestBatchPredict(unittest.TestCase):
    def setUp(self):
        """Overrides setUp from unittest to create toy pandas dataframes for testing"""
        # Toy dataframe
        self.df_data = [
            [
                np.nan,
                0.0810554999999999,
                0.132572,
                1.4094799999999998,
                np.nan,
                np.nan,
                22.0867,
                231.247,
                77.0,
                0.0,
                0.0,
                "inactive",
            ],
            [
                "2020-03-01 17:03:55",
                0.0814173,
                0.136616,
                1.942,
                np.nan,
                np.nan,
                22.0904,
                229.569,
                76.9794,
                np.nan,
                np.nan,
                "active",
            ],
            [
                "2020-03-01 17:03:56",
                0.082209,
                0.132413,
                1.60694,
                0.054711,
                86.7442,
                np.nan,
                np.nan,
                76.0209,
                0.0,
                np.nan,
                "active",
            ],
            [
                "2020-03-01 17:03:57",
                np.nan,
                0.137628,
                1.13916,
                np.nan,
                86.8673,
                22.0944,
                218.121,
                77.0,
                0.0,
                0.0,
                "active",
            ],
            [
                "2020-03-01 17:03:58",
                0.0819601,
                0.130911,
                1.57836,
                np.nan,
                86.9175,
                22.0887,
                231.304,
                77.0,
                np.nan,
                0.0,
                "inactive",
            ],
            [
                "2020-03-01 17:03:59",
                0.0808617,
                0.138537,
                1.84093,
                -0.2732159999999999,
                86.7062,
                np.nan,
                242.328,
                77.0,
                0.0,
                0.0,
                np.nan,
            ],
            [
                "2020-03-01 17:04:00",
                0.0832623,
                np.nan,
                np.nan,
                0.054711,
                np.nan,
                22.0878,
                241.67,
                77.0,
                0.0,
                0.0,
                "active",
            ],
            [
                "2020-03-01 17:04:01",
                np.nan,
                0.136159,
                1.32774,
                0.054711,
                87.0097,
                22.0878,
                232.003,
                77.0209,
                np.nan,
                0.0,
                "inactive",
            ],
            [
                np.nan,
                0.0809118,
                np.nan,
                np.nan,
                np.nan,
                86.9424,
                22.0911,
                228.966,
                77.9794,
                np.nan,
                0.0,
                "active",
            ],
            [
                "2020-03-01 17:04:03",
                0.0825833,
                0.131779,
                np.nan,
                np.nan,
                np.nan,
                22.1058,
                248.81,
                76.9794,
                0.0,
                0.0,
                "inactive",
            ],
        ]
        self.df_columns = [
            "datetime",
            "Accelerometer1RMS",
            "Accelerometer2RMS",
            "Current",
            "Pressure",
            "Temperature",
            "Thermocouple",
            "Voltage",
            "Volume Flow RateRMS",
            "anomaly",
            "changepoint",
            "status",
        ]
        self.df = pd.DataFrame(self.df_data, columns=self.df_columns)

        # Define expected values
        self.df_length = 10
        self.list_element = [1.32774, 0.054711, 87.0097, 232.003]


class TestValidateDataset(TestBatchPredict):
    def test_dataframe_size(self):
        """Checks if the function throws InsufficientColumnsError when a single column dataset is passed"""
        with self.assertRaises(InsufficientColumnsError):
            validate_dataset(self.df[["datetime"]])


class TestGetNumericalColumns(TestBatchPredict):
    def test_return_type(self):
        """Checks if the function returns a list for valid inputs"""
        self.assertIsInstance(
            get_numerical_columns(
                ["datetime", "Current", "Voltage", "status"], ["status"]
            ),
            list,
        )

    def test_return_length(self):
        """Checks if the function returns the expected list"""
        self.assertEqual(
            get_numerical_columns(
                ["datetime", "Current", "Voltage", "status"], ["status"]
            ),
            ["Current", "Voltage"],
        )


class TestFilterDataset(TestBatchPredict):
    def test_return_type(self):
        """Checks if the function returns a pandas dataframe and list for valid inputs"""
        self.assertIsInstance(
            filter_dataset(
                self.df,
                [
                    "datetime",
                    "Current",
                    "Pressure",
                    "Temperature",
                    "Voltage",
                    "status",
                    "anomaly",
                ],
            ),
            pd.core.frame.DataFrame,
        )

    def test_column_header(self):
        """Checks for ColumnDoesNotExistErrors for non-existent column headers"""
        with self.assertRaises(ColumnDoesNotExistError):
            filter_dataset(
                self.df,
                [
                    "datetime",
                    "Current",
                    "Pressure",
                    "Temperature",
                    "Voltage",
                    "status",
                    "label",
                ],
            )


class TestRemoveMissingTimestamps(TestBatchPredict):
    def test_return_type(self):
        """Checks if the function returns a pandas dataframe"""
        self.assertIsInstance(
            remove_missing_timestamps(self.df, "datetime"), pd.core.frame.DataFrame
        )

    def test_dataframe_size(self):
        """Checks if the size of dataset with dropped columns is less than original dataframe"""
        self.assertLess(
            remove_missing_timestamps(self.df, "datetime").shape[0], self.df.shape[0]
        )


class TestDataframeToList(TestBatchPredict):
    def test_return_type(self):
        """Checks if the function converts a dataframe to a list"""
        self.assertIsInstance(
            dataframe_to_list(
                self.df[["datetime", "Current", "Pressure", "Temperature", "Voltage"]]
            ),
            list,
        )

    def test_list_element(self):
        """Checks if an arbitrary element of the returned list matches the expected element"""
        self.assertEqual(
            dataframe_to_list(
                self.df[["datetime", "Current", "Pressure", "Temperature", "Voltage"]]
            )[7],
            self.list_element,
        )

    def test_return_length(self):
        """Checks if the number of rows in the input dataframe is equal to length of the list"""
        self.assertEqual(
            len(
                dataframe_to_list(
                    self.df[
                        ["datetime", "Current", "Pressure", "Temperature", "Voltage"]
                    ]
                )
            ),
            self.df_length,
        )
