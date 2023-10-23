import numpy as np
import pandas as pd
import sktime
import unittest
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from preprocess import (
    InsufficientColumnsError,
    ColumnDoesNotExistError,
    NoCSVError,
    ImputerNumberError,
    ImputerMappingError,
)
from preprocess import (
    validate_dataset,
    validate_arguments,
    filter_dataset,
    remove_missing_timestamps,
    remove_missing_sensordata,
    datetime_to_timestamp,
    impute_mean,
    impute_median,
    impute_knn,
    impute_labels,
    impute_categorical,
    one_hot_encode,
)


class TestPreprocessing(unittest.TestCase):
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

        # Define dummy arguments
        self.dataset_path = "data.xls"
        self.sensordata_columns = "Current,Temperature"
        self.imputation = "1,2,3"

        # Toy dataframe with 'ts' column
        self.ts_data = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        self.ts_df = pd.DataFrame(self.ts_data, columns=["ts"])
        self.with_ts = pd.concat(
            [self.df.reset_index(drop=True), self.ts_df.reset_index(drop=True)], axis=1
        )

        # Toy dataframe with imputed categorical values
        self.encode_data = [
            "inactive",
            "active",
            "active",
            "active",
            "inactive",
            "active",
            "active",
            "inactive",
            "active",
            "inactive",
        ]
        self.to_encode = pd.DataFrame(self.df_data, columns=self.df_columns)
        self.to_encode["status"] = pd.DataFrame(self.encode_data, columns=["status"])


class TestValidateDataset(TestPreprocessing):
    def test_dataframe_size(self):
        """Checks if the function throws InsufficientColumnsError when a single column dataset is passed"""
        with self.assertRaises(InsufficientColumnsError):
            validate_dataset(self.df[["datetime"]])


class TestValidateArguments(TestPreprocessing):
    def test_csv_error(self):
        """Checks for NoCSVError if incorrect file path is provided"""
        with self.assertRaises(NoCSVError):
            validate_arguments(self)

    def test_columns_impute(self):
        """Checks for ImputerNumberError when number of columns does not match number of imputation techniques"""
        self.dataset_path = "data.csv"
        with self.assertRaises(ImputerNumberError):
            validate_arguments(self)

    def test_invalid_impute(self):
        """Checks for ImputerMappingError when invalid mapping is provided"""
        self.dataset_path = "data.csv"
        self.imputation = "1,4"
        with self.assertRaises(ImputerMappingError):
            validate_arguments(self)


class TestFilterDataset(TestPreprocessing):
    def test_return_type(self):
        """Checks if the function returns a pandas dataframe and list for valid inputs"""
        new_df, columns_list = filter_dataset(
            self.df,
            "datetime",
            "Current,Pressure,Temperature,Voltage",
            "status",
            "anomaly",
        )
        self.assertIsInstance(new_df, pd.core.frame.DataFrame)
        self.assertIsInstance(columns_list, list)

    def test_column_header(self):
        """Checks for ColumnDoesNotExistErrors for non-existent column headers"""
        with self.assertRaises(ColumnDoesNotExistError):
            filter_dataset(
                self.df,
                "datetime",
                "Current,Pressure,Temperature,Volts",
                "status",
                "anomaly",
            )


class TestRemoveMissingTimestamps(TestPreprocessing):
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


class TestRemoveMissingSensordata(TestPreprocessing):
    def test_return_type(self):
        """Checks if the function returns a pandas dataframe"""
        self.assertIsInstance(
            remove_missing_sensordata(self.df), pd.core.frame.DataFrame
        )

    def test_dataframe_size(self):
        """Checks if the size of dataset with dropped columns is less than original dataframe"""
        self.assertLess(remove_missing_sensordata(self.df).shape[0], self.df.shape[0])


class TestDatetimeToTimestamp(TestPreprocessing):
    def test_return_type(self):
        """Checks if the function returns a pandas dataframe"""
        self.assertIsInstance(
            datetime_to_timestamp(self.df, "datetime"), pd.core.frame.DataFrame
        )

    def test_column_name(self):
        """Checks if the function creates a new column 'ts'"""
        self.assertTrue("ts" in datetime_to_timestamp(self.df, "datetime").columns)


class TestImputeMean(TestPreprocessing):
    def test_return_type(self):
        """Checks if the function returns a pandas dataframe and imputer object"""
        self.assertIsInstance(
            impute_mean(self.df, "Current")[0], pd.core.frame.DataFrame
        )
        self.assertIsInstance(
            impute_mean(self.df, "Current")[1],
            sktime.transformations.series.impute.Imputer,
        )

    def test_empty_values(self):
        """Checks if the dataframe has 0 missing values"""
        self.assertLess(impute_mean(self.df, "Current")[0]["Current"].isna().sum(), 1)


class TestImputeMedian(TestPreprocessing):
    def test_return_type(self):
        """Checks if the function returns a pandas dataframe and imputer object"""
        self.assertIsInstance(
            impute_median(self.df, "Current")[0], pd.core.frame.DataFrame
        )
        self.assertIsInstance(
            impute_median(self.df, "Current")[1],
            sktime.transformations.series.impute.Imputer,
        )

    def test_empty_values(self):
        """Checks if the dataframe has 0 missing values"""
        self.assertLess(
            impute_median(self.df, "Current")[0]["Current"].isnull().sum(), 1
        )


class TestImputeKNN(TestPreprocessing):
    def test_return_type(self):
        """Checks if the function returns a pandas dataframe and imputer object"""
        self.assertIsInstance(
            impute_knn(self.with_ts, "Current")[0], pd.core.frame.DataFrame
        )
        self.assertIsInstance(impute_knn(self.with_ts, "Current")[1], KNNImputer)

    def test_empty_values(self):
        """Checks if the dataframe has 0 missing values"""
        self.assertLess(
            impute_knn(self.with_ts, "Current")[0]["Current"].isna().sum(), 1
        )


class TestImputeLabels(TestPreprocessing):
    def test_return_type(self):
        """Checks if the function returns a pandas dataframe"""
        self.assertIsInstance(
            impute_labels(self.df, "anomaly"), pd.core.frame.DataFrame
        )

    def test_empty_values(self):
        self.assertLess(impute_labels(self.df, "anomaly")["anomaly"].isnull().sum(), 1)


class TestImputeCategorical(TestPreprocessing):
    def test_return_type(self):
        """Checks if the function returns a pandas dataframe and string value"""
        self.assertIsInstance(
            impute_categorical(self.df, "status")[0], pd.core.frame.DataFrame
        )
        self.assertIsInstance(impute_categorical(self.df, "status")[1], str)

    def test_number_of_columns(self):
        self.assertGreaterEqual(
            len(impute_categorical(self.df, "status")[0].columns), len(self.df.columns)
        )


class TestOneHotEncode(TestPreprocessing):
    def test_return_type(self):
        """Checks if the function returns a pandas dataframe and encoder object"""
        self.assertIsInstance(
            one_hot_encode(self.to_encode, "status")[0], pd.core.frame.DataFrame
        )
        self.assertIsInstance(
            one_hot_encode(self.to_encode, "status")[1], OneHotEncoder
        )

    def test_number_of_columns(self):
        self.assertGreater(
            len(one_hot_encode(self.to_encode, "status")[0].columns),
            len(self.df.columns),
        )
