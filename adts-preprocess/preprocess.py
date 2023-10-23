import argparse
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sktime.transformations.series.impute import Imputer

cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")


class InsufficientColumnsError(Exception):
    """Raise if original dataset has less than 2 columns"""

    def __init__(self, num_columns):
        super().__init__(num_columns)
        self.num_columns = num_columns

    def __str__(self):
        return f"InsufficientColumnsError: Your dataset contains {self.num_columns} columns. The original dataset should have atleast two columns i.e. timestamp and sensor data."


class NoCSVError(Exception):
    """Raise if input dataset is not in CSV format"""

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "NoCSVError: Input dataset needs to be in csv format for preprocessing!"


class ImputerNumberError(Exception):
    """Raise if number of imputers does not match number of sensor-data columns"""

    def __init__(self, num_columns, num_imputers):
        super().__init__(num_columns, num_imputers)
        self.num_columns = num_columns
        self.num_imputers = num_imputers

    def __str__(self):
        return f"ImputerNumberError: Your dataset contains {self.num_columns} sensor-data columns but you have provided {self.num_imputers} imputers. --sensordata_columns and --imputation need to have equal number of comma-separated values. Check your input arguments!"


class ImputerMappingError(Exception):
    """Raise if invalid imputer mapping occurs"""

    def __init__(self, imputation_method):
        super().__init__(imputation_method)
        self.imputation_method = imputation_method

    def __str__(self):
        return f"ImputerMappingError: {self.imputation_method} does not map to a valid imputation technique. Check your input arguments!"


class ColumnDoesNotExistError(Exception):
    """Raise if column header does not exist in dataframe"""

    def __init__(self, column_name):
        super().__init__(column_name)
        self.column_name = column_name

    def __str__(self):
        return f"ColumnDoesNotExistError: {self.column_name} does not exist in dataset!"


def parse_parameters():
    """Command line parser."""
    parser = argparse.ArgumentParser(description="""Preprocessing""")
    parser.add_argument(
        "--dataset_path",
        action="store",
        dest="dataset_path",
        required=True,
        help="""--- Path to the original dataset in csv format ---""",
    )
    parser.add_argument(
        "--timestamp_column",
        action="store",
        dest="timestamp_column",
        required=False,
        default="timestamp",
        help="""--- column name/header for timestamps ---""",
    )
    parser.add_argument(
        "--sensordata_columns",
        action="store",
        dest="sensordata_columns",
        required=False,
        default="sensor1,sensor2",
        help="""--- comma-separated column names/headers for data from different sensors ---""",
    )
    parser.add_argument(
        "--label_column",
        action="store",
        dest="label_column",
        required=False,
        default="label",
        help="""--- column name/header containing binary labels for anomalies ---""",
    )
    parser.add_argument(
        "--imputation",
        action="store",
        dest="imputation",
        required=False,
        default="None",
        help="""--- comma-separated data imputation techniques (1-mean, 2-median, 3-knn) for all sensor data columns ---""",
    )
    parser.add_argument(
        "--categorical_columns",
        action="store",
        dest="categorical_columns",
        required=False,
        default="None",
        help="""--- comma-separated column names/headers for data that has categorical values and that need to be one-hot encoded ---""",
    )
    parser.add_argument(
        "--output_dir",
        action="store",
        dest="output_dir",
        required=False,
        default=cnvrg_workdir,
        help="""--- The path to save preprocessed dataset file to ---""",
    )
    parser.add_argument(
        "--positive_label",
        action="store",
        dest="positive_label",
        required=False,
        default="1",
        help="""--- label for anomalous samples ---""",
    )
    parser.add_argument(
        "--negative_label",
        action="store",
        dest="negative_label",
        required=False,
        default="0",
        help="""--- label for non-anomalous (normal) samples ---""",
    )
    return parser.parse_args()


def validate_dataset(df):
    """Performs validation on original dataset

    Checks if the dataset has atleast two columns

    Args:
        df: The original dataset as a pandas dataframe

    Raises:
        InsufficientColumnsError: If dataset contains less than two columns
    """
    if len(df.columns) < 2:
        raise InsufficientColumnsError(len(df.columns))


def validate_arguments(args):
    """Validates input arguments

    Checks if input dataset path points to a csv file. Also checks if both arguments contain the same number of
    comma-seperated values and imputation techniques consist of valid values (1,2,3).

    Args:
        args: argparse object

    Raises:
        NoCSVError: If input dataset is not in csv format
        ImputerNumberError: If number of values in sensor-data columns and imputation techniques does not match
        ImputerMappingError: If imputation techniques have values other than 1,2,3
    """
    if ".csv" not in args.dataset_path:
        raise NoCSVError()

    if args.imputation.lower() == "none":
        return

    if len(args.sensordata_columns.split(",")) != len(args.imputation.split("!")):
        raise ImputerNumberError(
            len(args.sensordata_columns.split(",")), len(args.imputation.split("!"))
        )

    for method in args.imputation.split("!"):
        if int(method) not in [1, 2, 3]:
            raise ImputerMappingError(int(method))


def filter_dataset(
    df, timestamp_column, sensordata_columns, categorical_columns, label_column
):
    """Creates a new dataframe consisting of relevant columns specififed by the user.

    Selects specific columns from the original dataset and creates a pandas dataframe.
    Outputs a list containing column names/headers.

    Args:
        df: The original dataset as a pandas dataframe
        timestamp_column: A sring representing the column name for the timestamp feature
        sensordata_columns: A string representing comma-seperated column names for sensor data features

    Raises:
        ColumnDoesNotExistError: If any of the column headers do not exist in the dataset

    Returns:
        filtered_df: a pandas dataframe consisting of column headers (timestamp, sensor data and categorical data) specified by the user.
        column_list: list containing column names/headers
    """
    if timestamp_column not in df.columns:
        raise ColumnDoesNotExistError(timestamp_column)

    if label_column not in df.columns:
        raise ColumnDoesNotExistError(label_column)

    column_list = []
    column_list.append(timestamp_column)
    sensor_columns = sensordata_columns.split(",")

    for sensor_column in sensor_columns:
        sensor_column = sensor_column.strip()
        if sensor_column not in df.columns:
            raise ColumnDoesNotExistError(sensor_column)
        column_list.append(sensor_column)

    category_columns = categorical_columns.split(",")

    for categorical_column in category_columns:
        categorical_column = categorical_column.strip()
        if categorical_column not in df.columns:
            raise ColumnDoesNotExistError(categorical_column)
        column_list.append(categorical_column)

    column_list.append(label_column)
    filtered_df = df[column_list]
    return filtered_df, column_list


def remove_missing_timestamps(df, timestamp_column):
    """Removes row entries having missing timestamp values from a pandas dataframe."""
    return df.dropna(subset=[timestamp_column])


def remove_missing_sensordata(df):
    """Removes row entries having missing sensor data values from a pandas dataframe."""
    sensordata_columns = df.columns[1:]
    return df.dropna(subset=sensordata_columns)


def datetime_to_timestamp(df, timestamp_column):
    """Converts string timestamps to integer values and adds a new column to the dataframe."""
    df["ts"] = pd.to_datetime(df[timestamp_column]).values.astype(np.float64) // 10**9
    return df


def impute_mean(df, column_header):
    """Replaces missing sensor data values with mean of the corresponding column.

    Preprocesses the dataframe and creates an imputer object that can be used to preprocess test data

    Args:
        df: The original dataset as a pandas dataframe.
        column_header: Column to perform mean imputation on.

    Returns:
        df: the imputed pandas dataframe
        imputer: An sktime.transformations.series.impute.Imputer object
    """
    imp = Imputer(method="mean")
    mean_imputer = imp.fit(df[column_header])
    df[column_header] = mean_imputer.transform(df[column_header])
    return df, mean_imputer


def impute_median(df, column_header):
    """Replaces missing sensor data values with median of the corresponding column.

    Preprocesses the dataframe and creates an imputer object that can be used to preprocess test data

    Args:
        df: The original dataset as a pandas dataframe.
        column_header: Column to perform mean imputation on.

    Returns:
        df: the imputed pandas dataframe
        imputer: An sktime.transformations.series.impute.Imputer object
    """
    imp = Imputer(method="median")
    median_imputer = imp.fit(df[column_header])
    df[column_header] = median_imputer.transform(df[column_header])
    return df, median_imputer


def impute_knn(df, column_header):
    """Fills missing sensor data using KNN interpolation.

    This imputation technique considers 5 nearest neighbors while filling missing data.
    Makes use of 'ts' (timestamps as float64) column as index.

    Args:
        df: The original dataset as a pandas dataframe.
        column_header: Column to perform KNN imputation on.

    Returns:
        df: the imputed pandas dataframe
        imputer: An sklearn.impute.KNNImputer object
    """
    imp = KNNImputer(copy=False)
    knn_imputer = imp.fit(df[["ts", column_header]])
    df[["ts", column_header]] = knn_imputer.transform(df[["ts", column_header]])
    return df, knn_imputer


def impute_labels(df, label_column):
    """Fills missing labels with the most frequently occuring label"""
    df[label_column] = df[label_column].fillna(df[label_column].mode()[0])
    return df


def impute_categorical(df, categorical_column):
    """Fills missing categorical values with most frquently occuring category"""
    categorical_mode = df[categorical_column].mode()[0]
    df[categorical_column] = df[categorical_column].fillna(categorical_mode)
    return df, categorical_mode


def one_hot_encode(df, column_header):
    """One-hot encodes columns with categorical values into encoded values

    Args:
        df: The original dataset as a pandas dataframe.
        column_header: Column to perform one-hot encoding on.

    Returns:
        df: One-hot encoded dataframe
        encoder: An sklearn.preprocessing.LabelBinarizer object
    """
    encoder = OneHotEncoder(sparse=False)
    encoder = encoder.fit(df[column_header].to_numpy().reshape(-1, 1))
    encoded_data = encoder.transform(df[column_header].to_numpy().reshape(-1, 1))
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out())
    df = pd.concat(
        [df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1
    )
    df = df.drop(columns=[column_header])
    return df, encoder


def generate_plots(df_preprocessed, arguments):  # pragma: no cover
    """Generates plots for all numerical columns"""
    x_axis = df_preprocessed.values[:, 0]
    sensor_columns = [
        sensor_column.strip()
        for sensor_column in arguments.sensordata_columns.split(",")
    ]
    for i, column in enumerate(sensor_columns):
        plt.figure()
        y_axis = df_preprocessed.values[:, i + 1]
        plt.xlabel("Time Instance")
        plt.ylabel(column)
        plt.plot(x_axis, y_axis)
        anomaly_plot = plt.scatter(
            df_preprocessed.values[
                df_preprocessed[df_preprocessed.columns[-1]]
                == float(arguments.positive_label),
                0,
            ],
            df_preprocessed.values[
                df_preprocessed[df_preprocessed.columns[-1]]
                == float(arguments.positive_label),
                i + 1,
            ],
            c="red",
            label="Anomaly",
        )
        plt.legend([anomaly_plot], ["Anomaly"])
        plt.savefig(arguments.output_dir + "/" + column + "_plot.jpg")
        plt.close()


def preprocess_main():  # pragma: no cover
    """Command line execution."""
    # Get parameters, create dataframe and perfrom validation
    args = parse_parameters()
    validate_arguments(args)
    df = pd.read_csv(args.dataset_path)
    validate_dataset(df)

    # Impute labels and remove missing timestamps
    df = impute_labels(df, args.label_column)
    df = remove_missing_timestamps(df, args.timestamp_column)

    # Filter columns based on user input and store column names for subsequent tasks
    df, column_names = filter_dataset(
        df,
        args.timestamp_column,
        args.sensordata_columns,
        args.categorical_columns,
        args.label_column,
    )
    column_df = pd.DataFrame(column_names, columns=["column_names"])
    column_df.to_csv(args.output_dir + "/column_names.csv", index=False)

    # Add float64 timestamp column to dataframe
    df = datetime_to_timestamp(df, args.timestamp_column)

    # Deal with missing values
    imputers = []
    if args.imputation.lower() == "none":
        df = remove_missing_sensordata(df)
    else:
        imputations = [int(method.strip()) for method in args.imputation.split("!")]
        for i, impute_id in enumerate(imputations):
            if impute_id == 1:
                df, imputer = impute_mean(df, df.columns[i + 1])
            elif impute_id == 2:
                df, imputer = impute_median(df, df.columns[i + 1])
            elif impute_id == 3:
                df, imputer = impute_knn(df, df.columns[i + 1])
            imputers.append(imputer)

    # Store categorical column names for subsequent tasks, impute and one-hot encode categorical values
    encoders = []
    categorical_columns = []
    categorical_modes = []
    if args.categorical_columns.lower() != "none":
        categorical_columns = [
            categorical_column.strip()
            for categorical_column in args.categorical_columns.split(",")
        ]
        categorcial_column_df = pd.DataFrame(
            categorical_columns, columns=["categorical_column_names"]
        )
        categorcial_column_df.to_csv(
            args.output_dir + "/categorical_column_names.csv", index=False
        )
        for categorical_column in categorical_columns:
            df, column_mode = impute_categorical(df, categorical_column)
            categorical_modes.append(column_mode)
            df, ohencoder = one_hot_encode(df, categorical_column)
            encoders.append(ohencoder)
        # Set label column as last column of dataframe
        label_df = df[args.label_column]
        df = df.drop(columns=[args.label_column])
        df = pd.concat(
            [df.reset_index(drop=True), label_df.reset_index(drop=True)], axis=1
        )
        # Save a csv with categorical modes for future imputation
        mode_df = pd.DataFrame(categorical_modes, columns=["categorical_modes"])
        mode_df.to_csv(args.output_dir + "/categorical_modes.csv", index=False)

    # Drop float64 timestamp column
    df = df.drop(columns=["ts"])

    # Save the preprocessed dataset in csv format
    df.to_csv(args.output_dir + "/preprocessed.csv", index=False)

    # Save trained imputer in Pickle format as <column_name>_imputer.pkl
    for i, imputer in enumerate(imputers):
        joblib.dump(
            imputer, args.output_dir + "/" + str(df.columns[i + 1]) + "_imputer.pkl"
        )

    # Save trained one-hot encoders in Pickle format as <column_name>_encoder.pkl
    for i, encoder in enumerate(encoders):
        joblib.dump(
            encoder, args.output_dir + "/" + categorical_columns[i] + "_encoder.pkl"
        )

    # Save plots in the format date-time vs column_name
    generate_plots(df, args)


if __name__ == "__main__":
    preprocess_main()
