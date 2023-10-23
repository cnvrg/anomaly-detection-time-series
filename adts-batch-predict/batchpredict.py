import argparse
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import yaml
from tensorflow import keras

cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")

# Read config file
with open("batchpredict_config.yaml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


class InsufficientColumnsError(Exception):
    """Raise if original dataset has less than 2 columns"""

    def __init__(self, num_columns):
        super().__init__(num_columns)
        self.num_columns = num_columns

    def __str__(self):
        return f"InsufficientColumnsError: Your dataset contains {self.num_columns} columns. The original dataset should have atleast two columns i.e. timestamp and sensor data."


class ColumnDoesNotExistError(Exception):
    """Raise if column header does not exist in dataframe"""

    def __init__(self, column_name):
        super().__init__(column_name)
        self.column_name = column_name

    def __str__(self):
        return f"ColumnDoesNotExistError: {self.column_name} does not exist in dataset!"


def parse_parameters():
    """Command line parser."""
    parser = argparse.ArgumentParser(description="""Batch Predict""")
    parser.add_argument(
        "--dataset_path",
        action="store",
        dest="dataset_path",
        required=True,
        help="""--- Path to the test dataset in csv format ---""",
    )
    parser.add_argument(
        "--output_dir",
        action="store",
        dest="output_dir",
        required=False,
        default=cnvrg_workdir,
        help="""--- The path to save dataset file to ---""",
    )
    return parser.parse_args()


def validate_dataset(df):
    """Performs validation on test dataset

    Checks if the dataset has atleast two columns

    Args:
        df: The test dataset as a pandas dataframe

    Raises:
        InsufficientColumnsError: If dataset contains less than two columns
    """
    if len(df.columns) < 2:
        raise InsufficientColumnsError(len(df.columns))


def get_column_names(library_path, file_name, column_header):
    """Fetches list of relevant/categorical column names from preprocessing library"""
    columns_list = []
    file_path = library_path + file_name
    if os.path.exists(file_path):
        columns_df = pd.read_csv(file_path)
        columns_list = columns_df[column_header].values.tolist()
    return columns_list


def get_numerical_columns(relevant_columns, categorical_columns):
    """Returns a list containing numerical column names"""
    return [
        column for column in relevant_columns[1:] if column not in categorical_columns
    ]


def get_categorical_modes(categorical_columns, library_path):
    """Creates a dictionary mapping of categorical column names to modes for imputation on test"""
    mode_dict = {}
    file_path = library_path + config["categorical_modes_csv"]
    if os.path.exists(file_path):
        categorical_modes_df = pd.read_csv(file_path)
        categorical_modes_list = categorical_modes_df[
            "categorical_modes"
        ].values.tolist()
        for i, column in enumerate(categorical_columns):
            mode_dict[column] = categorical_modes_list[i]
    return mode_dict


def get_pkl_objects(columns, library_path, pkl_file_name):
    """Creates a dictionary mapping of column names to imputers/encoders from the preprocessing library"""
    pkl_dict = {}
    for column_name in columns:
        pkl_path = library_path + column_name + pkl_file_name
        if os.path.exists(pkl_path):
            imputer = joblib.load(pkl_path)
            pkl_dict[column_name] = imputer
    return pkl_dict


def load_model(library_path):
    """Fetches the pre-trained ML model and threshold (only applicable to deep models) from the compare library"""
    deepmodel_path = library_path + config["deepmodel_dir"]
    model_path = library_path + config["model_file"]
    winnerdetails_path = library_path + config["winner_csv"]
    model = None
    threshold = None
    if os.path.exists(deepmodel_path):
        model = keras.models.load_model(deepmodel_path)
        winner_details = pd.read_csv(winnerdetails_path)
        threshold = float(winner_details["threshold"].values[0])
    else:
        model = joblib.load(model_path)
    return model, threshold


def filter_dataset(df, column_names):
    """Selects relevant columns from the test dataset"""
    for column_name in column_names:
        if column_name not in df.columns:
            raise ColumnDoesNotExistError(column_name)
    return df[column_names]


def remove_missing_timestamps(df, timestamp_column):
    """Removes row entries having missing timestamp values from a pandas dataframe."""
    return df.dropna(subset=[timestamp_column])


def dataframe_to_list(df):
    """Converts a pandas dataframe to a list for running machine learning techniques"""
    columns = df.columns
    df_list = df[columns[1:]].values.tolist()
    return df_list


def load_scaler(library_path):
    """Fetches a Standard Scaler object trained on the training set"""
    scaler = joblib.load(library_path + config["scaler_file"])
    return scaler


def batch_predict(features, model, threshold):
    """Returns predictions for the test dataset"""
    predictions = model.predict(features)
    if threshold is not None:
        reconstruction_error = np.linalg.norm(features - predictions, axis=1)
        predictions = (reconstruction_error > threshold).astype("int").ravel()
    return predictions


def generate_plots(df, columns, arguments):
    """Generates plots for all numerical columns"""
    x_axis = df.values[:, 0]
    for i, column in enumerate(columns):
        plt.figure()
        y_axis = df.values[:, i + 1]
        plt.xlabel("Time Instance")
        plt.ylabel(column)
        plt.plot(x_axis, y_axis)
        anomaly_plot = plt.scatter(
            df.values[df[df.columns[-1]] == 1, 0],
            df.values[df[df.columns[-1]] == 1, i + 1],
            c="red",
            label="Anomaly",
        )
        plt.legend([anomaly_plot], ["Anomaly"])
        plt.savefig(arguments.output_dir + "/" + column + "_predictions_plot.jpg")
        plt.close()


def batchpredict_main():  # pragma: no cover
    """Command line execution."""
    # Get parameters and load the test dataset
    args = parse_parameters()
    test_df = pd.read_csv(args.dataset_path)
    validate_dataset(test_df)

    # Define preprocessing library path
    preprocessing_path = config["preprocessing_lib"]

    # Get original column names and corresponding imputer objects from preprocessing library
    relevant_columns = get_column_names(
        preprocessing_path, config["column_names_csv"], "column_names"
    )[:-1]
    imputer_dict = get_pkl_objects(
        relevant_columns, preprocessing_path, config["imputer_file_name"]
    )

    # Get categorical column names, column-wise modes and one-hot encoder objects
    categorical_columns = get_column_names(
        preprocessing_path, config["categorical_names_csv"], "categorical_column_names"
    )
    mode_dict = get_categorical_modes(categorical_columns, preprocessing_path)
    encoder_dict = get_pkl_objects(
        categorical_columns, preprocessing_path, config["encoder_file_name"]
    )

    # Get numerical columns
    numerical_columns = get_numerical_columns(relevant_columns, categorical_columns)

    # Define model comparison library path and load saved model objects
    comparison_path = config["comparison_lib"]
    anomaly_detector, anomaly_threshold = load_model(comparison_path)

    # Load Standard Scaler trained on training data
    train_valid_path = config["train_valid_lib"]
    scaler = load_scaler(train_valid_path)

    # Choose relevant columns from test dataset
    test_df = filter_dataset(test_df, relevant_columns)

    # Remove rows with missing date/time values
    test_df = remove_missing_timestamps(test_df, test_df.columns[0])

    # Add float64 timestamp column to input dataframe
    test_df["ts"] = (
        pd.to_datetime(test_df[test_df.columns[0]]).values.astype(np.float64) // 10**9
    )

    # Impute missing values in numerical columns
    for column in numerical_columns:
        if column in imputer_dict:
            test_df[["ts", column]] = imputer_dict[column].transform(
                test_df[["ts", column]]
            )

    # Delete float64 timestamp column
    test_df = test_df.drop(columns=["ts"])

    # Impute and one-hot encode categorical values
    ohencode_column_names = []
    categorical_column_dict = {}
    for column in categorical_columns:
        if column in mode_dict:
            test_df[column] = test_df[column].fillna(mode_dict[column])
        if column in encoder_dict:
            ohencoded_data = encoder_dict[column].transform(
                test_df[column]
                .to_numpy()
                .reshape(
                    -1, 1
                )  # convert column to numpy column vector for tranformation
            )
            ohencoded_df = pd.DataFrame(
                ohencoded_data, columns=encoder_dict[column].get_feature_names_out()
            )
            test_df = pd.concat(
                [test_df.reset_index(drop=True), ohencoded_df.reset_index(drop=True)],
                axis=1,
            )
            categorical_column_dict[column] = test_df[column]
            test_df = test_df.drop(columns=[column])
            ohencode_column_names = ohencode_column_names + list(
                encoder_dict[column].get_feature_names_out()
            )

    # Convert dataframe to list for performing batch prediction
    test_features = dataframe_to_list(test_df)

    # Standardize the test features
    test_features = scaler.transform(test_features)

    # Run batch prediction on the test dataset and append predictions to the pandas dataframe
    batch_preds = batch_predict(test_features, anomaly_detector, anomaly_threshold)
    display_df = test_df.drop(columns=ohencode_column_names)
    for column in categorical_column_dict:
        display_df = pd.concat(
            [
                display_df.reset_index(drop=True),
                categorical_column_dict[column].reset_index(drop=True),
            ],
            axis=1,
        )
    display_df["prediction"] = pd.DataFrame(batch_preds, columns=["prediction"])

    # Save dataframe with predictions in csv format
    display_df.to_csv(args.output_dir + "/predictions.csv", index=False)

    # Save plots in the format date-time vs column_name
    generate_plots(display_df, numerical_columns, args)


if __name__ == "__main__":
    batchpredict_main()
