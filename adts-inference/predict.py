import joblib
import numpy as np
import os
import pandas as pd
import yaml
from tensorflow import keras

# Define path to adts-inference library for dev or prod
lib_path = "/cnvrg_libraries/dev-adts-inference/"
if os.path.exists("/cnvrg_libraries/adts-inference"):
    lib_path = "/cnvrg_libraries/adts-inference/"

# Read config file
with open(lib_path + "inference_config.yaml", "r") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


def get_column_names(is_standalone, library_path, file_path, file_name, column_header):
    """Fetches relevant/categorical column names from preprocessing library or inference library depending on is_standalone flag"""
    if is_standalone:
        file_path = library_path + file_name
    if os.path.exists(file_path):
        columns_df = pd.read_csv(file_path)
        columns_list = columns_df[column_header].values.tolist()
        return columns_list
    return []


def get_categorical_modes(categorical_columns, is_standalone, library_path):
    """Creates a dictionary mapping of categorical column names to modes for imputation on test"""
    mode_dict = {}
    file_path = config["categorical_modes_path"]
    if is_standalone:
        file_path = library_path + config["categorical_modes_csv"]
    if os.path.exists(file_path):
        categorical_modes_df = pd.read_csv(file_path)
        categorical_modes_list = categorical_modes_df[
            "categorical_modes"
        ].values.tolist()
        for i, column in enumerate(categorical_columns):
            mode_dict[column] = categorical_modes_list[i]
    return mode_dict


def get_pkl_objects(columns, is_standalone, library_path, pkl_file_name):
    """Creates a dictionary mapping of column names to imputers/encoders from the preprocessing library or inference library depending on is_standalone flag"""
    pkl_dict = {}
    file_path = config["preprocess_lib"] + "/"
    if is_standalone:
        file_path = library_path
    for column_name in columns:
        pkl_path = file_path + column_name + pkl_file_name
        if os.path.exists(pkl_path):
            pkl_object = joblib.load(pkl_path)
            pkl_dict[column_name] = pkl_object
    return pkl_dict


def load_model(is_standalone, library_path):
    """Fetches the pre-trained ML model and threshold (only applicable to deep models) from the compare library or inference library depending on is_standalone flag"""
    file_path = config["comparison_lib"] + "/"
    if is_standalone:
        file_path = library_path
    deepmodel_path = file_path + config["deepmodel_dir"]
    model_path = file_path + config["model_file"]
    winnerdetails_path = file_path + config["winner_csv"]
    model = None
    threshold = None
    if os.path.exists(deepmodel_path):
        model = keras.models.load_model(deepmodel_path)
        winner_details = pd.read_csv(winnerdetails_path)
        threshold = float(winner_details["threshold"].values[0])
    else:
        model = joblib.load(model_path)
    return model, threshold


def load_scaler(is_standalone, library_path):
    """Fetches a Standard Scaler object trained on the training set"""
    file_path = config["train_valid_lib"] + "/"
    if is_standalone:
        file_path = library_path
    scaler = joblib.load(file_path + config["scaler_file"])
    return scaler


# Check if library is standalone (inference blueprint) or part of anomaly-detection for timeseries training blueprint
is_standalone = True
if os.path.exists(config["preprocess_lib"]) and os.path.exists(
    config["comparison_lib"]
):
    is_standalone = False

# Get original column names and corresponding imputer objects from preprocessing library
original_columns = get_column_names(
    is_standalone,
    lib_path,
    config["column_csv_path"],
    config["column_names_csv"],
    "column_names",
)
imputer_dict = get_pkl_objects(
    original_columns, is_standalone, lib_path, config["imputer_file_name"]
)

# Get categorical column names, column-wise modes and one-hot encoder objects
categorical_columns = get_column_names(
    is_standalone,
    lib_path,
    config["categorical_csv_path"],
    config["categorical_names_csv"],
    "categorical_column_names",
)
mode_dict = get_categorical_modes(categorical_columns, is_standalone, lib_path)
encoder_dict = get_pkl_objects(
    categorical_columns, is_standalone, lib_path, config["encoder_file_name"]
)

# Load model object outside predict function to avoid memory leak during runtime
anomaly_detector, anomaly_threshold = load_model(is_standalone, lib_path)

# Load Standard Scaler
scaler = load_scaler(is_standalone, lib_path)


def dataframe_to_list(df):
    """Converts a pandas dataframe to a list for running machine learning techniques"""
    columns = df.columns
    df_list = df[columns[1:]].values.tolist()
    return df_list


def make_prediction(features, model, threshold):
    """Returns the prediction input x"""
    prediction = model.predict(features)
    if threshold is not None:
        reconstruction_error = np.linalg.norm(features - prediction)
        prediction = (reconstruction_error > threshold).astype("int").ravel()
    return prediction


def predict(data):
    """Performs prediction on data provided as input to the webservice

    Args:
        data: a json object representing input data

    Returns:
        response: dictionary containing the prediction
    """
    # Read the data point from the json object and create a dataframe
    input_data = data["vars"].split(",")
    input_df = pd.DataFrame([input_data], columns=original_columns[:-1])

    # One-hot encode categorical columns if applicable
    for categorical_column in categorical_columns:
        if categorical_column in mode_dict:
            if input_df.iloc[0][categorical_column] == "":
                input_df.iloc[0][categorical_column] = np.nan
            input_df[categorical_column] = input_df[categorical_column].fillna(
                mode_dict[categorical_column]
            )
        if categorical_column in encoder_dict:
            ohencoded_data = encoder_dict[categorical_column].transform(
                input_df[categorical_column].to_numpy().reshape(-1, 1)
            )
            ohencoded_df = pd.DataFrame(
                ohencoded_data,
                columns=encoder_dict[categorical_column].get_feature_names_out(),
            )
            input_df = pd.concat(
                [input_df.reset_index(drop=True), ohencoded_df.reset_index(drop=True)],
                axis=1,
            )
            input_df = input_df.drop(columns=[categorical_column])

    # Add float64 timestamp column to input dataframe
    input_df["ts"] = (
        pd.to_datetime(input_df[input_df.columns[0]]).values.astype(np.float64)
        // 10**9
    )

    # Explicitly convert string values to float
    for column in input_df.columns[1:]:
        input_df[column] = pd.to_numeric(input_df[column], errors="coerce")

    for column in input_df.columns[1:]:
        if column in imputer_dict:
            input_df[["ts", column]] = imputer_dict[column].transform(
                input_df[["ts", column]]
            )

    # Delete float64 timestamp column
    input_df = input_df.drop(columns=["ts"])

    # Convert dataframe to list for performing predictions
    input_features = dataframe_to_list(input_df)

    # Load standard scaler and normalize test features
    input_features = scaler.transform(input_features)

    # Make a prediction
    pred = make_prediction(input_features, anomaly_detector, anomaly_threshold)

    # Create output dictionary
    response = {}
    response["anomaly"] = str(pred[0])
    return response
