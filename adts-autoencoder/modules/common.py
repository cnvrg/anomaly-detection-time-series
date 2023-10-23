import argparse, os
import matplotlib.pyplot as plt
import pandas as pd

class NoCSVError(Exception):
    """Raise if datasets are not in CSV format"""

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "NoCSVError: Both training and validation sets need to be in csv format for preprocessing!"


class ContaminationValueError(Exception):
    """Raise if contamination value is not between 0.0 and 0.5"""

    def __init__(self, contamination_value):
        super().__init__(contamination_value)
        self.contamination_value = contamination_value

    def __str__(self):
        return f"ContaminationValueError: The contamination value is {self.contamination_value}. Contamination cannot be 0 or greater than or equal to 0.5!"

def genCommonParser(model_name):
    """Command line parser."""
    parser = argparse.ArgumentParser(description=f'"{model_name}"') 

    parser.add_argument(
        "--train_path",
        action="store",
        dest="train_path",
        required=False,
        default="/input/adts_train_valid_split/train.csv",
        help="""--- Path to the original dataset in csv format ---""",
    )
    parser.add_argument(
        "--valid_path",
        action="store",
        dest="valid_path",
        required=False,
        default="/input/adts_train_valid_split/valid.csv",
        help="""--- Path to the original dataset in csv format ---""",
    )

    cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")
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
    return parser

def validate_arguments(args):
    """Validates input arguments

    Checks if the train dataset and validation dataset paths points to csv files

    Args:
        args: argparse object

    Raises:
        NoCSVError: If train and validation sets are not in csv format
    """
    if not (".csv" in args.train_path and ".csv" in args.valid_path):
        raise NoCSVError()


def get_contamination(train_df, positive_label):
    """Computes the contamination (No. of anomalies/Size of dataset) value from the training set

    Args:
        train_df: training set as pandas dataframe
        positive_label: positive_label value

    Raises:
        ContaminationValueError: If contamination is not between 0 and 0.5 (50%)

    Returns:
        contamination: contamination value
    """
    columns = train_df.columns
    num_anomalies = train_df[columns[-1]].value_counts()[float(positive_label)]
    dataset_size = train_df.shape[0]
    contamination = num_anomalies / dataset_size
    if not (contamination > 0 and contamination < 0.5):
        raise ContaminationValueError(contamination)
    return contamination


def split_features_labels(train_df, valid_df):
    """Splits training and validation sets into features and labels

    Removes the timestamp column while performing this step.

    Args:
        train_df: pandas dataframe for training set
        valid_df: pandas dataframe for valid set

    Returns:
        trainx: training features as a list
        trainy: training labels as a list
        validx: validation_features as a list
        validy: validation_labels as a list
    """
    train_columns = train_df.columns
    valid_columns = valid_df.columns
    trainx = train_df[train_columns[1:-1]].values.tolist()
    trainy = train_df[train_columns[-1]].values.tolist()
    validx = valid_df[valid_columns[1:-1]].values.tolist()
    validy = valid_df[valid_columns[-1]].values.tolist()
    return trainx, trainy, validx, validy


def map_labels(trainy, validy, positive_label, negative_label):
    """Maps positive labels to 1 and negative labels to 0"""
    for i in range(len(trainy)):
        if trainy[i] == float(positive_label):
            trainy[i] = 1
        elif trainy[i] == float(negative_label):
            trainy[i] = 0

    for i in range(len(validy)):
        if validy[i] == float(positive_label):
            validy[i] = 1
        elif validy[i] == float(negative_label):
            validy[i] = 0
    return trainy, validy


def generate_plots(config, df_train, df_valid, train_pred, valid_pred, args):
    """Generates plots for all numerical columns in train and validation sets"""
    df_train["prediction"] = pd.DataFrame(train_pred, columns=["prediction"])
    df_valid["prediction"] = pd.DataFrame(valid_pred, columns=["prediction"])
    num_total_columns = pd.read_csv(config["column_csv"]).shape[0]
    num_categorical_columns = 0
    if os.path.exists(config["categorical_csv"]):
        num_categorical_columns = pd.read_csv(config["categorical_csv"]).shape[0]
    num_numerical_columns = num_total_columns - 2 - num_categorical_columns
    train_xaxis = df_train.values[:, 0]
    for i in range(num_numerical_columns):
        plt.figure()
        train_yaxis = df_train.values[:, i + 1]
        plt.xlabel("Time Instance")
        plt.ylabel(str(df_train.columns[i + 1]))
        plt.plot(train_xaxis, train_yaxis)
        anomaly_plot = plt.scatter(
            df_train.values[df_train[df_train.columns[-1]] == 1, 0],
            df_train.values[df_train[df_train.columns[-1]] == 1, i + 1],
            c="red",
            label="Anomaly",
        )
        plt.legend([anomaly_plot], ["Anomaly"])
        plt.savefig(
            args.output_dir
            + "/"
            + "train_"
            + str(df_train.columns[i + 1])
            + "_plot.jpg"
        )
        plt.close()
    valid_xaxis = df_valid.values[:, 0]
    for i in range(num_numerical_columns):
        plt.figure()
        valid_yaxis = df_valid.values[:, i + 1]
        plt.xlabel("Time Instance")
        plt.ylabel(str(df_valid.columns[i + 1]))
        plt.plot(valid_xaxis, valid_yaxis)
        anomaly_plot = plt.scatter(
            df_valid.values[df_valid[df_valid.columns[-1]] == 1, 0],
            df_valid.values[df_valid[df_valid.columns[-1]] == 1, i + 1],
            c="red",
            label="Anomaly",
        )
        plt.legend([anomaly_plot], ["Anomaly"])
        plt.savefig(
            args.output_dir
            + "/"
            + "valid_"
            + str(df_valid.columns[i + 1])
            + "_plot.jpg"
        )
        plt.close()
