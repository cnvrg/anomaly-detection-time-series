import argparse
import joblib
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")


class ValidationSizeError(Exception):
    """Raise if validation size is not between 0.0 and 0.4"""

    def __init__(self, valid_size):
        super().__init__(valid_size)
        self.valid_size = valid_size

    def __str__(self):
        return f"ValidationSizeError: {self.valid_size} is an invalid validation size. Validation size needs to be a value between 0.0 and 0.4!"


def parse_parameters():
    """Command line parser."""
    parser = argparse.ArgumentParser(description="""Train-Validation Split""")
    parser.add_argument(
        "--dataset_path",
        action="store",
        dest="dataset_path",
        required=False,
        default="/input/adts_preprocess/preprocessed.csv",
        help="""--- Path to the preprocessed dataset in csv format ---""",
    )
    parser.add_argument(
        "--valid_size",
        action="store",
        dest="valid_size",
        required=False,
        default="0.3",
        help="""--- size of validation set as percentage of entire dataset ---""",
    )
    parser.add_argument(
        "--output_dir",
        action="store",
        dest="output_dir",
        required=False,
        default=cnvrg_workdir,
        help="""--- The path to save train and validation dataset files to ---""",
    )
    return parser.parse_args()


def validate_arguments(args):
    """Validates input arguments

    Makes sure that validation size lies between 0.0 and 0.4

    Args:
        args: argparse object

    Raises:
        ValidationSizeError: If validation size does not lie between 0.0 and 0.4
    """
    if not (float(args.valid_size) > 0 and float(args.valid_size) <= 0.4):
        raise ValidationSizeError(float(args.valid_size))


def split_dataset(df, split_size):
    """Splits preprocessed data into train and validation sets

    The train and validation sets are split depending on the intended size of the validation set

    Args:
        df: The preprocessed dataset as a pandas dataframe
        split_size: size of the validation set as percentage of entire dataset

    Returns:
        tdf: The train set as a pandas dataframe
        vdf: The validation set as a pandas dataframe
    """
    dataset_size = df.shape[0]
    valid_size = round(split_size * dataset_size)
    split_index = dataset_size - valid_size
    tdf = df.iloc[:split_index, :]
    vdf = df.iloc[split_index:, :]
    return tdf, vdf


def normalize_data(df):
    """Trains a SimpleImputer on the training dataset and saves it

    Args:
        df: The training dataset as a pandas dataframe

    Returns:
        scaler: An sklearn.preprocessing.SimpleImputer object
    """
    scaler = StandardScaler()
    scaler = scaler.fit(df[df.columns[1:-1]].to_numpy())
    return scaler


def tvsplit_main():  # pragma: no cover
    """Command line execution."""
    # Get parameters and read preprocessed dataset file
    args = parse_parameters()
    validate_arguments(args)
    df = pd.read_csv(args.dataset_path)

    # Perform splitting
    train_df, valid_df = split_dataset(df, float(args.valid_size))

    # Save the train and validation datasets in csv format
    train_df.to_csv(args.output_dir + "/train.csv", index=False)
    valid_df.to_csv(args.output_dir + "/valid.csv", index=False)

    # Train a Standard scaler and save it for future use.
    std_scaler = normalize_data(train_df)
    joblib.dump(std_scaler, args.output_dir + "/train_scaler.pkl")


if __name__ == "__main__":
    tvsplit_main()
