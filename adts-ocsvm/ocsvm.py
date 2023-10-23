import argparse
import os, joblib, yaml
import pandas as pd
import sys

from modules.common import *

from cnvrgv2 import Experiment
from pyod.models.ocsvm import OCSVM
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    roc_auc_score,
)

def parse_parameters():
    """Command line parser."""
    parser = genCommonParser('OCSVM Model')
    parser.add_argument(
        "--nu",
        action="store",
        dest="nu",
        required=False,
        default="0.5",
        help="""--- an upper bound on the fraction of training errors and a lower bound on the fraction of support vectors ---""",
    )
    return parser.parse_args()

def ocsvm_main():  # pragma: no cover
    """Command line execution."""
    cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")

    # Read config file
    with open("ocsvm_config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Get user arguments and perform validation
    args = parse_parameters()
    validate_arguments(args)
    df_train = pd.read_csv(args.train_path)
    df_valid = pd.read_csv(args.valid_path)

    # Create a Cnvrg Experiment
    exp = Experiment()

    # Computate contamination in training set
    contamination = get_contamination(df_train, args.positive_label)

    # Separate features and labels. Also map positive/negative labels to 1/0
    train_x, train_y, valid_x, valid_y = split_features_labels(df_train, df_valid)
    train_y, valid_y = map_labels(
        train_y, valid_y, args.positive_label, args.negative_label
    )

    # Define iForest Model
    ocsvm_model = OCSVM(kernel="rbf", nu=float(args.nu), contamination=contamination)

    # Load standard scaler and normalize train and validation features
    scaler = joblib.load(config["scaler_file"])
    train_x = scaler.transform(train_x)
    valid_x = scaler.transform(valid_x)

    # Perform training and print training metrics
    ocsvm_model.fit(train_x)
    train_pred = ocsvm_model.predict(train_x)
    train_decisionfunc = ocsvm_model.decision_function(train_x)
    print(f"Training Accuracy: {accuracy_score(train_y, train_pred)}")
    print(f"Training Classification Report:")
    print(classification_report(train_y, train_pred))
    print(f"Training ROC Score: {roc_auc_score(train_y, train_decisionfunc)}")
    print(
        f"Training Average Precision Score: {average_precision_score(train_y, train_decisionfunc)}"
    )

    # Perform validation and log validation metrics
    valid_pred = ocsvm_model.predict(valid_x)
    valid_decisionfunc = ocsvm_model.decision_function(valid_x)
    print(f"Validation Accuracy: {accuracy_score(valid_y, valid_pred)}")
    print(f"Validation Classification Report:")
    print(classification_report(valid_y, valid_pred))
    valid_rocscore = roc_auc_score(valid_y, valid_decisionfunc)
    valid_precisionscore = average_precision_score(valid_y, valid_decisionfunc)
    print(f"Validation ROC Score: {valid_rocscore}")
    print(f"Validation Average Precision Score: {valid_precisionscore}")
    exp.log_param("average_precision_score", valid_precisionscore)

    # Save model in pickle format
    joblib.dump(ocsvm_model, args.output_dir + "/model.pkl")

    # Save plots showing model predictions on train and validation sets
    generate_plots(config, df_train, df_valid, train_pred, valid_pred, args)

if __name__ == "__main__":
    ocsvm_main()
