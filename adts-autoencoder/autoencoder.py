import argparse
import os, joblib, yaml
import pandas as pd
from modules.common import *

from cnvrgv2 import Experiment
from keras.losses import mean_squared_error
from pyod.models.auto_encoder import AutoEncoder
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    roc_auc_score,
)

def parse_parameters():
    """Command line parser."""
    parser = genCommonParser('AutoEncoder Model')
    parser.add_argument(
        "--dropout",
        action="store",
        dest="dropout",
        required=False,
        default="0.2",
        help="""--- dropout rate for deep neural network ---""",
    )
    parser.add_argument(
        "--l2_reg",
        action="store",
        dest="l2_reg",
        required=False,
        default="0.1",
        help="""--- L2 regularization constant for deep neural network ---""",
    )
    return parser.parse_args()

def autoencoder_main():  # pragma: no cover
    """Command line execution."""
    cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")

    # Read config file
    with open("autoencoder_config.yaml", "r") as file:
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
    num_features = len(train_x[0])
    autoencoder_model = AutoEncoder(
        hidden_neurons=[num_features, 64, 32, 16, 16, 32, 64, num_features],
        loss=mean_squared_error,
        dropout_rate=float(args.dropout),
        l2_regularizer=float(args.l2_reg),
        contamination=contamination,
        verbose=0,
    )

    # Load standard scaler and normalize train and validation features
    scaler = joblib.load(config["scaler_file"])
    train_x = scaler.transform(train_x)
    valid_x = scaler.transform(valid_x)

    # Perform training and print training metrics
    autoencoder_model.fit(train_x)
    train_pred = autoencoder_model.predict(train_x)
    train_decisionfunc = autoencoder_model.decision_function(train_x)
    print(f"Training Accuracy: {accuracy_score(train_y, train_pred)}")
    print("Training Classification Report:")
    print(classification_report(train_y, train_pred))
    print(f"Training ROC Score: {roc_auc_score(train_y, train_decisionfunc)}")
    print(
        f"Training Average Precision Score: {average_precision_score(train_y, train_decisionfunc)}"
    )

    # Perform validation and log validation metrics
    valid_pred = autoencoder_model.predict(valid_x)
    valid_decisionfunc = autoencoder_model.decision_function(valid_x)
    print(f"Validation Accuracy: {accuracy_score(valid_y, valid_pred)}")
    print("Validation Classification Report:")
    print(classification_report(valid_y, valid_pred))
    valid_rocscore = roc_auc_score(valid_y, valid_decisionfunc)
    valid_precisionscore = average_precision_score(valid_y, valid_decisionfunc)
    print(f"Validation ROC Score: {valid_rocscore}")
    print(f"Validation Average Precision Score: {valid_precisionscore}")
    exp.log_param("average_precision_score", valid_precisionscore)
    exp.log_param("threshold", autoencoder_model.threshold_)

    # Save model
    autoencoder_model.model_.save(args.output_dir + "/model")

    # Save plots showing model predictions on train and validation sets
    generate_plots(config, df_train, df_valid, train_pred, valid_pred, args)

if __name__ == "__main__":
    autoencoder_main()
