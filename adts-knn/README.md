# Anomaly Detection for Time-Series (ADTS) KNN Model
## _cnvrg_

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

The ADTS KNN library trains and validates a k-nearest neighbor anomaly detection model. This model has tunable hyperparameters which give users more control over the training process. Performance metrics such as training/validation accuracy, training/validation AUC score and training/validation average precision score are dispalyed at the end of the experiment.

Click [here](https://github.com/cnvrg/anomaly-detection-timeseries/tree/main/adts-knn) for more information on this library.

## Library Flow
The following list outlines this library's high-level flow:
- The user defines a set of input arguments that includes paths to the train/validation sets, hyperparameters, positive labels and negative labels.
- The library then validates these arguments and computes the contamination ratio (No. of anomalies/Size of training dataset).
- Once the model has been trained and validated, it is saved in Pickle format as `model.pkl`.

## Inputs
This library assumes that the user has access to the training and validation datasets. The input datasets should be in CSV format.
The ADTS KNN library requires the following inputs:
* `--train_path` - string, required. Provide the path to the training dataset in CSV format.
* `--valid_path` - string, required. Provide the path to the validation dataset in CSV format.
* `--num_neighbors` - string, optional. This is a hyperparameter which represents the number of neighbors to consider while training and inference. Default value: `5`.
* `--knn_radius` - string, optional. This is a hyperparameter which represents the range of the parameter space while computing distances. Default value: `1`.
* `--positive_label` - string, optional. Label for anomalous samples. Default value: `1`.
* `--negative_label` - string, optional. Label for non-anomalous samples. Default value: `0`.

Note: Cnvrg's hyperparamter optimization feature can be activated by passing comma-separated values to hyperparameter arguments on the UI. Eg:- `1,2,3,4`

## Sample Command
Refer to the following sample command:

```bash
python knn.py --train_path /input/train_valid-split/train.csv --valid_path /input/train_valid-split/valid.csv --num_neighbors 5 --knn_radius 1 --positive_label 1 --negative_label 0
```

## Outputs
The ADTS KNN library generates the following outputs:
- The library saves the trained model in Pickle format as `model.pkl`.
- The library writes this output file to the default path `/cnvrg`.
- The library also saves data visualization plots as .jpg files.

## Troubleshooting
- Ensure the input arguments to the library are valid and accurate.
- In case of hyperparamter optimization, ensure the library launches multiple experiments.
- Check the experiment's Artifacts section to confirm the library has generated the output files.
