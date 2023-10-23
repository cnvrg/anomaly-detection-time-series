# Anomaly Detection for Time-Series (ADTS) Preprocessing
## _cnvrg_

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

The ADTS Preprocess library performs several preprocessing techniques on time-series datasets. Some techniques supported by this library are dataset validation, column selection, data imputation and one-hot encoding of categorical values. As part of the [ADTS Training Blueprint](), this library preprocesses the input dataset and makes it accessible to subsequent libraries in the Blueprint.

Click [here](https://github.com/cnvrg/anomaly-detection-timeseries/tree/main/adts-preprocess) for more information on this library.

## Library Flow
The following list outlines this library's high-level flow:
- The user defines a set of input arguments to specify the dataset and preprocessing techniques to be used.
- The library then validates these arguments and performs the required techniques on the input dataset.
- The library also stores several CSV and Pickle files which can be used by other libraries in the Blueprint.

## Inputs
This library assumes that the user has access to the original dataset via Connectors. The input dataset must be in CSV format.
The ADTS Preprocess library requires the following inputs:
* `--dataset_path` - string, required. Provide the path to the input dataset in CSV format.
* `--timestamp_column` - string, required. Provide the column name for column having timestamp/date/time values.
* `--sensordata_columns` - string, required. Provide comma-separated column names for columns with sensor data values.
* `--label_column` - string, required. Provide the column name for column containing labels. For eg:- binary labels (1/0) for anomalies.
* `--imputation` - string, optional. Provide comma-separated data imputation techniques (1-mean, 2-median, 3-KNN) for all sensor data columns. The library drops rows with missing sensor data if this parameter is not provided. Default value: `None`.
* `--categorical_columns` - string, optional. Provide comma-separated column names for columns with categorical values. Default value: `None`.
* `--positive_label` - string, optional. Label for anomalous samples. Default value: `1`.
* `--negative_label` - string, optional. Label for non-anomalous samples. Default value: `0`.

Note: Please enclose all comma-seperated string values with double quotes while passing arguments through the UI. Eg:- `"Current,Voltage,Temperature"`, `"1,2,3,3"`

## Sample Command
Refer to the following sample command:

```bash
python preprocess.py python preprocess.py --dataset_path /input/s3_connector/data.csv --timestamp_column datetime --sensordata_columns Current,Voltage,Temperature,Pressure --label_column anomaly --imputation 1,2,3,4 --categorical_columns status --positive_label 1 --negative_label 0
```

## Outputs
The ADTS Preprocess library generates the following outputs:
- The library generates a CSV file `preprocessed.csv`. The preprocessed dataset contains relevant column names and imputed missing values based on the input arguments.
- The library writes all files created to the default path `/cnvrg`.
- Several imputer (`<column_name>_imputer.pkl`) and one-hot encoder (`<categorical_column_name>_encoder.pkl`) objects are stored in Pickle format along with CSV files.
- The library also saves data visualization plots as .jpg files.

## Troubleshooting
- Ensure the input arguments to the library are valid and accurate.
- Check the experiment's Artifacts section to confirm the library has generated the output files.