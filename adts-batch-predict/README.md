# Anomaly Detection for Time-Series (ADTS) Batch Predict
## _cnvrg_

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

The ADTS Batch Predict library enables the user to make batch predictions on unseen time-series data. A 'predictions' column is added to the test dataset which needs to be in CSV format. As part of the [ADTS Training Blueprint](), this library enables the user to make predictions using a pre-trained model.

Click [here](https://github.com/cnvrg/anomaly-detection-timeseries/tree/main/adts-batch-predict) for more information on this library.

## Library Flow
The following list outlines this library's high-level flow:
- The user defines a single input argument to specify the path to the test dataset.
- The library drops rows with missing time instances.
- The library then performs batch prediction and saves artifacts.

## Inputs
This library assumes that the user has access to the test dataset. The dataset can be pulled either via Connectors or other libraries in the Blueprint.
The ADTS Batch Predict library requires the following inputs:
- `--dataset_path` - string, required. Provide the path to the test dataset in CSV format.

Note: Please ensure the test dataset does not have a labels column.

## Sample Command

```bash
python batchpredict.py --dataset_path /input/s3_connector/test_data.csv
```

## Outputs
The ADTS Batch Predict library generates the following outputs:
- The library generates a CSV file `predictions.csv`. As the name suggests, it consists of relevant columns and predictions (1/0).
- The library writes all files created to the default path `/cnvrg`.
- The library also saves data visualization plots as .jpg files.

## Troubleshooting
- Ensure the input argument to the library is valid and accurate.
- Check the experiment's Artifacts section to confirm the library has generated the output files.