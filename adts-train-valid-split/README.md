# Anomaly Detection for Time-Series (ADTS) Train-Validation Split
## _cnvrg_

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

The ADTS Train-Valid Split library splits the preprocessed dataset into training and validation sets. As this library operates on time-series data of size M, the first N samples are used for training and the remaining M-N samples are used for validation.

Click [here](https://github.com/cnvrg/anomaly-detection-timeseries/tree/main/adts-train-valid-split) for more information on this library.

## Library Flow
The following list outlines this library's high-level flow:
- The user defines a set of input arguments to specify the dataset path and intended size of the validation set.
- The library validates these arguments and then splits the preprocessed dataset into training and validation sets.

## Inputs
This library assumes that the user has access to the preprocessed dataset via previous libraries in the Blueprint. The input dataset must be in CSV format.
The ADTS Train-Valid Split library requires the following inputs:
* `--dataset_path` - string, required. Provide the path to the preprocessed dataset in CSV format.
* `--valid_size` - string, optional. Specify size of the validation set as a number between 0.0 and 1.0. Default value: `0.3`.

## Sample Command
Refer to the following sample command:

```bash
python tvsplit.py --dataset_path /input/preprocessing/preprocessed.csv --valid_size 0.2
```

## Outputs
The ADTS Train-Valid Split library generates the following outputs :
- The library generates CSV files `train.csv` and `valid.csv`.
- The library writes these output files to the default path `/cnvrg`.

## Troubleshooting
- Ensure the input arguments to the library are valid and accurate.
- Check the experiment's Artifacts section to confirm the library has generated the output files.