# Anomaly Detection for Time-Series (ADTS) Model Comparison
## _cnvrg_

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

The ADTS Compare library selects the best-performing model based on the average precision score metric. The selected model can then be used for subsequent tasks such as inference using an endpoint.

Click [here](https://github.com/cnvrg/anomaly-detection-timeseries/tree/main/adts-compare) for more information on this library.

## Library Flow
The following list outlines this library's high-level flow:
- The library iterates through all the environment variables/saved models and chooses the model with the highest average precision score.
- It moves the saved model and other artifacts to from previous libraries to the current working directory.

## Inputs
This library mainly operates on artifacts from previous libraries and does not require any input arguments.

## Sample Command
Refer to the following sample command:

```bash
python compare.py
```

## Outputs
The ADTS Compare library does not generate any output artifacts by itself. It moves artifacts from previous libraries to the working directory `/cnvrg`.

## Troubleshooting
- Check the experiment's Artifacts section to confirm the library has moved the required files.