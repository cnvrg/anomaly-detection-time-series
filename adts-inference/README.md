# Anomaly Detection for Time-Series (ADTS) Inference
## _cnvrg_

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

The ADTS Inference library enables the user to run inference by setting up an endpoint. Curl commands or Cnvrg's `Try it Live` feature can be used to make API calls to the endpoint. This library can be used as part as the [ADTS Training Blueprint]() as well as the standalone [ADTS Inference Blueprint]().

Click [here](https://github.com/cnvrg/anomaly-detection-timeseries/tree/main/adts-inference) for more information on this library.

## Library Flow
The following list outlines this library's high-level flow:
- The user passes an input to the endpoint.
- The model on the backend makes a prediction and displays it to the user.

## Inputs
The user can make calls to the API using the following example curl commands:
* The user can pass a comma-separated (without white-spaces after commas) string as input to the endpoint. However, the values should only be from columns selected during the preprocessing task.  
```bash
curl -X POST \
    {link to your deployed endpoint} \
-H 'Cnvrg-Api-Key: {your_api_key}' \
-H 'Content-Type: application/json' \
-d '{"vars": "2020-03-01 17:03:57,100.0,200.0,300.0,400.0,active"}'
```
* This library can also impute missing values with the help of imputer/encoder artifacts from previous libraries.
```bash
curl -X POST \
    {link to your deployed endpoint} \
-H 'Cnvrg-Api-Key: {your_api_key}' \
-H 'Content-Type: application/json' \
-d '{"vars": "2020-03-01 17:03:57,,200.0,300.0,,inactive"}'
```

## Output
The sample output looks as follows:
```bash
{"anomaly": "1"}
```