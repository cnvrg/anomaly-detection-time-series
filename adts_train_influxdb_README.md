The InfluxDB Anomaly Detection Time-Series (ADTS) Training Blueprint provides an end-to-end framework for pulling data resources from InfluxDB and training anomaly detection models on time-series data. The best-performing trained model can be used for making predictions by making API calls to an inference endpoint. Furthermore, users can also perform batch prediction on large test datasets. Complete the following steps to run this blueprint:

1. Click the **Use Blueprint** button.
2. Select a suitable Compute Template to run the blueprint libraries.
3. Enter and/or edit input arguments to the blueprint libraries and save the changes. Refer to the documentation pages of libraries for more details.
4. Click the **Run** button to launch the end-to-end framework.
5. Once the inference endpoint is ready, use either the Try it Live feature or the Integration panel to integrate the endpoint into your code.

Note: This blueprint may launch multiple experiments with the same title while performing hyperparameter optimization. The best-performing model is selected based on the average precision score metric.

Click [here](https://github.com/cnvrg/anomaly-detection-timeseries) for more information on this Blueprint.