The Anomaly Detection Time-Series (ADTS) Inference Blueprint provides an API endpoint for making predictions on unseen time-series data samples. This blueprint uses a pretrained anomaly detection model and accepts a user-input comma-separated string (e.g.,- `2020-03-01 17:03:57,100.0,200.0,300.0,400.0,active`). Input strings with missing values (e.g.,- `2020-03-01 17:03:57,,200.0,300.0,,inactive` or `2020-03-01 17:03:57,,,,,`) are imputed with the help of imputer and encoder objects available to the endpoint. Complete the following steps to run this blueprint:

1. Click the **Use Blueprint** button.
2. Select a suitable Compute Template to run the inference endpoint and click the **Start** button.
3. Once the inference endpoint is ready, use either the Try it Live feature or the Integration panel to integrate the endpoint into your code.

Note: This blueprint serves as an example of the inference endpoint and cannot be used to make predictions on other custom data. Use this inference blueprint's [training](https://metacloud.cloud.cnvrg.io/marketplace/blueprints/influxdb-adts-training) counterpart to train anomaly detection models on your own custom time-series data and establish an endpoint based on the newly trained model.