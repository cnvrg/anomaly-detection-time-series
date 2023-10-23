import os
import argparse
import pandas as pd
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS

cnvrg_workdir = os.environ.get("CNVRG_WORKDIR", "/cnvrg")


def parse_parameters():
    """Command line parser."""
    parser = argparse.ArgumentParser(description="""influxdb Connector""")
    parser.add_argument(
        "--token",
        action="store",
        dest="token",
        required=True,
        help="""--- influxdb API Access Token ---""",
    )
    parser.add_argument(
        "--url",
        action="store",
        dest="url",
        required=True,
        help="""--- influxdb access url ---""",
    )
    parser.add_argument(
        "--org",
        action="store",
        dest="org",
        required=True,
        help="""--- influxdb access organization account ---""",
    )
    parser.add_argument(
        "--bucket",
        action="store",
        dest="bucket",
        required=True,
        help="""--- bucket where the data is pulled from ---""",
    )
    parser.add_argument(
        "--measurement",
        action="store",
        dest="measurement",
        required=True,
        help="""--- measurement name where the data is pulled from ---""",
    )
    parser.add_argument(
        "--time_col",
        action="store",
        dest="time_col",
        required=True,
        help="""--- name of the time column, which will be used for indexing ---""",
    )
    parser.add_argument(
        "--range_start",
        action="store",
        dest="range_start",
        required=False,
        default="-10y",
        help="""--- fetch from starting datetime in the format of 2020-01-01T00:00:00Z, or specify -1d, -1h, -1m for last day, hour, or minute ---""",
    )
    parser.add_argument(
        "--range_end",
        action="store",
        dest="range_end",
        required=False,
        default="now()",
        help="""--- fetch through ending datetime in the format of 2020-01-01T00:00:00Z, or defaults to now() for current time ---""",
    )
    parser.add_argument(
        "--local_dir",
        action="store",
        dest="local_dir",
        required=False,
        default=cnvrg_workdir,
        help="""--- The path to save the dataset file to ---""",
    )
    parser.add_argument(
        "--cnvrg_dataset",
        action="store",
        dest="cnvrg_dataset",
        required=False,
        default="None",
        help="""--- the name of the cnvrg dataset to store in ---""",
    )
    parser.add_argument(
        "--file_name",
        action="store",
        dest="file_name",
        required=False,
        default="influxdb.csv",
        help="""--- name of the dataset csv file ---""",
    )
    return parser.parse_args()


# custom exceptions
class EmptyDataError(Exception):
    """Raise if there is no data"""

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "EmptyDataError: The query resulted in empty data, please check if the bucket is empty"


class IncorrectDirectoryError(Exception):
    """Raise if directory is invalid"""

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "IncorrectDirectoryError: Please ensure the input directory is valid!"


class Influxdb:
    def __init__(self, url, token, org):
        self.client = InfluxDBClient(url=url, token=token, org=org)

    def ping(self):
        """
        Returns bool depending on the status of connection
        Returns:
            True if connection is established, else false
        """
        return self.client.ping()

    def get_data(
        self,
        org,
        bucket,
        measurement,
        time_col,
        range_start=None,
        range_end=None,
        verbose=False,
    ):
        """
        Creates pandas dataframe from query of given inputs. Column order maybe different
            Args:
                org: email address for user authentication
                bucket: name of the bucket
                measurement: name of the measurement, if you used the influxdb_write_tester.ipynb, the default as sensors
                time_col: name of the column used as timestamps
                range_start: starting range for query, default set to last 10 years
                range_end: ending range for query, default to current datetime
        Returns:
            A pandas dataframe containing time and field entries from influxdb query
        """
        # default arguments if no input was given
        if str(range_start).lower() == "none":
            range_start = "-10y"
        if str(range_end).lower() == "none":
            range_end = "now()"
        # custom query in flux
        query_api = self.client.query_api()

        query = f"""from(bucket: "{bucket}")
        |> range(start: {range_start}, stop: {range_end}) 
        |> filter(fn: (r) => r._measurement == "{measurement}") 
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        """
        data = query_api.query_data_frame(query=query, org=org)

        if len(data) == 0:
            raise EmptyDataError

        # convert back the time column name
        if "_time" in data.columns:
            data.rename(columns={"_time": time_col}, inplace=True)
        # remove meta data columns and timezone utc from datetime
        if not verbose:
            meta_columns = ["result", "table", "_start", "_stop", "_measurement"]
            data = data.drop(meta_columns, axis=1)
            # if you need to preserve timezone utc on timestamps, remove the line below
            data[time_col] = pd.to_datetime(data[time_col]).dt.tz_convert(None)
        return data

    def write_data(self, file_dir, time_col, bucket, measurement, rows_to_write=None):
        """
        Write data to specific bucket and measurement
            Args:
                file_dir: file directory of the csv file to write
                time_col: name of the time column
                bucket: name of the bucket
                measurement: name of the measurement, if you used the influxdb_write_tester.ipynb, the default as sensors
                rows_to_write: integer of num of rows to write, mostly for testing purpose
        """
        df = pd.read_csv(file_dir, sep=",", engine="python", index_col=False)
        df[time_col] = pd.to_datetime(df[time_col])
        if rows_to_write != None:
            df = df.head(rows_to_write)
        self.client.write_api(write_options=SYNCHRONOUS).write(
            bucket=bucket,
            record=df,
            data_frame_measurement_name=measurement,
            data_frame_timestamp_column=time_col,
        )


def main():
    args = parse_parameters()
    # grab secret
    if args.token.lower() == "secret":
        args.token = os.environ.get("INFLUXDB_TOKEN")
    if args.url.lower() == "secret":
        args.url = os.environ.get("INFLUXDB_URL")
    if args.org.lower() == "secret":
        args.org = os.environ.get("INFLUXDB_ORG")
    if args.bucket.lower() == "secret":
        args.bucket = os.environ.get("INFLUXDB_BUCKET")
    # check if directory is valid
    if str(args.cnvrg_dataset).lower() != "none":
        if not os.path.exists(args.local_dir):
            raise IncorrectDirectoryError()
    # return pandas dataframe from custom query
    influxdb = Influxdb(url=args.url, token=args.token, org=args.org)
    df = influxdb.get_data(
        args.org,
        args.bucket,
        args.measurement,
        args.time_col,
        args.range_start,
        args.range_end,
        verbose=False,
    )
    df.to_csv(args.local_dir + "/" + args.file_name, index=False)

    # Store csv as cnvrg dataset
    if str(args.cnvrg_dataset).lower() != "none":
        # cnvrgv2 dependencies have version mismatch issue with influxdb_client so
        # the library is imported here
        from cnvrgv2 import Cnvrg

        cnvrg = Cnvrg()
        ds = cnvrg.datasets.get(args.cnvrg_dataset)
        try:
            ds.reload()
        except:
            print("The provided data was not found")
            print(f"Creating a new data named {args.cnvrg_dataset}")
            ds = cnvrg.datasets.create(name=args.cnvrg_dataset)
        print("Uploading files to Cnvrg")
        os.chdir(args.local_dir)
        ds.put_files(paths=[args.file_name])


if __name__ == "__main__":
    main()
