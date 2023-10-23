import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import unittest
from influxdb_connector import Influxdb
import yaml
from yaml.loader import SafeLoader


YAML_ARG_TO_TEST = "test_arguments"
NUM_ROW_TO_TEST = 5


class TestInfluxdb(unittest.TestCase):
    def setUp(self) -> None:
        cfg_path = os.path.dirname(os.path.abspath(__file__))
        cfg_file = cfg_path + "/" + "test_config.yaml"
        self.test_cfg = {}
        with open(cfg_file) as c_info_file:
            self.test_cfg = yaml.load(c_info_file, Loader=SafeLoader)
        self.test_cfg = self.test_cfg[YAML_ARG_TO_TEST]
        self.influxdb = Influxdb(
            url=self.test_cfg["url"],
            token=self.test_cfg["token"],
            org=self.test_cfg["org"],
        )


@unittest.skip("skipping ")
class TestConnection(TestInfluxdb):
    def test_connection(self):
        """Checks if connection to influxdb client is established"""
        self.assertTrue(self.influxdb.ping())


@unittest.skip("skipping ")
class TestReturnType(TestInfluxdb):
    def test_return_type(self):
        """Checks if the get_data function returns pandas dataframe type"""
        self.assertIsInstance(
            self.influxdb.get_data(
                org=self.test_cfg["org"],
                bucket=self.test_cfg["bucket"],
                measurement=self.test_cfg["measurement"],
                time_col=self.test_cfg["time_col"],
                range_start=self.test_cfg["range_start"],
                range_end=self.test_cfg["range_end"],
                verbose=False,
            ),
            pd.DataFrame,
        )


# @unittest.skip('skipping ')
class TestWriteReadEquality(TestInfluxdb):
    def test_write_read_equality(self):
        """Checks if the written dataframe is same as the dataframe read, agnostic of column order"""
        csv_dir = (
            os.path.dirname(os.path.abspath(__file__))
            + "/../"
            + self.test_cfg["file_dir"]
        )
        self.influxdb.write_data(
            csv_dir,
            self.test_cfg["time_col"],
            self.test_cfg["bucket"],
            self.test_cfg["measurement"],
            NUM_ROW_TO_TEST,
        )
        # slice by number or row to test, drop index, and sort by column name
        written_data = pd.read_csv(csv_dir).head(NUM_ROW_TO_TEST).reset_index(drop=True)
        written_data = written_data.sort_index(axis=1)
        # by default influxdb attaches utc on datetime if it doesn't exist, so add it on time column
        written_data[self.test_cfg["time_col"]] = pd.to_datetime(
            written_data[self.test_cfg["time_col"]], utc=True
        ).dt.tz_convert(None)

        read_data = self.influxdb.get_data(
            self.test_cfg["org"],
            self.test_cfg["bucket"],
            self.test_cfg["measurement"],
            self.test_cfg["time_col"],
            self.test_cfg["range_start"],
            self.test_cfg["range_end"],
            verbose=False,
        ).tail(NUM_ROW_TO_TEST)

        # drop index, and sort by column name
        read_data = read_data.rename({"_time": self.test_cfg["time_col"]}, axis=1)
        read_data = read_data.reset_index(drop=True)
        read_data = read_data.sort_index(axis=1)

        self.assertTrue(pd.DataFrame.equals(written_data, read_data))


if __name__ == "__main__":
    unittest.main()
