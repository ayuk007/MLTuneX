# tests/data/test_loader.py

import unittest
import pandas as pd
import os
# import sys
# sys.path.append('src')



from mltunex.data.loader import (
    DataFrame_Loader,
    CSVLoader,
    Data_Loader_Factory,
    Data_Loader
)

class TestDataFrameLoader(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "target": [0, 1, 0]
        })
        self.loader = DataFrame_Loader(self.df)

    def test_load_data_from_dataframe(self):
        X, y = self.loader.load_data("target")
        self.assertEqual(X.shape, (3, 2))
        self.assertEqual(y.tolist(), [0, 1, 0])


class TestCSVLoader(unittest.TestCase):
    def setUp(self):
        self.csv_path = os.path.join(os.path.dirname(__file__), '..', 'test_assets', 'data', 'california_housing_test_1.csv')
        self.loader = CSVLoader(self.csv_path)

    def test_load_data_from_csv(self):
        X, y = self.loader.load_data("median_house_value")
        self.assertEqual(X.shape, (3000, 8))


class TestDataLoaderFactory(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "feature1": [1],
            "feature2": [2],
            "target": [1]
        })
        self.csv_path = os.path.join(os.path.dirname(__file__), '..', 'test_assets', 'data', 'california_housing_test_1.csv')

    def test_get_loader_from_dataframe(self):
        loader = Data_Loader_Factory.get_data_loader(self.df)
        self.assertIsInstance(loader, DataFrame_Loader)

    def test_get_loader_from_csv_path(self):
        loader = Data_Loader_Factory.get_data_loader(self.csv_path)
        self.assertIsInstance(loader, CSVLoader)

    def test_unsupported_data_source(self):
        with self.assertRaises(ValueError):
            Data_Loader_Factory.get_data_loader(123)


class TestUnifiedDataLoader(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "feature1": [1, 2],
            "feature2": [3, 4],
            "target": [0, 1]
        })
        self.csv_path = os.path.join(os.path.dirname(__file__), '..', 'test_assets', 'data', 'california_housing_test_1.csv')
        self.loader = Data_Loader()

    def test_load_data_from_df(self):
        X, y = self.loader.load_data(self.df, target_column="target")
        self.assertEqual(X.shape, (2, 2))
        self.assertEqual(len(y), 2)

    def test_load_data_from_csv(self):
        X, y = self.loader.load_data(self.csv_path, target_column="median_house_value")
        self.assertEqual(X.shape, (3000, 8))


if __name__ == "__main__":
    unittest.main()
