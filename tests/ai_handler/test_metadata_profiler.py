import unittest
import pandas as pd

from mltunex.data.ingestion import Data_Ingestion
from mltunex.ai_handler.metadata_profiler import MetaDataProfiler


data_path = "../test_assets/data/california_housing_test_1.csv"
target_column = "median_house_value"

ingestor = Data_Ingestion()
x_train, x_test, y_train, y_test = ingestor.ingest_data(data_path, target_column)

class TestMetaDataProfiler(unittest.TestCase):
    def setUp(self):
        self.profiler = MetaDataProfiler((x_train, x_test, y_train, y_test), target_column = "median_house_value", task_type = "regression")
    
    def test_get_shape(self):
        shape = self.profiler.get_shape()
        self.assertIsInstance(shape, str)
        self.assertIn("num_rows", shape)
        self.assertIn("num_features", shape)
        print("Shape: ", shape)

    def test_find_missing_values(self):
        missing_values = self.profiler.find_missing_values()
        self.assertIsInstance(missing_values, pd.Series)
        print("Missing Values: ", missing_values.to_json())
    
    def test_get_data_stats(self):
        data_stats = self.profiler.get_data_stats()
        self.assertIsInstance(data_stats, str)
        print("Data Stats: ", data_stats)

    def test_feature_distribution_insights(self):
        insights = self.profiler.feature_distribution_insights()
        self.assertIsInstance(insights, str)
        print("Feature Distribution Insights: ", insights)

    def test_correlation_analysis(self):
        correlation = self.profiler.correlation_analysis()
        self.assertIsInstance(correlation, str)
        print("Correlation Analysis: ", correlation)
    
    def test_get_data_types(self):
        data_types = self.profiler.get_data_types()
        self.assertIsInstance(data_types, str)
        print("Data Types: ", data_types)
    
    def test_target_distribution(self):
        target_dist = self.profiler.target_distribution()
        self.assertIsInstance(target_dist, str)
        print("Target Distribution: ", target_dist)

    def test_get_skew_kurtosis(self):
        skew_kurtosis = self.profiler.get_skew_kurtosis()
        self.assertIsInstance(skew_kurtosis, str)
        print("Skewness and Kurtosis: ", skew_kurtosis)
    
    def test_get_features(self):
        features = self.profiler.get_features()
        self.assertIsInstance(features, str)
        print("Features: ", features)
    
    def test_extract(self):
        data_insights = self.profiler.extract()
        self.assertIsInstance(data_insights, str)
        print("Data Insights: ", data_insights)

    def test_merge_data_split(self):
        merged_data = self.profiler.merge_data_split((x_train, x_test, y_train, y_test))
        self.assertIsInstance(merged_data, pd.DataFrame)
        print("Merged Data: ", merged_data.head())

if __name__ == "__main__":
    unittest.main()