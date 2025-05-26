import unittest
from mltunex.main import MLTuneX

class TestMLTuneX(unittest.TestCase):
    def setUp(self):
        self.mltunex = MLTuneX(
            data = "../test_assets/data/california_housing_test_1.csv",
            target_column = "median_house_value",
            task_type="regression",
            models_library="sklearn"
        )
    
    def test_run(self):
        self.mltunex.run(
            result_csv_path = "../test_assets/results/california_housing_results.csv",
            model_dir_path = "../test_assets/models",
        )

if __name__ == "__main__":
    unittest.main()