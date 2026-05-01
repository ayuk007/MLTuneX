import unittest

from mltunex.data.ingestion import Data_Ingestion
from mltunex.model_registry.model_registry import Model_Registry
from mltunex.library_trainer.library_trainer import LibraryTrainer
from mltunex.library_trainer.base import BaseLibraryTrainer
from mltunex.evaluate.evaluate_model import EvaluateModel
from mltunex.evaluate.metrics_registry import MetricsRegistry


data_path = "../test_assets/data/california_housing_test_1.csv"
target_column = "median_house_value"

ingestor = Data_Ingestion()
X_train, X_test, y_train, y_test = ingestor.ingest_data(data_path, target_column)
registry = Model_Registry.get_model_registry("sklearn")
models = registry.get_regression_models()
trainer = LibraryTrainer.get_trainer("sklearn")
model = models[0]
trained_model = trainer.train_model(model[-1], X_train, y_train, "regression")

class TestEvaluateModel(unittest.TestCase):
    def test_evaluate_model(self):
        evaluator = EvaluateModel("regression")
        evaluation_results = evaluator.evaluate(model, trained_model, X_test, y_test)
        self.assertIsInstance(evaluation_results, dict)
        self.assertIn(model, evaluation_results)
        self.assertIsInstance(evaluation_results[model], dict)
        for metric_name in MetricsRegistry.REGRESSION_METRICS.keys():
            self.assertIn(metric_name, evaluation_results[model])
            print(f"{metric_name}: {evaluation_results[model][metric_name]}")


if __name__ == "__main__":
    unittest.main()