import unittest

from mltunex.data.ingestion import Data_Ingestion
from mltunex.trainer.trainer import ModelTrainer

data_path = "../test_assets/data/california_housing_test_1.csv"
target_column = "median_house_value"

ingestor = Data_Ingestion()
X_train, X_test, y_train, y_test = ingestor.ingest_data(data_path, target_column)


class TestModelTrainer(unittest.TestCase):
    def setUp(self):
        self.trainer = ModelTrainer(X_train, X_test, y_train, y_test, models_library="sklearn", cross_validation_strategy="kfold", task_type="regression")

    def test_load_models(self):
        models = self.trainer._load_models()
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)

    def test_train_model(self):
        model = self.trainer._load_models()[0]
        trained_model = self.trainer._train_model(model, X_train, y_train)
        self.assertIsInstance(trained_model, tuple)  # Assuming the model is returned as a tuple (name, estimator)

    def test_evaluate_model(self):
        model = self.trainer._load_models()[0]
        trained_model = self.trainer._train_model(model, X_train, y_train)
        evaluation_results = self.trainer._evaluate_model(trained_model, X_test, y_test)
        self.assertIsInstance(evaluation_results, dict)  # Assuming evaluation results are returned as a dictionary

    def test_run(self):
        results, evaluation_results = self.trainer._run(X_train, y_train, X_test, y_test)
        self.assertIsInstance(results, dict)
        self.assertIsInstance(evaluation_results, list)
        print("Evaluation Results: ", evaluation_results)

if __name__ == "__main__":
    unittest.main()
