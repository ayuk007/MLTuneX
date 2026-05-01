import unittest

from mltunex.data.ingestion import Data_Ingestion
from mltunex.model_registry.model_registry import Model_Registry
from mltunex.library_trainer.library_trainer import LibraryTrainer
from mltunex.library_trainer.base import BaseLibraryTrainer

data_path = "tests/test_assets/data/california_housing_test_1.csv"
target_column = "median_house_value"

ingestor = Data_Ingestion()
X_train, X_test, y_train, y_test = ingestor.ingest_data(data_path, target_column)
registry = Model_Registry.get_model_registry("sklearn")
models = registry.get_regression_models()

class TestSklearnTrainer(unittest.TestCase):
    def test_get_trainer(self):
        trainer = LibraryTrainer.get_trainer("sklearn")
        self.assertIsInstance(trainer, BaseLibraryTrainer)

    def test_train_model(self):
        trainer = LibraryTrainer.get_trainer("sklearn")
        model = models[0]
        trained_model = trainer.train_model(model[-1], X_train, y_train, "regression")
        self.assertIsNotNone(trained_model)
        self.assertTrue(hasattr(trained_model, "predict"))
    
if __name__ == "__main__":
    unittest.main()