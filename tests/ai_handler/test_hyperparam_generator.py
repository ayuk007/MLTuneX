import unittest
import pandas as pd

from mltunex.data.ingestion import Data_Ingestion
from mltunex.ai_handler.metadata_profiler import MetaDataProfiler
from mltunex.trainer.trainer import ModelTrainer
from mltunex.utils.model_utils import ModelUtils
from mltunex.model_registry.model_registry import Model_Registry
from mltunex.ai_handler.llm_manager.openai_handler import OpenAIHyperparamGenerator



data_path = "../test_assets/data/california_housing_test_1.csv"
target_column = "median_house_value"

ingestor = Data_Ingestion()
X_train, X_test, y_train, y_test = ingestor.ingest_data(data_path, target_column)

trainer = ModelTrainer(X_train, X_test, y_train, y_test, models_library="sklearn", cross_validation_strategy="kfold", task_type="regression")
results, evaluation_results = trainer._run(X_train, y_train, X_test, y_test)
evaluation_df = ModelUtils.save_results(evaluation_results = evaluation_results)
top_models = ModelUtils.get_topK_models(results_csv = evaluation_df, task_type = "regression", k = 3)


profiler = MetaDataProfiler((X_train, X_test, y_train, y_test), target_column = "median_house_value", task_type = "regression")
metadata = profiler.extract()

class TestHyperparamGenerator(unittest.TestCase):
    def setUp(self):
        self.model_registry = Model_Registry.get_model_registry("sklearn")
        self.model_hyperparameter_schema = self.model_registry.get_all_hyperparameters(top_models = top_models["Model"].tolist(), models = results)
        self.hyperparam_generator = OpenAIHyperparamGenerator()
        assert self.hyperparam_generator is not None, "OpenAIHyperparamGenerator instance is not created."

    def test_generate_response(self):
        self.response = self.hyperparam_generator.generate_response(
            data_profile = metadata,
            top_models = top_models.to_json(),
            model_hyperparameter_schema = str(self.model_hyperparameter_schema),
        )
        self.assertIsInstance(self.response, list)

    
if __name__ == "__main__":
    unittest.main()