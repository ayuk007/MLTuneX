import unittest
import pandas as pd

from mltunex.data.ingestion import Data_Ingestion
from mltunex.ai_handler.metadata_profiler import MetaDataProfiler
from mltunex.trainer.trainer import ModelTrainer
from mltunex.utils.model_utils import ModelUtils
from mltunex.model_registry.model_registry import Model_Registry
from mltunex.ai_handler.hyperparam_generator import OpenAIHyperparamGenerator
from mltunex.hyperparam_tuner.hyperparameter_tuner import HyperparameterTunerFactory


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

model_registry = Model_Registry.get_model_registry("sklearn")
model_hyperparameter_schema = model_registry.get_all_hyperparameters(top_models = top_models["Model"].tolist(), models = results)
hyperparam_generator = OpenAIHyperparamGenerator()
response = hyperparam_generator.generate_response(
    data_profile = metadata,
    top_models = top_models.to_json(),
    model_hyperparameter_schema = str(model_hyperparameter_schema),
)

class TestHyperparameterTuner(unittest.TestCase):
    def setUp(self):
        self.tuner = HyperparameterTunerFactory.create_tuner(method = "optuna", task_type = "regression", training_results = results)
        self.assertIsNotNone(self.tuner, "OptunaHyperparameterTuner instance is not created.")

    def test_run_tuning(self):
        best_model, best_params = self.tuner.run_optuna(
            model_search_spaces = response,
            x_train = X_train,
            y_train = y_train,
        )

        self.assertIsNotNone(best_model, "Best model is None.")
        self.assertIsNotNone(best_params, "Best parameters are None.")
        print(f"Best model: {best_model}")
        print(f"Best parameters: {best_params}")


if __name__ == "__main__":
    unittest.main()