import os
import joblib
import uuid
import pandas as pd
from typing import Dict, List

class ModelUtils:
    def __init__(self):
        pass
    
    @staticmethod
    def load_model(model_path: str) -> object:
        """Load a model from the specified path."""
        # Check if the model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        # Load the model using joblib or pickle
        model = joblib.load(model_path)
        return model

    @staticmethod
    def load_models(model_dir_path: str) -> Dict[str, object]:
        """Load multiple models from the specified directory."""
        models = {}
        for file_name in os.listdir(model_dir_path):
            if file_name.endswith(".joblib"):
                model_name = file_name[:-7]  # Remove the '.joblib' extension
                model_path = os.path.join(model_dir_path, file_name)
                models[model_name] = ModelUtils.load_model(model_path)
        return models
    
    @staticmethod
    def save_model(model: object, model_path: str) -> None:
        """Save a single model to the specified path."""
        # Check if the directory exists, if not create it
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # Save the model using joblib or pickle
        try:
            joblib.dump(model, model_path)
            print(f"Model saved at {model_path}")
        except Exception as e:
            print(f"Error saving model {model_path}: {e}")

    @staticmethod
    def save_models(models: Dict[str, object], model_dir_path: str) -> None:
        """Save multiple models to the specified directory."""
        # Check if the directory exists, if not create it
        if not os.path.exists(model_dir_path):
            os.makedirs(model_dir_path)
        for model_name, model in models.items():
            # Save the model using joblib or pickle
            ModelUtils.save_model(model, os.path.join(model_dir_path, f"{model_name}.joblib"))

    @staticmethod
    def load_results(results_path: str) -> pd.DataFrame:
        """Load results from the specified path."""
        # Check if the results file exists
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"Results file not found at {results_path}")
        
        # Load the results using pandas
        results = pd.read_csv(results_path)
        return results

    @staticmethod    
    def save_results(evaluation_results: Dict[str, Dict], evaluation_results_path: str = None, save: bool = True) -> pd.DataFrame:
        rows = []
        for result in evaluation_results:
            # Assuming each item is a dictionary with model names as keys and metrics as values
            for model, metrics in result.items():
                row = {'Model': model, **metrics}
                rows.append(row)

        results_ = pd.DataFrame(rows)
        if save and evaluation_results_path:
            # Check if the directory exists, if not create it
            os.makedirs(evaluation_results_path, exist_ok=True)
            # Save the results to a CSV file
            evaluation_results_path = os.path.join(evaluation_results_path, f"results_{uuid.uuid4()}.csv")
            results_.to_csv(evaluation_results_path, index=False)
            print(f"Results saved to {evaluation_results_path}")
        
        return results_
    
    @staticmethod
    def get_topK_models(results_csv: pd.DataFrame, task_type : str, k: int = 5) -> pd.DataFrame:
        """Get the top K models based on accuracy."""
        # Assuming 'Accuracy' is the metric to sort by
        top_models = results_csv.nlargest(k, 'Accuracy') if task_type == "classification" else results_csv.nlargest(k, 'R2')
        return top_models