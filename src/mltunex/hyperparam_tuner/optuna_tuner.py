"""
Optuna-based hyperparameter tuner implementation for MLTuneX.

This module provides a concrete implementation of hyperparameter tuning using
the Optuna optimization framework. It supports various parameter types and
implements cross-validation for robust optimization.

Examples:
    >>> tuner = OptunaHyperparameterTuner(training_results, "classification")
    >>> best_model, best_params = tuner.run_optuna(search_spaces, X_train, y_train)
"""

import optuna
import warnings
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import cross_val_score
from mltunex.hyperparam_tuner.base import BaseHyperparameterTuner
from mltunex.library_trainer.library_trainer import LibraryTrainer

# Suppress Optuna's user warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="optuna")


class OptunaHyperparameterTuner(BaseHyperparameterTuner):
    """
    Optuna-based implementation of hyperparameter tuning.

    This class uses Optuna to perform hyperparameter optimization with
    cross-validation. It supports various parameter types including
    integer, float, categorical, and boolean parameters.

    Parameters
    ----------
    training_results : List
        Previous training results for model initialization.
    task_type : str
        Type of ML task ('classification' or 'regression').

    Attributes
    ----------
    scoring_metric : str
        Metric used for optimization ('accuracy' for classification, 
        'r2' for regression).
    """

    def __init__(self, training_results: List, task_type: str, library: str = "sklearn"):
        """Initialize OptunaHyperparameterTuner with configuration."""
        self.training_results = training_results
        self.task_type = task_type
        self.model_trainer = LibraryTrainer.get_trainer(library = library)
        self.scoring_metric = "accuracy" if task_type == "classification" else "r2"
        
    def get_best_hyperparameters(self, best_params_dict: dict, 
                                model_name: str) -> Tuple[Dict[str, Any], str]:
        """
        Extract model-specific parameters from Optuna results.

        Parameters
        ----------
        best_params_dict : dict
            Dictionary containing all optimized parameters.
        model_name : str
            Name of the model to extract parameters for.

        Returns
        -------
        Tuple[Dict[str, Any], str]
            Tuple containing:
            - Dictionary of extracted parameters
            - Model name
        """
        extracted_params = {}
        prefix = f"{model_name}_"
        # Extract parameters for specific model
        for key, value in best_params_dict.items():
            if key.startswith(prefix):
                param_name = key.replace(prefix, "")
                extracted_params[param_name] = value
        return extracted_params, model_name
    
    @staticmethod
    def suggest_param(trial: optuna.Trial, param_name: str, 
                     param_def: Dict[str, Any]) -> Any:
        """
        Suggest parameter value based on definition.

        Parameters
        ----------
        trial : optuna.Trial
            Current optimization trial.
        param_name : str
            Name of the parameter.
        param_def : Dict[str, Any]
            Parameter definition dictionary.

        Returns
        -------
        Any
            Suggested parameter value.

        Raises
        ------
        ValueError
            If parameter type is not supported.
        """
        type_ = param_def["type"]

        if type_ == "int":
            if "values" in param_def:
                return trial.suggest_categorical(param_name, param_def["values"])
            return trial.suggest_int(
                param_name, 
                param_def["low"], 
                param_def["high"], 
                step=param_def.get("step", 1)
            )

        elif type_ == "float":
            return trial.suggest_float(
                param_name,
                param_def["low"],
                param_def["high"],
                step=param_def.get("step"),
                log=param_def.get("log", False)
            )

        elif type_ == "categorical":
            return trial.suggest_categorical(param_name, param_def["values"])

        elif type_ == "bool":
            return trial.suggest_categorical(param_name, [True, False])

        elif type_ == "fixed":
            return param_def["value"]

        else:
            raise ValueError(f"Unsupported parameter type: {type_}")
        
    def tune_hyperparameters(self, model_search_spaces: List[Dict], 
                           x_train: pd.DataFrame, 
                           y_train: pd.Series) -> Any:
        """
        Create Optuna objective function for hyperparameter tuning.

        Parameters
        ----------
        model_search_spaces : List[Dict]
            List of model configurations with their search spaces.
        x_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training target values.

        Returns
        -------
        Callable
            Objective function for Optuna optimization.
        """
        def objective(trial):
            # Select model configuration
            model_config = trial.suggest_categorical("model_config", model_search_spaces)
            model_name = model_config["model_name"]
            param_defs = model_config["suggested_hyperparameters"]

            # Suggest hyperparameters for the model
            params = {}
            for param_name, param_def in param_defs.items():
                full_name = f"{model_name}_{param_name}"
                params[param_name] = self.suggest_param(trial, full_name, param_def)

            try:
                model_tuple = self.training_results[0][model_name]
                model = self.model_trainer.train_model(model = model_tuple[-1], params = params, tune = True)
                # print(model)
                if model is None:
                    raise ValueError(f"Model {model_name} could not be trained with the given parameters: {params}")
                # Evaluate
                return cross_val_score(model, x_train, y_train, cv = 3, scoring=self.scoring_metric).mean()
            except Exception as e:
                print(f"Error during hyperparameter tuning for {model_name} with params {params}: {e}")
                # Return a very low score to avoid this trial
                return 0.0

        return objective
    
    def run_optuna(self, model_search_spaces: List[Dict], 
                  x_train: pd.DataFrame, y_train: pd.Series, 
                  n_trials: int = 25) -> List[Dict[str, Any]]:
        """
        Run Optuna optimization process.

        Parameters
        ----------
        model_search_spaces : List[Dict]
            List of model configurations with their search spaces.
        x_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training target values.
        n_trials : int, optional (default=25)
            Number of optimization trials to run.

        Returns
        -------
        Tuple[str, Dict[str, Any]]
            Tuple containing:
            - Name of the best model
            - Dictionary of best hyperparameters
        """
        # Create and run Optuna study
        study = optuna.create_study(direction="maximize")
        study.optimize(
            self.tune_hyperparameters(model_search_spaces, x_train, y_train), 
            n_trials=n_trials
        )

        # Extract best parameters
        best_params, best_model = self.get_best_hyperparameters(
            study.best_trial.params,
            study.best_trial.params["model_config"]["model_name"]
        )
        return best_model, best_params