"""
Model trainer implementation for MLTuneX.

This module provides concrete implementation of the model training pipeline,
handling model loading, training, evaluation, and orchestration of the
complete workflow. It supports different machine learning libraries and
training strategies.

Examples:
    >>> trainer = ModelTrainer(X_train, X_test, y_train, y_test, 
                             "sklearn", "kfold", "classification")
    >>> results, eval_results = trainer._run(X_train, y_train, X_test, y_test)
"""

import pandas as pd
from typing import Literal, Tuple, Dict, List, Any
from mltunex.trainer.base import BaseTrainer
from mltunex.training_strategy.iteration_strategy import IterationStrategy
from mltunex.model_registry.model_registry import Model_Registry
from mltunex.evaluate.evaluate_model import EvaluateModel
from mltunex.library_trainer.library_trainer import LibraryTrainer


class ModelTrainer(BaseTrainer):
    """
    Concrete implementation of model training pipeline.

    This class implements the complete model training workflow including
    model loading, training, evaluation, and validation. It supports
    different machine learning libraries and training strategies.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Testing features.
    y_train : pd.Series
        Training target values.
    y_test : pd.Series
        Testing target values.
    models_library : str
        Name of the machine learning library to use (e.g., "sklearn").
    cross_validation_strategy : str
        Type of cross-validation strategy to use.
    task_type : Literal["classification", "regression"]
        Type of machine learning task.
    train_parallelization : bool, optional (default=False)
        Whether to use parallel training strategy.

    Attributes
    ----------
    model_registry : BaseModelRegistry
        Registry for accessing machine learning models.
    training_strategy : Callable
        Strategy for model training iteration.
    library_trainer : BaseLibraryTrainer
        Trainer for specific machine learning library.
    evaluator : EvaluateModel
        Model evaluation handler.
    """

    def __init__(self, X_train: pd.DataFrame,
                 X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, 
                 models_library: str, cross_validation_strategy: str,
                 task_type: Literal["classification", "regression"], 
                 train_parallelization: bool = False):
        """Initialize ModelTrainer with data and configuration."""
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.task_type = task_type
        
        # Initialize components
        self.model_registry = Model_Registry.get_model_registry(models_library)
        self.models = self.model_registry.get_models(task_type=task_type)
        
        # Set training strategy (parallel or sequential)
        self.training_strategy = (
            ValueError("Model parallelization strategy not ready yet!") 
            if train_parallelization 
            else IterationStrategy.get_iteration_strategy(library=models_library)
        )
        
        # Initialize trainer and evaluator
        self.library_trainer = LibraryTrainer.get_trainer(library=models_library)
        self.evaluator = EvaluateModel(task_type=task_type)
        self.evaluate = None
        self.validate = None

    def _load_models(self) -> List[Tuple[str, Any]]:
        """
        Load models from the registry.

        Returns
        -------
        List[Tuple[str, Any]]
            List of tuples containing (model_name, model_instance).
        """
        return self.models

    def _train_model(self, model: Tuple[str, Any], 
                    X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[str, Any]:
        """
        Train a single model with the provided data.

        Parameters
        ----------
        model : Tuple[str, Any]
            Tuple of (model_name, model_instance).
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training target values.

        Returns
        -------
        Tuple[str, Any]
            Tuple of (model_name, trained_model).
        """
        model_name, estimator = model
        print("Model: ", model_name)
        # Configure task type for training
        task_type = "classification" if self.task_type == "classifier" else "regressor"
        estimator = self.library_trainer.train_model(
            model=estimator, X_train=X_train, 
            y_train=y_train, task_type=task_type
        )
        return model_name, estimator

    def _evaluate_model(self, model: Tuple[str, Any], 
                       X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate a trained model using test data.

        Parameters
        ----------
        model : Tuple[str, Any]
            Tuple of (model_name, trained_model).
        X_test : pd.DataFrame
            Test features.
        y_test : pd.Series
            Test target values.

        Returns
        -------
        Dict
            Dictionary containing evaluation metrics.
        """
        return self.evaluator.evaluate(model[0], model[1], X_test, y_test)

    def _validate_model(self, models: List[Tuple[str, Any]], 
                       X_test: pd.DataFrame, y_test: pd.Series):
        """
        Validate trained models (placeholder for future implementation).

        Parameters
        ----------
        models : List[Tuple[str, Any]]
            List of (model_name, trained_model) tuples.
        X_test : pd.DataFrame
            Test features.
        y_test : pd.Series
            Test target values.
        """
        pass

    def _run(self, X_train: pd.DataFrame, y_train: pd.Series, 
             X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Dict, List]:
        """
        Execute complete training pipeline.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training target values.
        X_test : pd.DataFrame
            Test features.
        y_test : pd.Series
            Test target values.

        Returns
        -------
        Tuple[Dict, List]
            Tuple containing:
            - Dictionary of trained models
            - List of evaluation results
        """
        # Load and train models
        models = self._load_models()
        results, evaluation_results = self.training_strategy(
            self._train_model, self._evaluate_model, 
            models, X_train, y_train
        )
        return results, evaluation_results
