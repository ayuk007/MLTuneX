"""
Base trainer module for MLTuneX.

This module defines the abstract base class for model trainers, providing
a common interface for training machine learning models. It ensures consistent
training workflow across different implementations.

Classes
-------
BaseTrainer : Abstract base class for model training pipeline
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Any


class BaseTrainer(ABC):
    """
    Abstract base class for model trainers.

    This class defines the interface for model training pipelines in MLTuneX.
    It provides abstract methods for loading models, training, evaluation,
    validation, and orchestrating the complete training workflow.

    Methods
    -------
    _load_models() -> List[Any]
        Load models for training.
    _train_model(X_train, y_train) -> List[Any]
        Train models with provided data.
    _evaluate_model(models, X_test, y_test) -> List[Any]
        Evaluate trained models.
    _validate_model(models, X_test, y_test) -> List[Any]
        Validate models' performance.
    _run(X_train, y_train, X_test, y_test) -> List[Any]
        Execute complete training pipeline.
    """

    @abstractmethod
    def _load_models(self) -> List[Any]:
        """
        Load models for training.

        Returns
        -------
        List[Any]
            List of model instances to be trained.

        Raises
        ------
        NotImplementedError
            If the concrete class does not implement this method.
        """
        pass

    @abstractmethod
    def _train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> List[Any]:
        """
        Train models with provided data.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training target values.

        Returns
        -------
        List[Any]
            List of trained model instances.

        Raises
        ------
        NotImplementedError
            If the concrete class does not implement this method.
        """
        pass

    @abstractmethod
    def _evaluate_model(self, models: List[Any], X_test: pd.DataFrame, y_test: pd.Series) -> List[Any]:
        """
        Evaluate trained models using test data.

        Parameters
        ----------
        models : List[Any]
            List of trained model instances.
        X_test : pd.DataFrame
            Test features.
        y_test : pd.Series
            Test target values.

        Returns
        -------
        List[Any]
            List of evaluation results for each model.

        Raises
        ------
        NotImplementedError
            If the concrete class does not implement this method.
        """
        pass

    @abstractmethod
    def _validate_model(self, models: List[Any], X_test: pd.DataFrame, y_test: pd.Series) -> List[Any]:
        """
        Validate models' performance using test data.

        Parameters
        ----------
        models : List[Any]
            List of trained model instances.
        X_test : pd.DataFrame
            Test features.
        y_test : pd.Series
            Test target values.

        Returns
        -------
        List[Any]
            List of validation results for each model.

        Raises
        ------
        NotImplementedError
            If the concrete class does not implement this method.
        """
        pass

    @abstractmethod
    def _run(self, X_train: pd.DataFrame, y_train: pd.Series, 
             X_test: pd.DataFrame, y_test: pd.Series) -> List[Any]:
        """
        Execute complete training pipeline.

        This method orchestrates the entire training workflow including
        model loading, training, evaluation, and validation.

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
        List[Any]
            List containing results from the complete training pipeline.

        Raises
        ------
        NotImplementedError
            If the concrete class does not implement this method.
        """
        pass