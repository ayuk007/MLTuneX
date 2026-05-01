"""
Base Hyperparameter Tuner module for MLTuneX.

This module defines the abstract base class for hyperparameter tuning operations.
It provides a standardized interface for implementing different hyperparameter
optimization strategies and algorithms.

Examples:
    >>> tuner = ConcreteHyperparameterTuner()
    >>> best_params = tuner.tune_hyperparameters(model, X_train, y_train, X_test, y_test)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd


class BaseHyperparameterTuner(ABC):
    """
    Abstract base class for hyperparameter tuning implementations.

    This class defines the interface for hyperparameter optimization strategies.
    All concrete hyperparameter tuner implementations should inherit from this
    class and implement its abstract methods.

    Methods
    -------
    tune_hyperparameters(model, X_train, y_train, X_test, y_test)
        Optimize model hyperparameters using training and validation data.
    get_best_hyperparameters()
        Retrieve the best hyperparameters found during tuning.
    """

    @abstractmethod
    def tune_hyperparameters(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Tune model hyperparameters using specified data.

        Parameters
        ----------
        model : Any
            Machine learning model to optimize.
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training target values.
        X_test : pd.DataFrame
            Validation features.
        y_test : pd.Series
            Validation target values.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing optimized hyperparameters.

        Raises
        ------
        NotImplementedError
            If the concrete class does not implement this method.
        """
        pass

    @abstractmethod
    def get_best_hyperparameters(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve the best hyperparameters found during optimization.

        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary containing the best hyperparameters found,
            or None if no tuning has been performed.

        Raises
        ------
        NotImplementedError
            If the concrete class does not implement this method.
        """
        pass