"""
Hyperparameter Tuner Factory module for MLTuneX.

This module provides a factory pattern implementation for creating hyperparameter
tuning instances. It supports different optimization frameworks and manages their
instantiation based on method specification.

Examples:
    >>> factory = HyperparameterTunerFactory()
    >>> tuner = factory.create_tuner("optuna", "classification", training_results)
    >>> best_params = tuner.tune_hyperparameters(model, X_train, y_train)
"""

from typing import List, Optional, Union
from mltunex.hyperparam_tuner.base import BaseHyperparameterTuner
from mltunex.hyperparam_tuner.optuna_tuner import OptunaHyperparameterTuner


class HyperparameterTunerFactory:
    """
    Factory class to create hyperparameter tuners based on the specified method.

    This class implements the factory pattern to create appropriate hyperparameter
    tuning instances based on the specified optimization framework. Currently
    supports Optuna with extensibility for other frameworks.

    Methods
    -------
    create_tuner(method, task_type, training_results)
        Creates and returns appropriate hyperparameter tuner instance.
    """

    @staticmethod
    def create_tuner(
        method: str, 
        task_type: str, 
        training_results: Optional[List] = None
    ) -> BaseHyperparameterTuner:
        """
        Create a hyperparameter tuner based on the specified method.

        This method instantiates the appropriate hyperparameter tuning
        implementation based on the specified method. It handles configuration
        and initialization of the tuner with provided parameters.

        Parameters
        ----------
        method : str
            The method for hyperparameter tuning.
            Currently supported: ["optuna"]
        task_type : str
            Type of machine learning task.
            Must be one of: ["classification", "regression"]
        training_results : Optional[List], default=None
            Training results for the tuner, if available.

        Returns
        -------
        BaseHyperparameterTuner
            An instance of the specified hyperparameter tuner.

        Raises
        ------
        ValueError
            If the specified tuning method is not supported.

        Examples
        --------
        >>> tuner = HyperparameterTunerFactory.create_tuner(
        ...     "optuna", "classification", results
        ... )
        """
        # Create appropriate tuner based on method
        if method == "optuna":
            return OptunaHyperparameterTuner(training_results, task_type)
        else:
            raise ValueError(
                f"Unsupported tuning method: {method}. "
                "Currently supported methods: ['optuna']"
            )