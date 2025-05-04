"""
Model evaluation module for MLTuneX.

This module provides implementation for model evaluation using various metrics
from the MetricsRegistry. It supports both classification and regression tasks,
with robust error handling and reporting capabilities.

Examples:
    >>> evaluator = EvaluateModel("classification")
    >>> results = evaluator.evaluate("RandomForest", model, X_test, y_test)
    >>> print(results["RandomForest"]["Accuracy"])
"""

from typing import Dict, Any, Optional
from mltunex.evaluate.metrics_registry import MetricsRegistry
from mltunex.evaluate.base import BaseEvaluator


class EvaluateModel(BaseEvaluator):
    """
    Model evaluation implementation with metric computation.

    This class handles the evaluation of machine learning models using
    appropriate metrics based on the task type. It provides robust error
    handling and returns structured evaluation results.

    Attributes
    ----------
    metrics : Dict[str, Callable]
        Dictionary of evaluation metrics from MetricsRegistry.

    Parameters
    ----------
    task_type : str
        Type of machine learning task ('classification' or 'regression').
    """

    def __init__(self, task_type: str):
        """
        Initialize the model evaluator.

        Parameters
        ----------
        task_type : str
            Type of machine learning task ('classification' or 'regression').

        Raises
        ------
        ValueError
            If task_type is not supported by MetricsRegistry.
        """
        # Get appropriate metrics for the specified task type
        self.metrics = MetricsRegistry.get_metrics(task_type)

    def evaluate(self, model_name: str, model: Any, X_test: Any, y_test: Any) -> Dict[str, Optional[Dict[str, float]]]:
        """
        Evaluate a trained model using registered metrics.

        This method applies all registered metrics for the task type to the
        model's predictions and returns the results in a structured format.

        Parameters
        ----------
        model_name : str
            Name of the model being evaluated.
        model : Any
            Trained model instance with predict method.
        X_test : array-like
            Test features.
        y_test : array-like
            True test labels.

        Returns
        -------
        Dict[str, Optional[Dict[str, float]]]
            Nested dictionary containing:
            - Outer key: model name
            - Inner key: metric names
            - Values: computed metric scores
            Returns {model_name: None} if evaluation fails.

        Examples
        --------
        >>> results = evaluator.evaluate("RandomForest", rf_model, X_test, y_test)
        >>> accuracy = results["RandomForest"]["Accuracy"]
        """
        # Log evaluation start
        print(f"Evaluating model: {model_name}")

        try:
            # Generate predictions using the model
            y_pred = model.predict(X_test)

            # Compute all metrics and return results
            return {
                model_name: {
                    metric_name: metric(y_test, y_pred) 
                    for metric_name, metric in self.metrics.items()
                }
            }

        except Exception as e:
            # Log error and return None for failed evaluation
            print(f"Error evaluating model {model_name}: {e}")
            return {model_name: None}