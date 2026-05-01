"""
Metrics Registry module for MLTuneX.

This module provides a centralized registry for evaluation metrics used in
machine learning model assessment. It supports both classification and
regression metrics, with built-in integration of scikit-learn metrics.

Examples:
    >>> metrics = MetricsRegistry.get_metrics("classification")
    >>> accuracy = metrics["Accuracy"](y_true, y_pred)
"""

import numpy as np
from typing import Callable, Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, roc_auc_score, precision_recall_curve,
    mean_squared_error, mean_absolute_error, r2_score, auc
)


class MetricsRegistry:
    """
    Registry for machine learning evaluation metrics.

    This class provides a centralized collection of evaluation metrics for both
    classification and regression tasks. It includes standard metrics from
    scikit-learn as well as custom implementations.

    Attributes
    ----------
    CLASSIFICATION_METRICS : Dict[str, Callable]
        Dictionary of classification metrics mapping names to metric functions.
    REGRESSION_METRICS : Dict[str, Callable]
        Dictionary of regression metrics mapping names to metric functions.

    Methods
    -------
    calculate_aucpr(y_true, y_pred) : float
        Calculate Area Under Precision-Recall Curve.
    get_metrics(task_type: str) -> Dict[str, Callable]
        Get metrics dictionary for specified task type.
    """

    @staticmethod
    def calculate_aucpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Area Under Precision-Recall Curve.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth (correct) labels.
        y_pred : np.ndarray
            Predicted probabilities or scores.

        Returns
        -------
        float
            Area under the precision-recall curve.

        Notes
        -----
        Uses scikit-learn's precision_recall_curve and auc functions.
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        return auc(recall, precision)

    # Dictionary of classification metrics with their corresponding functions
    CLASSIFICATION_METRICS = {
        "Accuracy": accuracy_score,
        "f1": f1_score,
        "LogLoss": log_loss,
        "AUC": roc_auc_score,
        "AUCPR": calculate_aucpr,  # Custom AUCPR implementation
    }

    # Dictionary of regression metrics with their corresponding functions
    REGRESSION_METRICS = {
        "MSE": mean_squared_error,
        "RMSE": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error,
        "R2": r2_score,
    }

    @staticmethod
    def get_metrics(task_type: str) -> Dict[str, Callable]:
        """
        Get appropriate metrics dictionary based on the task type.

        Parameters
        ----------
        task_type : str
            Type of machine learning task.
            Must be one of: ["classification", "regression"]

        Returns
        -------
        Dict[str, Callable]
            Dictionary mapping metric names to their computation functions.

        Raises
        ------
        ValueError
            If task_type is not "classification" or "regression".

        Examples
        --------
        >>> metrics = MetricsRegistry.get_metrics("classification")
        >>> accuracy = metrics["Accuracy"](y_true, y_pred)
        """
        if task_type == "classification":
            return MetricsRegistry.CLASSIFICATION_METRICS
        elif task_type == "regression":
            return MetricsRegistry.REGRESSION_METRICS
        else:
            raise ValueError(
                f"Unsupported task type: {task_type}. "
                'Must be either "classification" or "regression".'
            )