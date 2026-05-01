"""
Evaluator interface and interchangeable strategy implementations for MLTuneX.

Design
------
* Evaluator               — abstract interface (ISP-compliant, small contract)
* ClassificationEvaluator — concrete strategy for classification tasks
* RegressionEvaluator     — concrete strategy for regression tasks
* EvaluatorFactory        — creates the correct evaluator by task type

Strategies are fully interchangeable (LSP) and new ones can be registered
without touching existing code (OCP).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, auc, f1_score, log_loss,
    mean_absolute_error, mean_squared_error,
    precision_recall_curve, r2_score, roc_auc_score,
)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class Evaluator(ABC):
    """
    Abstract interface for model evaluators.

    Responsibilities
    ----------------
    Given a trained model and test data, compute task-specific metrics
    and return them in a standardised dictionary.
    """

    @abstractmethod
    def evaluate(
        self,
        model_name: str,
        model: Any,
        X_test: Any,
        y_test: Any,
    ) -> Dict[str, Optional[Dict[str, float]]]:
        """
        Evaluate *model* on *X_test* / *y_test*.

        Returns
        -------
        Dict[str, Optional[Dict[str, float]]]
            ``{model_name: {metric_name: score, ...}}`` or
            ``{model_name: None}`` on failure.
        """

    @abstractmethod
    def metrics(self) -> Dict[str, Callable]:
        """Return the metric functions this evaluator uses."""


# ---------------------------------------------------------------------------
# Metric registry helpers (kept local to this module)
# ---------------------------------------------------------------------------

def _aucpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return float(auc(recall, precision))


_CLASSIFICATION_METRICS: Dict[str, Callable] = {
    "Accuracy":  accuracy_score,
    "f1":        lambda yt, yp: f1_score(yt, yp, average="weighted", zero_division=0),
    "LogLoss":   log_loss,
    "AUC":       roc_auc_score,
    "AUCPR":     _aucpr,
}

_REGRESSION_METRICS: Dict[str, Callable] = {
    "MSE":  mean_squared_error,
    "RMSE": lambda yt, yp: float(np.sqrt(mean_squared_error(yt, yp))),
    "MAE":  mean_absolute_error,
    "R2":   r2_score,
}


# ---------------------------------------------------------------------------
# Concrete evaluators
# ---------------------------------------------------------------------------

class ClassificationEvaluator(Evaluator):
    """Evaluator for binary / multiclass classification models."""

    def metrics(self) -> Dict[str, Callable]:
        return dict(_CLASSIFICATION_METRICS)

    def evaluate(
        self,
        model_name: str,
        model: Any,
        X_test: Any,
        y_test: Any,
    ) -> Dict[str, Optional[Dict[str, float]]]:
        # print(f"Evaluating model: {model_name}")
        try:
            y_pred = model.predict(X_test)
            scores: Dict[str, float] = {}
            for metric_name, fn in self.metrics().items():
                try:
                    scores[metric_name] = float(fn(y_test, y_pred))
                except Exception:
                    scores[metric_name] = float("nan")
            return {model_name: scores}
        except Exception as exc:
            # print(f"Error evaluating model {model_name}: {exc}")
            return {model_name: None}


class RegressionEvaluator(Evaluator):
    """Evaluator for regression models."""

    def metrics(self) -> Dict[str, Callable]:
        return dict(_REGRESSION_METRICS)

    def evaluate(
        self,
        model_name: str,
        model: Any,
        X_test: Any,
        y_test: Any,
    ) -> Dict[str, Optional[Dict[str, float]]]:
        # print(f"Evaluating model: {model_name}")
        try:
            y_pred = model.predict(X_test)
            scores: Dict[str, float] = {}
            for metric_name, fn in self.metrics().items():
                try:
                    scores[metric_name] = float(fn(y_test, y_pred))
                except Exception:
                    scores[metric_name] = float("nan")
            return {model_name: scores}
        except Exception as exc:
            # print(f"Error evaluating model {model_name}: {exc}")
            return {model_name: None}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class EvaluatorFactory:
    """
    Factory for Evaluator strategies.

    Registered strategies
    ---------------------
    "classification" → ClassificationEvaluator
    "regression"     → RegressionEvaluator
    """

    _registry: dict[str, type[Evaluator]] = {
        "classification": ClassificationEvaluator,
        "regression":     RegressionEvaluator,
    }

    @classmethod
    def register(cls, task_type: str, evaluator_class: type[Evaluator]) -> None:
        """Register a custom evaluator for a new task type."""
        cls._registry[task_type.lower()] = evaluator_class

    @classmethod
    def create(cls, task_type: str) -> Evaluator:
        """
        Return an Evaluator for *task_type*.

        Raises
        ------
        ValueError
            If *task_type* is not registered.
        """
        key = task_type.lower()
        if key not in cls._registry:
            raise ValueError(
                f"Unsupported task type '{task_type}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[key]()
