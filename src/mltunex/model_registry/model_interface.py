"""
Model abstraction layer for MLTuneX.

Defines:
  - Model        : abstract interface that every concrete model must implement
  - ModelFactory : abstract factory; concrete sub-factories produce Model
                   instances for a specific backend (sklearn, xgboost, etc.)

Following SOLID principles:
  - Model provides a single, stable interface (SRP + LSP)
  - ModelFactory is open for extension: add a new backend factory without
    touching any existing code (OCP)
  - Consumers depend on Model / ModelFactory abstractions, never on sklearn
    or any concrete class (DIP)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Model interface
# ---------------------------------------------------------------------------

class Model(ABC):
    """
    Abstract interface for all machine-learning models.

    Every concrete model must support three operations:
    train, predict, and evaluate.  The rest of the system interacts
    exclusively through this interface.
    """

    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        params: Optional[Dict[str, Any]] = None,
    ) -> "Model":
        """
        Fit the model on training data.

        Parameters
        ----------
        X_train : pd.DataFrame
        y_train : pd.Series
        params : dict, optional
            Hyperparameters to apply before fitting.

        Returns
        -------
        Model
            self, for method chaining.
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Any:
        """
        Generate predictions for *X*.

        Returns
        -------
        array-like
            Predicted values or class labels.
        """

    @abstractmethod
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics on *X_test* / *y_test*.

        Returns
        -------
        Dict[str, float]
            Metric-name → score mapping.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for this model."""

    @property
    @abstractmethod
    def underlying(self) -> Any:
        """The raw estimator object (e.g., an sklearn BaseEstimator)."""


# ---------------------------------------------------------------------------
# ModelFactory interface
# ---------------------------------------------------------------------------

class ModelFactory(ABC):
    """
    Abstract factory for creating Model instances.

    Concrete factories (SklearnModelFactory, XGBoostModelFactory, …)
    implement this interface.  Callers request a named model without
    knowing which library backs it.
    """

    @abstractmethod
    def create(self, model_name: str) -> Model:
        """
        Instantiate and return a Model by name.

        Parameters
        ----------
        model_name : str
            Registry key identifying the desired model.

        Returns
        -------
        Model

        Raises
        ------
        ValueError
            If *model_name* is not known to this factory.
        """

    @abstractmethod
    def list_models(self) -> List[str]:
        """Return the names of all models this factory can create."""


# ---------------------------------------------------------------------------
# SklearnModel — wraps any sklearn-compatible estimator
# ---------------------------------------------------------------------------

class SklearnModel(Model):
    """
    Concrete Model wrapping a scikit-learn–compatible estimator class.

    Parameters
    ----------
    model_name : str
        Display name used throughout the framework.
    estimator_class : type
        The uninstantiated estimator class (e.g., RandomForestClassifier).
    task_type : str
        "classification" or "regression" — controls which metrics are used.
    """

    def __init__(
        self,
        model_name: str,
        estimator_class: type,
        task_type: str = "classification",
    ) -> None:
        self._name = model_name
        self._estimator_class = estimator_class
        self._task_type = task_type
        self._estimator: Any = None

    # ------------------------------------------------------------------
    # Model interface
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        params: Optional[Dict[str, Any]] = None,
    ) -> "SklearnModel":
        instance = self._estimator_class()
        if params:
            instance.set_params(**params)
        if hasattr(instance, "n_jobs"):
            instance.set_params(n_jobs=-1)
        instance.fit(X_train, y_train)
        self._estimator = instance
        return self

    def predict(self, X: pd.DataFrame) -> Any:
        if self._estimator is None:
            raise RuntimeError(f"Model '{self._name}' has not been trained yet.")
        return self._estimator.predict(X)

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, float]:
        from mltunex.evaluate.metrics_registry import MetricsRegistry

        y_pred = self.predict(X_test)
        metrics = MetricsRegistry.get_metrics(self._task_type)
        results: Dict[str, float] = {}
        for metric_name, metric_fn in metrics.items():
            try:
                results[metric_name] = float(metric_fn(y_test, y_pred))
            except Exception:
                results[metric_name] = float("nan")
        return results

    @property
    def name(self) -> str:
        return self._name

    @property
    def underlying(self) -> Any:
        return self._estimator
