"""
ModelSelector interface and concrete selection strategies for MLTuneX.

Design
------
* SelectorConfig         — strategy-agnostic configuration dataclass; every
                           strategy reads only the fields it needs, so the
                           orchestrator passes one object and stays decoupled
                           from each strategy's internal parameter names.
* ModelSelector          — abstract interface (Strategy Pattern)
* TopKByMetricSelector   — selects top-K models ranked by a single metric
* StabilityAwareSelector — penalises high-variance models via a
                           stability coefficient
* GeneralizationSelector — rewards models whose train/test gap is small
* ModelSelectorFactory   — creates the right strategy from a name string

Consistency contract
--------------------
All three strategies share one constructor signature:

    __init__(self, config: SelectorConfig) -> None

The factory always calls ``StrategyClass(config=cfg)``  — never with
positional keyword arguments whose names differ per class.  This removes
the mismatch that previously existed between the orchestrator (which
passed ``metric=...``) and ``StabilityAwareSelector`` /
``GeneralizationSelector`` (which expected ``primary_metric=...`` /
``test_metric=...``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Shared configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class SelectorConfig:
    """
    Strategy-agnostic configuration for all ModelSelector implementations.

    Each strategy reads only the fields relevant to it; unused fields are
    simply ignored.  This means the orchestrator never needs to know which
    concrete strategy is in use — it always passes the same SelectorConfig.

    Parameters
    ----------
    primary_metric : str
        Main metric column to rank / optimise (e.g. "Accuracy", "R2").
        Used by TopKByMetricSelector and StabilityAwareSelector.
    stability_weight : float
        Penalty weight for metric variance in StabilityAwareSelector.
        Range [0, 1].  0 = ignore instability; 1 = fully penalise.
    train_metric : str | None
        Column name for train-set performance in GeneralizationSelector.
        If None or not present in the results DataFrame, the strategy
        logs a warning and falls back to plain ``primary_metric`` ranking
        instead of silently degrading without notice.
    gap_penalty : float
        Penalty weight for the train/test metric gap in
        GeneralizationSelector.
    """

    primary_metric: str = "Accuracy"
    stability_weight: float = 0.2
    train_metric: Optional[str] = None
    gap_penalty: float = 0.5


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class ModelSelector(ABC):
    """
    Abstract interface for model selection strategies.

    Responsibilities
    ----------------
    Receive a results DataFrame (one row per model, columns = metrics)
    and return the top-k rows, sorted best-first.

    All concrete strategies are constructed with a single ``SelectorConfig``
    so the factory and orchestrator never depend on strategy-specific
    parameter names.
    """

    @abstractmethod
    def select(
        self,
        results_df: pd.DataFrame,
        k: int = 3,
    ) -> pd.DataFrame:
        """
        Select the top *k* candidate models.

        Parameters
        ----------
        results_df : pd.DataFrame
            Must contain at least a "Model" column and one or more metric
            columns produced by the Evaluator component.
        k : int
            Number of candidates to return.

        Returns
        -------
        pd.DataFrame
            Subset of *results_df* containing the selected rows, sorted
            best-first.
        """


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------

class TopKByMetricSelector(ModelSelector):
    """
    Select the top-K models ranked by ``config.primary_metric`` (descending).
    """

    def __init__(self, config: SelectorConfig) -> None:
        self._metric = config.primary_metric

    def select(self, results_df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
        if self._metric not in results_df.columns:
            raise ValueError(
                f"Metric '{self._metric}' not found in results. "
                f"Available: {results_df.columns.tolist()}"
            )
        return (
            results_df
            .dropna(subset=[self._metric])
            .nlargest(k, self._metric)
            .reset_index(drop=True)
        )


class StabilityAwareSelector(ModelSelector):
    """
    Select models that balance performance with low cross-metric variance.

    Composite score = ``primary_metric`` * (1 − ``stability_weight`` × penalty)
    where penalty is the row-wise standard deviation across all numeric metric
    columns, normalised to [0, 1].

    Reads from ``SelectorConfig``: ``primary_metric``, ``stability_weight``.
    """

    def __init__(self, config: SelectorConfig) -> None:
        self._metric = config.primary_metric
        self._weight = config.stability_weight

    def select(self, results_df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
        numeric_cols = results_df.select_dtypes(include="number").columns.tolist()
        if self._metric not in numeric_cols:
            raise ValueError(
                f"Primary metric '{self._metric}' not found in numeric columns. "
                f"Available numeric columns: {numeric_cols}"
            )

        df = results_df.dropna(subset=[self._metric]).copy()

        # Row-wise std across all numeric metrics as instability proxy
        if len(numeric_cols) > 1:
            row_std = df[numeric_cols].std(axis=1)
            max_std = row_std.max() or 1.0
            penalty = row_std / max_std
        else:
            penalty = pd.Series(0.0, index=df.index)

        df["_composite_score"] = df[self._metric] * (1.0 - self._weight * penalty)
        return (
            df.nlargest(k, "_composite_score")
              .drop(columns=["_composite_score"])
              .reset_index(drop=True)
        )


class GeneralizationSelector(ModelSelector):
    """
    Prefer models that generalise well (small train/test performance gap).

    Score = ``primary_metric`` − ``gap_penalty`` × |train_metric − primary_metric|

    If ``train_metric`` is None or the column is absent from the DataFrame,
    the selector emits a warning and falls back to plain ``primary_metric``
    ranking rather than silently returning a misleading result.

    Reads from ``SelectorConfig``: ``primary_metric``, ``train_metric``,
    ``gap_penalty``.
    """

    def __init__(self, config: SelectorConfig) -> None:
        self._test_metric = config.primary_metric
        self._train_metric = config.train_metric
        self._gap_penalty = config.gap_penalty

    def select(self, results_df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
        if self._test_metric not in results_df.columns:
            raise ValueError(
                f"Test metric '{self._test_metric}' not found in results. "
                f"Available: {results_df.columns.tolist()}"
            )

        df = results_df.dropna(subset=[self._test_metric]).copy()

        if self._train_metric and self._train_metric in df.columns:
            gap = (df[self._train_metric] - df[self._test_metric]).abs()
            df["_gen_score"] = df[self._test_metric] - self._gap_penalty * gap
        else:
            if self._train_metric:
                # train_metric was requested but is absent — warn explicitly
                import warnings
                warnings.warn(
                    f"GeneralizationSelector: train_metric column "
                    f"'{self._train_metric}' not found in results DataFrame "
                    f"(columns: {results_df.columns.tolist()}). "
                    f"Falling back to plain '{self._test_metric}' ranking. "
                    f"Ensure the evaluator records train-set scores under "
                    f"this column name if generalisation scoring is required.",
                    UserWarning,
                    stacklevel=2,
                )
            df["_gen_score"] = df[self._test_metric]

        return (
            df.nlargest(k, "_gen_score")
              .drop(columns=["_gen_score"])
              .reset_index(drop=True)
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class ModelSelectorFactory:
    """
    Factory for ModelSelector strategies.

    All strategies are constructed with ``StrategyClass(config=cfg)``.
    The caller supplies a ``SelectorConfig``; the factory never passes
    strategy-specific keyword arguments.

    Registered strategies
    ---------------------
    "topk"           → TopKByMetricSelector
    "stability"      → StabilityAwareSelector
    "generalization" → GeneralizationSelector
    """

    _registry: dict[str, type[ModelSelector]] = {
        "topk":           TopKByMetricSelector,
        "stability":      StabilityAwareSelector,
        "generalization": GeneralizationSelector,
    }

    @classmethod
    def register(cls, name: str, selector_class: type[ModelSelector]) -> None:
        """Register a custom ModelSelector strategy."""
        cls._registry[name.lower()] = selector_class

    @classmethod
    def create(cls, strategy: str = "topk", config: Optional[SelectorConfig] = None) -> ModelSelector:
        """
        Instantiate a ModelSelector strategy.

        Parameters
        ----------
        strategy : str
            One of the registered strategy names.
        config : SelectorConfig, optional
            Selection configuration.  A default ``SelectorConfig()`` is used
            when not supplied.

        Returns
        -------
        ModelSelector

        Raises
        ------
        ValueError
            If *strategy* is not registered.
        """
        key = strategy.lower()
        if key not in cls._registry:
            raise ValueError(
                f"Unknown selection strategy '{strategy}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        cfg = config if config is not None else SelectorConfig()
        return cls._registry[key](config=cfg)
