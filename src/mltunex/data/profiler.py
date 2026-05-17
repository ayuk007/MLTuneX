"""
DataProfiler interface and strategy implementations for MLTuneX.

Defines a pluggable profiling system where:
  - DataProfiler  — the abstract interface every profiler must satisfy
  - BasicDataProfiler — lightweight summary suitable for most AutoML flows
  - ExtendedDataProfiler — richer analysis (skew, kurtosis, correlations)

Strategy Pattern: callers depend only on DataProfiler; concrete strategies
are swapped without touching any consumer code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class DataProfiler(ABC):
    """
    Abstract interface for dataset profiling.

    Responsibilities
    ----------------
    Analyse a DataFrame and return a metadata dictionary that downstream
    components (preprocessing, AI advisor, optimizer) can consume.
    This class deliberately has NO side-effects on the data it receives.
    """

    @abstractmethod
    def profile(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """
        Analyse *df* and return a metadata dictionary.

        Parameters
        ----------
        df : pd.DataFrame
            The full dataset (features + target).
        target_column : str
            Name of the target column.

        Returns
        -------
        Dict[str, Any]
            A flat-ish dictionary containing profiling metadata.
            Keys are standardised (see concrete classes for full schemas).
        """


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------

class BasicDataProfiler(DataProfiler):
    """
    Lightweight profiler that captures the most actionable dataset metadata.

    Produced keys
    -------------
    num_rows, num_features, feature_names, target_column,
    dtypes, missing_counts, missing_pct, cardinality,
    target_distribution, imbalance_ratio (classification only)
    """

    def profile(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        feature_cols = [c for c in df.columns if c != target_column]
        features = df[feature_cols]
        target = df[target_column]

        missing_counts = df.isnull().sum().to_dict()
        missing_pct = {k: round(v / len(df) * 100, 2) for k, v in missing_counts.items()}

        cardinality = {col: int(df[col].nunique()) for col in df.columns}

        target_dist = target.value_counts(normalize=True).to_dict()
        target_dist = {str(k): round(v, 4) for k, v in target_dist.items()}

        # Imbalance ratio (max_class_freq / min_class_freq) — meaningful for
        # classification; set to None for continuous targets.
        imbalance_ratio = None
        if target.nunique() <= 30:
            counts = target.value_counts()
            if counts.min() > 0:
                imbalance_ratio = round(counts.max() / counts.min(), 2)

        numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = features.select_dtypes(exclude=[np.number]).columns.tolist()

        return {
            "num_rows": int(len(df)),
            "num_features": int(len(feature_cols)),
            "feature_names": feature_cols,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "target_column": target_column,
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_counts": missing_counts,
            "missing_pct": missing_pct,
            "cardinality": cardinality,
            "target_distribution": target_dist,
            "imbalance_ratio": imbalance_ratio,
        }


class ExtendedDataProfiler(DataProfiler):
    """
    Rich profiler that adds distribution, variance, correlation and
    shape statistics on top of the basic profile.

    Produced keys
    -------------
    All keys from BasicDataProfiler PLUS:
    variance_groups, skewness, kurtosis, correlation_matrix,
    describe_stats
    """

    def __init__(self) -> None:
        self._basic = BasicDataProfiler()

    def profile(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        meta = self._basic.profile(df, target_column)

        numeric_df = df.select_dtypes(include=[np.number])

        # Variance grouping
        stds = numeric_df.std()
        lo = stds.quantile(0.05)
        hi = stds.quantile(0.95)
        meta["variance_groups"] = {
            "dead_or_constant":  stds[stds == 0].index.tolist(),
            "low_variance":      stds[(stds > 0) & (stds <= lo)].index.tolist(),
            "medium_variance":   stds[(stds > lo) & (stds <= hi)].index.tolist(),
            "high_variance":     stds[stds > hi].index.tolist(),
        }

        meta["skewness"] = numeric_df.skew().to_dict()
        meta["kurtosis"] = numeric_df.kurt().to_dict()
        meta["correlation_matrix"] = numeric_df.corr(method="pearson").to_dict()
        meta["describe_stats"] = numeric_df.describe().to_dict()

        return meta


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

class DataProfilerFactory:
    """
    Factory that returns a DataProfiler strategy by name.

    Strategies
    ----------
    "basic"    → BasicDataProfiler
    "extended" → ExtendedDataProfiler
    """

    _registry: dict[str, type[DataProfiler]] = {
        "basic":    BasicDataProfiler,
        "extended": ExtendedDataProfiler,
    }

    @classmethod
    def register(cls, name: str, profiler_class: type[DataProfiler]) -> None:
        """Register a custom profiler strategy."""
        cls._registry[name.lower()] = profiler_class

    @classmethod
    def create(cls, strategy: str = "extended") -> DataProfiler:
        """
        Return a DataProfiler instance for *strategy*.

        Parameters
        ----------
        strategy : str
            One of the registered strategy names.

        Raises
        ------
        ValueError
            If *strategy* is not registered.
        """
        key = strategy.lower()
        if key not in cls._registry:
            raise ValueError(
                f"Unknown profiling strategy '{strategy}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[key]()
