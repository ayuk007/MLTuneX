"""
Preprocessing engine for MLTuneX.

Design
------
* PreprocessingStrategy  — abstract interface (Strategy Pattern)
* Concrete strategies    — NumericImputer, CategoricalImputer, StandardScaler,
                           MinMaxScaler, OneHotEncoder, OrdinalEncoder,
                           OutlierClipper
* PreprocessingPipeline  — assembles and applies strategies in order
* PreprocessingPipelineBuilder — constructs the pipeline (Builder Pattern)
* AdaptivePipelineDirector — builds a pipeline from a DataProfiler metadata
                             dict without callers knowing which strategies are
                             chosen (profiles-driven auto-configuration)

The engine is model-aware through the *task_type* hint passed to the
director, but it remains loosely coupled from all model implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler as _MinMaxScaler,
    OneHotEncoder as _OneHotEncoder,
    OrdinalEncoder as _OrdinalEncoder,
    StandardScaler as _StandardScaler,
)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class PreprocessingStrategy(ABC):
    """
    Abstract interface for a single preprocessing transformation.

    Responsibilities
    ----------------
    Transform a subset (or all) of the columns of a DataFrame.
    Each strategy encapsulates one coherent transformation.
    """

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "PreprocessingStrategy":
        """Fit the strategy on *df*; return self for chaining."""

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted transformation to *df*; return new DataFrame."""

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience method: fit then transform."""
        return self.fit(df).transform(df)


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------

class NumericImputer(PreprocessingStrategy):
    """Fill missing values in numeric columns (default: median)."""

    def __init__(self, strategy: str = "median") -> None:
        self._strategy = strategy
        self._imputer: Optional[SimpleImputer] = None
        self._cols: List[str] = []

    def fit(self, df: pd.DataFrame) -> "NumericImputer":
        self._cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if self._cols:
            self._imputer = SimpleImputer(strategy=self._strategy)
            self._imputer.fit(df[self._cols])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self._cols and self._imputer is not None:
            out[self._cols] = self._imputer.transform(out[self._cols])
        return out


class CategoricalImputer(PreprocessingStrategy):
    """Fill missing values in categorical columns (default: most_frequent)."""

    def __init__(self, strategy: str = "most_frequent") -> None:
        self._strategy = strategy
        self._imputer: Optional[SimpleImputer] = None
        self._cols: List[str] = []

    def fit(self, df: pd.DataFrame) -> "CategoricalImputer":
        self._cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if self._cols:
            self._imputer = SimpleImputer(strategy=self._strategy)
            self._imputer.fit(df[self._cols])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self._cols and self._imputer is not None:
            out[self._cols] = self._imputer.transform(out[self._cols])
        return out


class StandardScalerStrategy(PreprocessingStrategy):
    """Z-score normalisation of numeric columns."""

    def __init__(self) -> None:
        self._scaler: Optional[_StandardScaler] = None
        self._cols: List[str] = []

    def fit(self, df: pd.DataFrame) -> "StandardScalerStrategy":
        self._cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if self._cols:
            self._scaler = _StandardScaler()
            self._scaler.fit(df[self._cols])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self._cols and self._scaler is not None:
            out[self._cols] = self._scaler.transform(out[self._cols])
        return out


class MinMaxScalerStrategy(PreprocessingStrategy):
    """Min-max scaling of numeric columns to [0, 1]."""

    def __init__(self) -> None:
        self._scaler: Optional[_MinMaxScaler] = None
        self._cols: List[str] = []

    def fit(self, df: pd.DataFrame) -> "MinMaxScalerStrategy":
        self._cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if self._cols:
            self._scaler = _MinMaxScaler()
            self._scaler.fit(df[self._cols])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self._cols and self._scaler is not None:
            out[self._cols] = self._scaler.transform(out[self._cols])
        return out


class OneHotEncoderStrategy(PreprocessingStrategy):
    """One-hot encode low-cardinality categorical columns."""

    def __init__(self, max_cardinality: int = 20) -> None:
        self._max_cardinality = max_cardinality
        self._encoder: Optional[_OneHotEncoder] = None
        self._cols: List[str] = []

    def fit(self, df: pd.DataFrame) -> "OneHotEncoderStrategy":
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        self._cols = [c for c in cat_cols if df[c].nunique() <= self._max_cardinality]
        if self._cols:
            self._encoder = _OneHotEncoder(
                handle_unknown="ignore", sparse_output=False
            )
            self._encoder.fit(df[self._cols])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self._cols and self._encoder is not None:
            encoded = pd.DataFrame(
                self._encoder.transform(out[self._cols]),
                columns=self._encoder.get_feature_names_out(self._cols),
                index=out.index,
            )
            out = out.drop(columns=self._cols)
            out = pd.concat([out, encoded], axis=1)
        return out


class OrdinalEncoderStrategy(PreprocessingStrategy):
    """Ordinal-encode high-cardinality or ordered categorical columns."""

    def __init__(self, min_cardinality: int = 21) -> None:
        self._min_cardinality = min_cardinality
        self._encoder: Optional[_OrdinalEncoder] = None
        self._cols: List[str] = []

    def fit(self, df: pd.DataFrame) -> "OrdinalEncoderStrategy":
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        self._cols = [c for c in cat_cols if df[c].nunique() > self._min_cardinality]
        if self._cols:
            self._encoder = _OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            self._encoder.fit(df[self._cols])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self._cols and self._encoder is not None:
            out[self._cols] = self._encoder.transform(out[self._cols])
        return out


class OutlierClipper(PreprocessingStrategy):
    """Clip numeric outliers to [Q1 - k*IQR, Q3 + k*IQR]."""

    def __init__(self, k: float = 1.5) -> None:
        self._k = k
        self._bounds: Dict[str, Tuple[float, float]] = {}
        self._cols: List[str] = []

    def fit(self, df: pd.DataFrame) -> "OutlierClipper":
        self._cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in self._cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            self._bounds[col] = (q1 - self._k * iqr, q3 + self._k * iqr)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self._cols:
            lo, hi = self._bounds[col]
            out[col] = out[col].clip(lo, hi)
        return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class PreprocessingPipeline:
    """
    Ordered sequence of PreprocessingStrategy objects.

    The pipeline applies each strategy in insertion order during both
    fit and transform phases.
    """

    def __init__(self) -> None:
        self._steps: List[Tuple[str, PreprocessingStrategy]] = []

    def add_step(self, name: str, strategy: PreprocessingStrategy) -> "PreprocessingPipeline":
        """Append a strategy step; returns self for fluent chaining."""
        self._steps.append((name, strategy))
        return self

    def fit(self, df: pd.DataFrame) -> "PreprocessingPipeline":
        for _, strategy in self._steps:
            df = strategy.fit_transform(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for _, strategy in self._steps:
            df = strategy.transform(df)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    @property
    def steps(self) -> List[Tuple[str, PreprocessingStrategy]]:
        return list(self._steps)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class PreprocessingPipelineBuilder:
    """
    Builder for constructing PreprocessingPipeline instances step by step.

    Usage
    -----
    >>> pipeline = (
    ...     PreprocessingPipelineBuilder()
    ...     .add_numeric_imputer()
    ...     .add_categorical_imputer()
    ...     .add_one_hot_encoder()
    ...     .add_standard_scaler()
    ...     .build()
    ... )
    """

    def __init__(self) -> None:
        self._pipeline = PreprocessingPipeline()

    def add_numeric_imputer(self, strategy: str = "median") -> "PreprocessingPipelineBuilder":
        self._pipeline.add_step("numeric_imputer", NumericImputer(strategy))
        return self

    def add_categorical_imputer(self, strategy: str = "most_frequent") -> "PreprocessingPipelineBuilder":
        self._pipeline.add_step("categorical_imputer", CategoricalImputer(strategy))
        return self

    def add_standard_scaler(self) -> "PreprocessingPipelineBuilder":
        self._pipeline.add_step("standard_scaler", StandardScalerStrategy())
        return self

    def add_min_max_scaler(self) -> "PreprocessingPipelineBuilder":
        self._pipeline.add_step("minmax_scaler", MinMaxScalerStrategy())
        return self

    def add_one_hot_encoder(self, max_cardinality: int = 20) -> "PreprocessingPipelineBuilder":
        self._pipeline.add_step("one_hot_encoder", OneHotEncoderStrategy(max_cardinality))
        return self

    def add_ordinal_encoder(self, min_cardinality: int = 21) -> "PreprocessingPipelineBuilder":
        self._pipeline.add_step("ordinal_encoder", OrdinalEncoderStrategy(min_cardinality))
        return self

    def add_outlier_clipper(self, k: float = 1.5) -> "PreprocessingPipelineBuilder":
        self._pipeline.add_step("outlier_clipper", OutlierClipper(k))
        return self

    def add_custom_step(
        self, name: str, strategy: PreprocessingStrategy
    ) -> "PreprocessingPipelineBuilder":
        """Plug in any custom PreprocessingStrategy without modifying the builder."""
        self._pipeline.add_step(name, strategy)
        return self

    def build(self) -> PreprocessingPipeline:
        """Return the assembled pipeline."""
        return self._pipeline


# ---------------------------------------------------------------------------
# Director — profile-driven auto-configuration
# ---------------------------------------------------------------------------

class AdaptivePipelineDirector:
    """
    Constructs a PreprocessingPipeline from a profiling metadata dict.

    The director inspects the profile produced by a DataProfiler and
    decides which strategies are appropriate, keeping the caller free
    from those details.

    Parameters
    ----------
    task_type : str
        "classification" or "regression" — influences scaling choice.
    """

    def __init__(self, task_type: str = "classification") -> None:
        self._task_type = task_type

    def build_from_profile(self, profile: Dict[str, Any]) -> PreprocessingPipeline:
        """
        Build a PreprocessingPipeline tailored to *profile*.

        Parameters
        ----------
        profile : Dict[str, Any]
            Output of a DataProfiler.profile() call.

        Returns
        -------
        PreprocessingPipeline
        """
        builder = PreprocessingPipelineBuilder()

        has_numeric = bool(profile.get("numeric_features"))
        has_categorical = bool(profile.get("categorical_features"))
        has_missing_numeric = any(
            profile.get("missing_counts", {}).get(c, 0) > 0
            for c in profile.get("numeric_features", [])
        )
        has_missing_categorical = any(
            profile.get("missing_counts", {}).get(c, 0) > 0
            for c in profile.get("categorical_features", [])
        )
        has_outliers = self._has_significant_outliers(profile)

        # Impute first so later steps receive complete data
        if has_numeric and has_missing_numeric:
            builder.add_numeric_imputer()
        if has_categorical and has_missing_categorical:
            builder.add_categorical_imputer()

        # Outlier clipping before scaling
        if has_numeric and has_outliers:
            builder.add_outlier_clipper()

        # Encode categoricals
        if has_categorical:
            builder.add_one_hot_encoder()
            builder.add_ordinal_encoder()

        # Scale numerics (regression → MinMax; classification → Standard)
        if has_numeric:
            if self._task_type == "regression":
                builder.add_min_max_scaler()
            else:
                builder.add_standard_scaler()

        return builder.build()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_significant_outliers(profile: Dict[str, Any]) -> bool:
        """Heuristic: any numeric feature with |skew| > 2 implies potential outliers."""
        skewness = profile.get("skewness", {})
        return any(abs(v) > 2.0 for v in skewness.values())
