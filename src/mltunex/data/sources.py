"""
DataSource abstractions and concrete implementations for MLTuneX.

Defines a generic DataSource interface and concrete implementations for
CSV, Excel, Parquet, Feather, SQL, and in-memory sources. A DataSourceFactory
creates the appropriate instance based on the source type.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class DataSource(ABC):
    """
    Abstract interface for all data sources.

    Responsibilities
    ----------------
    Provide a unified DataFrame regardless of the underlying storage
    format or location.  All concrete implementations must fulfil this
    single contract so the rest of the system remains source-agnostic.
    """

    @abstractmethod
    def read(self) -> pd.DataFrame:
        """
        Read data and return a unified DataFrame.

        Returns
        -------
        pd.DataFrame
            The complete dataset as a DataFrame.

        Raises
        ------
        IOError
            If the underlying source cannot be accessed.
        """


# ---------------------------------------------------------------------------
# Concrete sources
# ---------------------------------------------------------------------------

class CSVDataSource(DataSource):
    """DataSource backed by a CSV file."""

    def __init__(self, path: str, **read_kwargs: Any) -> None:
        self._path = path
        self._read_kwargs = read_kwargs

    def read(self) -> pd.DataFrame:
        return pd.read_csv(self._path, **self._read_kwargs)


class ExcelDataSource(DataSource):
    """DataSource backed by an Excel file (.xlsx / .xls)."""

    def __init__(self, path: str, **read_kwargs: Any) -> None:
        self._path = path
        self._read_kwargs = read_kwargs

    def read(self) -> pd.DataFrame:
        return pd.read_excel(self._path, **self._read_kwargs)


class ParquetDataSource(DataSource):
    """DataSource backed by a Parquet file."""

    def __init__(self, path: str, **read_kwargs: Any) -> None:
        self._path = path
        self._read_kwargs = read_kwargs

    def read(self) -> pd.DataFrame:
        return pd.read_parquet(self._path, **self._read_kwargs)


class FeatherDataSource(DataSource):
    """DataSource backed by a Feather file."""

    def __init__(self, path: str, **read_kwargs: Any) -> None:
        self._path = path
        self._read_kwargs = read_kwargs

    def read(self) -> pd.DataFrame:
        return pd.read_feather(self._path, **self._read_kwargs)


class SQLDataSource(DataSource):
    """
    DataSource backed by a SQL query.

    Parameters
    ----------
    query : str
        SQL SELECT statement to execute.
    connection : Any
        A SQLAlchemy engine, connection, or any object accepted by
        ``pandas.read_sql``.
    """

    def __init__(self, query: str, connection: Any) -> None:
        self._query = query
        self._connection = connection

    def read(self) -> pd.DataFrame:
        return pd.read_sql(self._query, self._connection)


class InMemoryDataSource(DataSource):
    """DataSource wrapping an already-loaded DataFrame."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("InMemoryDataSource requires a pandas DataFrame.")
        self._df = dataframe

    def read(self) -> pd.DataFrame:
        return self._df.copy()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class DataSourceFactory:
    """
    Factory for creating DataSource instances.

    Uses the Open/Closed principle: new source types are registered via
    ``register`` without modifying existing code.

    Examples
    --------
    >>> factory = DataSourceFactory()
    >>> source = factory.create("path/to/data.csv")
    >>> df = source.read()
    """

    # Default extension → DataSource class mapping
    _registry: dict[str, type[DataSource]] = {
        ".csv":     CSVDataSource,
        ".xlsx":    ExcelDataSource,
        ".xls":     ExcelDataSource,
        ".parquet": ParquetDataSource,
        ".feather": FeatherDataSource,
    }

    @classmethod
    def register(cls, extension: str, source_class: type[DataSource]) -> None:
        """Register a new DataSource class for a file extension."""
        cls._registry[extension.lower()] = source_class

    @classmethod
    def create(cls, source: Any, **kwargs: Any) -> DataSource:
        """
        Create the appropriate DataSource for *source*.

        Parameters
        ----------
        source : str | pd.DataFrame | Any
            A file path, a pandas DataFrame, or an object whose string
            representation contains a registered extension.
        **kwargs
            Additional keyword arguments forwarded to the DataSource constructor.

        Returns
        -------
        DataSource

        Raises
        ------
        ValueError
            If the source type cannot be matched to a registered DataSource.
        """
        if isinstance(source, pd.DataFrame):
            return InMemoryDataSource(source)

        if isinstance(source, str):
            import os
            _, ext = os.path.splitext(source.lower())
            if ext in cls._registry:
                return cls._registry[ext](source, **kwargs)

        raise ValueError(
            f"Unsupported data source: {source!r}. "
            f"Registered extensions: {list(cls._registry.keys())}"
        )
