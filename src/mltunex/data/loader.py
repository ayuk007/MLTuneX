"""
Data loading implementations for MLTuneX.

This module provides concrete implementations of data loading operations
for different data sources including pandas DataFrames and CSV files.
It follows a factory pattern to create appropriate loaders based on
the input source type.
"""

import pandas as pd
from typing import Tuple, Union, AnyStr
from mltunex.data.base import BaseDataLoader

class DataFrame_Loader(BaseDataLoader):
    """
    Data loader implementation for pandas DataFrame input.

    This class handles loading data from an existing pandas DataFrame,
    separating features and target variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input pandas DataFrame containing both features and target variable.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DataFrameLoader with a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame to load data from.
        """
        self.df = df

    def load_data(self, target_column: str) -> Tuple:
        """
        Load data from DataFrame and return features and target.

        Parameters
        ----------
        target_column : str
            Name of the column to use as target variable.

        Returns
        -------
        Tuple
            (X, y) where X is the feature matrix and y is the target vector.
        """
        X = self.df.drop(target_column, axis=1)
        y = self.df[target_column]
        return X, y


class CSVLoader(BaseDataLoader):
    """
    Data loader implementation for CSV file input.

    This class handles loading data from a CSV file,
    separating features and target variables.

    Parameters
    ----------
    path_to_csv : str
        Path to the CSV file containing the data.
    """

    def __init__(self, path_to_csv: str):
        """
        Initialize the CSVLoader with the path to the CSV file.

        Parameters
        ----------
        path_to_csv : str
            File path to the CSV file.
        """
        self.path_to_csv = path_to_csv

    def load_data(self, target_column: str) -> Tuple:
        """
        Load data from CSV and return features and target.

        Parameters
        ----------
        target_column : str
            Name of the column to use as target variable.

        Returns
        -------
        Tuple
            (X, y) where X is the feature matrix and y is the target vector.

        Raises
        ------
        ValueError
            If the target column is not found in the CSV file.
        """
        df = pd.read_csv(self.path_to_csv)

        # Check if target_column exists in the dataframe
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the CSV.")

        X = df.drop(target_column, axis=1)
        y = df[target_column]
        return X, y


class Data_Loader_Factory:
    """
    Factory class to create appropriate data loaders.

    This class implements the factory pattern to create data loader
    instances based on the input source type.
    """

    @staticmethod
    def get_data_loader(source: AnyStr) -> BaseDataLoader:
        """
        Create and return appropriate data loader based on source type.

        Parameters
        ----------
        source : Union[str, pd.DataFrame]
            Data source (either file path or DataFrame).

        Returns
        -------
        BaseDataLoader
            Appropriate data loader instance for the source.

        Raises
        ------
        ValueError
            If the source type is not supported.
        """
        if isinstance(source, str) and source.endswith(".csv"):
            return CSVLoader(source)
        if isinstance(source, pd.DataFrame):
            return DataFrame_Loader(source)
        else:
            raise ValueError(f"Unsupported data source: {source}")


class Data_Loader:
    """
    Main loader class responsible for loading data from various sources.

    This class serves as the main interface for data loading operations,
    utilizing the factory pattern to handle different data sources.
    """

    def __init__(self):
        """Initialize Data_Loader with a factory instance."""
        self.data_loader_factory = Data_Loader_Factory()

    def load_data(self, source: Union[AnyStr, pd.DataFrame], target_column: str = None) -> Tuple:
        """
        Load data from the specified source.

        Parameters
        ----------
        source : Union[str, pd.DataFrame]
            Data source (either file path or DataFrame).
        target_column : str, optional
            Name of the target variable column.

        Returns
        -------
        Tuple
            (X, y) where X is the feature matrix and y is the target vector.
        """
        loader = self.data_loader_factory.get_data_loader(source)
        return loader.load_data(target_column)