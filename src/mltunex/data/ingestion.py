"""
Data ingestion module for MLTuneX.

This module provides a unified interface for data ingestion operations,
combining data loading and splitting functionalities. It serves as a
high-level API for preparing data for machine learning tasks.
"""

from typing import AnyStr, Tuple, Union
from mltunex.data.loader import Data_Loader
from mltunex.data.splitter import Data_Splitter


class Data_Ingestion:
    """
    Class to handle data ingestion by combining loading and splitting.

    This class provides a unified interface for loading data from various
    sources and splitting it into training and testing sets. It encapsulates
    both the Data_Loader and Data_Splitter classes to provide a seamless
    data ingestion pipeline.

    Attributes
    ----------
    data_loader : Data_Loader
        Instance of Data_Loader to handle data loading operations.
    data_splitter : Data_Splitter
        Instance of Data_Splitter to handle data splitting operations.
    """

    def __init__(self):
        """
        Initialize Data_Ingestion with loader and splitter instances.

        Creates new instances of Data_Loader and Data_Splitter for
        handling data operations.
        """
        self.data_loader = Data_Loader()
        self.data_splitter = Data_Splitter()

    def ingest_data(self, source: AnyStr, target_column: str, test_size: float = 0.2) -> Tuple:
        """
        Load and split data from the specified source.

        This method provides a single interface for both loading data
        from a source and splitting it into training and testing sets.

        Parameters
        ----------
        source : AnyStr
            Data source (either file path or DataFrame).
        target_column : str
            Name of the target variable column.
        test_size : float, optional (default=0.2)
            Proportion of the dataset to include in the test split.

        Returns
        -------
        Tuple
            (X_train, X_test, y_train, y_test) containing split dataset.

        Examples
        --------
        >>> ingestion = Data_Ingestion()
        >>> X_train, X_test, y_train, y_test = ingestion.ingest_data(
        ...     "data.csv", "target", 0.2
        ... )
        """
        # Load the data from source
        X, y = self.data_loader.load_data(source, target_column)
        # Split the data into training and testing sets
        return self.data_splitter.split_data(X, y, test_size)