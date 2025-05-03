"""
Base classes for data loading and splitting operations in MLTuneX.

This module provides abstract base classes that define the interface for
data loading and splitting operations. These classes serve as templates
for concrete implementations in the MLTuneX framework.
"""

from abc import ABC, abstractmethod, abstractstaticmethod
from typing import AnyStr, Tuple, Union


class BaseDataLoader(ABC):
    """
    Abstract base class for data loading operations.

    This class defines the interface for loading data from various sources.
    All concrete data loader implementations should inherit from this class
    and implement the load_data method.

    Methods
    -------
    load_data(target_column: AnyStr) -> Tuple
        Abstract method to load data from a source.
    """

    @abstractmethod
    def load_data(self, target_column: AnyStr) -> Tuple:
        """
        Load data from a source and prepare it for model training.

        Parameters
        ----------
        target_column : str
            The name of the target variable column.

        Returns
        -------
        Tuple
            A tuple containing features (X) and target variable (y).

        Raises
        ------
        NotImplementedError
            If the concrete class does not implement this method.
        """
        pass


class BaseDataSplitter(ABC):
    """
    Abstract base class for data splitting operations.

    This class defines the interface for splitting data into training
    and testing sets. All concrete data splitter implementations should
    inherit from this class and implement the split_data method.

    Methods
    -------
    split_data(X, y, test_size: float) -> Tuple
        Abstract method to split the data into training and testing sets.
    """

    @abstractmethod
    def split_data(self, X, y, test_size: float = 0.2) -> Tuple:
        """
        Split the dataset into training and testing sets.

        Parameters
        ----------
        X : array-like
            Features matrix.
        y : array-like
            Target variable.
        test_size : float, optional (default=0.2)
            The proportion of the dataset to include in the test split.

        Returns
        -------
        Tuple
            A tuple containing (X_train, X_test, y_train, y_test).

        Raises
        ------
        NotImplementedError
            If the concrete class does not implement this method.
        """
        pass