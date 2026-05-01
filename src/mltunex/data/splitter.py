"""
Data splitting implementation for MLTuneX.

This module provides concrete implementation of data splitting operations
using scikit-learn's train_test_split functionality. It implements the
BaseDataSplitter interface to maintain consistency across the framework.
"""

from typing import Tuple, AnyStr
from sklearn.model_selection import train_test_split
from mltunex.data.base import BaseDataSplitter


class Data_Splitter(BaseDataSplitter):
    """
    Implementation of data splitting operations.

    This class provides concrete implementation for splitting data into
    training and testing sets using scikit-learn's train_test_split.
    It ensures reproducibility by using a fixed random state.

    Methods
    -------
    split_data(X, y, test_size: float) -> Tuple
        Splits input data into training and testing sets.
    """

    def split_data(self, X, y, test_size: float = 0.2) -> Tuple:
        """
        Split the data into training and testing sets.

        Parameters
        ----------
        X : array-like
            Features matrix.
        y : array-like
            Target variable vector.
        test_size : float, optional (default=0.2)
            The proportion of the dataset to include in the test split.

        Returns
        -------
        Tuple
            (X_train, X_test, y_train, y_test) containing:
            - X_train: Training features
            - X_test: Testing features
            - y_train: Training target
            - y_test: Testing target

        Notes
        -----
        Uses fixed random_state=42 for reproducibility.
        """
        return train_test_split(X, y, test_size=test_size, random_state=42)