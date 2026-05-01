"""
Library trainer factory module for MLTuneX.

This module provides a factory class for creating model trainers based on 
the specified machine learning library. Currently supports scikit-learn models
with extensibility for other libraries.

Examples:
    >>> trainer = LibraryTrainer.get_trainer("sklearn")
    >>> trained_model = trainer.train_model(RandomForestClassifier, X_train, y_train)
"""

from mltunex.library_trainer.base import BaseLibraryTrainer
from mltunex.library_trainer.sklearn_trainer import SklearnTrainer


class LibraryTrainer:
    """
    Factory class for creating model trainers.

    This class implements the factory pattern to create appropriate trainer
    instances based on the specified machine learning library. Currently
    supports scikit-learn with extensibility for other libraries.

    Methods
    -------
    get_trainer(library: str) -> BaseLibraryTrainer
        Creates and returns a trainer for the specified library.
    """

    @staticmethod
    def get_trainer(library: str) -> BaseLibraryTrainer:
        """
        Create and return a trainer for the specified library.

        Parameters
        ----------
        library : str
            Name of the machine learning library to create trainer for.
            Current supported values: "sklearn"

        Returns
        -------
        BaseLibraryTrainer
            Trainer instance for the specified library.

        Raises
        ------
        ValueError
            If the specified library is not supported.

        Examples
        --------
        >>> trainer = LibraryTrainer.get_trainer("sklearn")
        >>> model = trainer.train_model(RandomForestClassifier, X, y)
        """
        # Return appropriate trainer based on library name
        if library == "sklearn":
            return SklearnTrainer()
        else:
            raise ValueError(f"Unsupported library: {library}")