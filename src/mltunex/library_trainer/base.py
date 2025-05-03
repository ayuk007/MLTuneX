"""
Base library trainer module for MLTuneX.

This module defines the abstract base class for model trainers, providing
a common interface for training models from different machine learning libraries.
It ensures consistent model training functionality across different implementations.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseLibraryTrainer(ABC):
    """
    Abstract base class for model trainers.

    This class defines the interface for model trainers in MLTuneX. It provides
    an abstract method for training models, ensuring consistent training
    functionality across different machine learning library implementations.

    Methods
    -------
    train_model(model, X_train, y_train)
        Abstract method to train a model with given data.
    """

    @abstractmethod
    def train_model(self, model: Any, X_train: Any, y_train: Any) -> Any:
        """
        Train a model with the given training data.

        Parameters
        ----------
        model : Any
            The model class or instance to train.
        X_train : Any
            Training features.
        y_train : Any
            Training labels/target values.

        Returns
        -------
        Any
            Trained model instance.

        Raises
        ------
        NotImplementedError
            If the concrete class does not implement this method.
        """
        pass