"""
Base model registry module for MLTuneX.

This module defines the abstract base class for model registries, providing
a common interface for managing machine learning models. It ensures consistent
model access and management across different model types and libraries.

Attributes:
    ModelType: Type alias for tuple containing (model_name: str, model_class: Type)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Type, Tuple

# Type alias for model definitions
ModelType = Tuple[str, Type]

class BaseModelRegistry(ABC):
    """
    Abstract base class for model registries.

    This class defines the interface for model registries in MLTuneX. It provides
    abstract methods for accessing classification and regression models, ensuring
    consistent model management across different implementations.

    Methods
    -------
    get_classification_models() -> List[ModelType]
        Get available classification models.
    get_regression_models() -> List[ModelType]
        Get available regression models.
    get_models(task_type: str) -> List[ModelType]
        Get models based on task type.
    """

    @abstractmethod
    def get_classification_models(self) -> List[ModelType]:
        """
        Get available classification models.

        Returns
        -------
        List[ModelType]
            List of tuples containing (model_name, model_class) for all
            available classification models.

        Raises
        ------
        NotImplementedError
            If the concrete class does not implement this method.
        """
        pass

    @abstractmethod
    def get_regression_models(self) -> List[ModelType]:
        """
        Get available regression models.

        Returns
        -------
        List[ModelType]
            List of tuples containing (model_name, model_class) for all
            available regression models.

        Raises
        ------
        NotImplementedError
            If the concrete class does not implement this method.
        """
        pass

    @abstractmethod
    def get_models(self, task_type: str) -> List[ModelType]:
        """
        Get models based on the specified task type.

        Parameters
        ----------
        task_type : str
            Type of machine learning task ('classification' or 'regression').

        Returns
        -------
        List[ModelType]
            List of tuples containing (model_name, model_class) appropriate
            for the specified task type.

        Raises
        ------
        NotImplementedError
            If the concrete class does not implement this method.
        ValueError
            If the task_type is not supported.
        """
        pass