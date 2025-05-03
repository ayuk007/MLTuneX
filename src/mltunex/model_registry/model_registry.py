"""
Model Registry factory module for MLTuneX.

This module provides a factory class for creating model registries based on 
the specified machine learning library. Currently supports scikit-learn models
with extensibility for other libraries.

Example:
    >>> registry = Model_Registry.get_model_registry("sklearn")
    >>> models = registry.get_models("classification")
"""

from dataclasses import dataclass
from typing import List, Dict
from mltunex.model_registry.base import BaseModelRegistry
from mltunex.model_registry.sklearn_registry import SkLearn_Model_Registry


class Model_Registry:
    """
    Factory class for creating model registries.

    This class implements the factory pattern to create appropriate model
    registry instances based on the specified machine learning library.
    Currently supports scikit-learn with extensibility for other libraries.

    Methods
    -------
    get_model_registry(models_library: str) -> BaseModelRegistry
        Creates and returns a model registry for the specified library.
    """

    @staticmethod
    def get_model_registry(models_library: str) -> BaseModelRegistry:
        """
        Create and return a model registry for the specified library.

        Parameters
        ----------
        models_library : str
            Name of the machine learning library to create registry for.
            Current supported values: "sklearn"

        Returns
        -------
        BaseModelRegistry
            Model registry instance for the specified library.

        Raises
        ------
        ValueError
            If the specified models_library is not supported.

        Examples
        --------
        >>> registry = Model_Registry.get_model_registry("sklearn")
        >>> classification_models = registry.get_classification_models()
        """
        # Return appropriate registry based on library name
        if models_library == "sklearn":
            return SkLearn_Model_Registry
        else:
            raise ValueError(f"Unsupported models library: {models_library}")