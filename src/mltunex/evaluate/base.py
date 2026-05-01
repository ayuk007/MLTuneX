"""
Base evaluator module for MLTuneX.

This module defines the abstract base class for model evaluators, providing
a common interface for evaluating machine learning models. It ensures consistent
evaluation functionality across different implementations.

Classes
-------
BaseEvaluator : Abstract base class for model evaluation
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseEvaluator(ABC):
    """
    Abstract base class for model evaluators.

    This class defines the interface for model evaluation in MLTuneX. It provides
    an abstract method for evaluating models, ensuring consistent evaluation
    functionality across different implementations.

    Methods
    -------
    evaluate(model, X_test, y_test)
        Abstract method to evaluate a model with given test data.
    """

    @abstractmethod
    def evaluate(self, 
                model: Any, 
                X_test: Any, 
                y_test: Any) -> Dict[str, Optional[Dict[str, float]]]:
        """
        Evaluate a model using test data.

        Parameters
        ----------
        model : Any
            The trained model to evaluate.
        X_test : Any
            Test features.
        y_test : Any
            True test labels.

        Returns
        -------
        Dict[str, Optional[Dict[str, float]]]
            Nested dictionary containing evaluation metrics:
            - Outer key: model name
            - Inner key: metric names
            - Values: computed metric scores

        Raises
        ------
        NotImplementedError
            If the concrete class does not implement this method.
        """
        pass