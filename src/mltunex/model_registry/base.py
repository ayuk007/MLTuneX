from abc import ABC, abstractmethod
from typing import List, Dict, Type, Tuple

ModelType = Tuple[str, Type]

class BaseModelRegistry(ABC):
    """
    Abstract base class for model registries.
    """

    @abstractmethod
    def get_classification_models(self) -> List[ModelType]:
        pass

    @abstractmethod
    def get_regression_models(self) -> List[ModelType]:
        pass

    @abstractmethod
    def get_models(self, task_type: str) -> List[ModelType]:
        pass