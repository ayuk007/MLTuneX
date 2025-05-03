from dataclasses import dataclass
from typing import List, Dict
from mltunex.model_registry.base import BaseModelRegistry
from mltunex.model_registry.sklearn_registry import SkLearn_Model_Registry

class Model_Registry:
  @staticmethod
  def get_model_registry(models_library: str) -> BaseModelRegistry:
    if models_library == "sklearn":
      return SkLearn_Model_Registry
    else:
      raise ValueError(f"Unsupported models library: {models_library}")