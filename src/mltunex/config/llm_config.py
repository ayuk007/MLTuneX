"""
mltunex.config.llm_config
──────────────────────────
LLM provider configuration dataclasses.

Changes from original
─────────────────────
* Removed hard model allowlists in __post_init__ — they prevented users
  from passing new model versions without editing source code.
  Validation of model names is now the provider's responsibility at
  API call time (they return a clear error if the model is unknown).

* LLMConfig.get_llm_config() is kept for backward compatibility but now
  constructs LLMHandlerConfig objects used by the new registry-based
  system, rather than the handler directly.

All existing call sites that do:
    LLMConfig.get_llm_config("Groq:qwen/qwen3-32b")
    LLMManager.get_llm_instance("Groq:qwen/qwen3-32b")
continue to work unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mltunex.ai_handler.prompt import LLMPrompts


@dataclass
class OpenAIConfig:
    """Configuration for OpenAI-backed handlers (legacy, kept for compat)."""
    model: str = "gpt-4o"
    temperature: float = 0.0
    SYSTEM_PROMPT: str = LLMPrompts.OpenAIPrompt


@dataclass
class GroqConfig:
    """Configuration for Groq-backed handlers (legacy, kept for compat)."""
    model: str = "qwen/qwen3-32b"
    temperature: float = 0.0
    SYSTEM_PROMPT: str = LLMPrompts.OpenAIPrompt


@dataclass
class LLMConfig:
    """
    Factory helper — returns the appropriate config dataclass for a given
    ``Provider:ModelName`` string.

    Still used by LLMManager for backward-compat; new code should use
    LLMHandlerRegistry.create() directly.
    """

    @staticmethod
    def get_llm_config(model_provider_model_name: str):
        """
        Return an OpenAIConfig or GroqConfig for *model_provider_model_name*.

        Parameters
        ----------
        model_provider_model_name : str
            ``"Provider:ModelName"`` format.

        Returns
        -------
        OpenAIConfig | GroqConfig
        """
        if ":" not in model_provider_model_name:
            raise ValueError(
                f"model_provider_model_name must be 'Provider:ModelName', "
                f"got '{model_provider_model_name}'."
            )
        llm_type, model_name = model_provider_model_name.split(":", 1)

        if llm_type.lower() == "openai":
            cfg = OpenAIConfig()
            cfg.model = model_name
            return cfg
        elif llm_type.lower() == "groq":
            cfg = GroqConfig()
            cfg.model = model_name
            return cfg
        else:
            raise ValueError(
                f"Unsupported LLM provider: '{llm_type}'. "
                f"Register a new handler with LLMHandlerRegistry.register(MyHandler) "
                f"to support custom providers."
            )