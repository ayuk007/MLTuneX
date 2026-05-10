"""Central factory for LLM handler instances."""
from __future__ import annotations
from mltunex.ai_handler.llm_handler_base import BaseLLMHandler, LLMHandlerConfig, LLMHandlerRegistry
from mltunex.ai_handler.prompt import LLMPrompts
from mltunex.ai_handler.llm_manager.groq_handler import GroqHyperparamGenerator
from mltunex.ai_handler.llm_manager.openai_handler import OpenAIHyperparamGenerator

LLMHandlerRegistry.register(GroqHyperparamGenerator)
LLMHandlerRegistry.register(OpenAIHyperparamGenerator)


class LLMManager:
    @staticmethod
    def get_llm_instance(
        model_provider_model_name: str,
        framework: str = "Optuna",
    ) -> BaseLLMHandler:
        if ":" not in model_provider_model_name:
            raise ValueError(
                f"model_provider_model_name must be 'Provider:ModelName', "
                f"got '{model_provider_model_name}'."
            )
        provider, model_name = model_provider_model_name.split(":", 1)
        config = LLMHandlerConfig(
            model_name=model_name,
            temperature=0.0,
            system_prompt=LLMPrompts.OpenAIPrompt,
            framework=framework,
        )
        key = provider.lower()
        if key not in LLMHandlerRegistry._registry:
            raise ValueError(
                f"No LLM handler registered for provider '{provider}'. "
                f"Registered: {LLMHandlerRegistry.list_providers()}."
            )
        return LLMHandlerRegistry._registry[key](config=config)