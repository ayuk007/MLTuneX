from dataclasses import dataclass
from mltunex.ai_handler.prompt import LLMPrompts
from typing import Literal

@dataclass
class OpenAIConfig:
    model: str = Literal["gpt-4o"] #type: ignore
    temperature: float = 0
    SYSTEM_PROMPT: str = LLMPrompts.OpenAIPrompt

@dataclass
class GroqConfig:
    model: str = Literal["deepseek-r1-distill-llama-70b", "qwen/qwen3-32b"] # type: ignore  # This should be set to the Groq model name, e.g., "groq-1" 
    temperature: float = 0
    SYSTEM_PROMPT: str = LLMPrompts.OpenAIPrompt

@dataclass
class LLMConfig:

    @staticmethod
    def get_llm_config(model_provider_model_name: str):
        """
        Get the configuration for the specified LLM type.

        Parameters
        ----------
        llm_type : str
            The type of LLM to configure ("OpenAI" or "Groq").

        Returns
        -------
        OpenAIConfig | GroqConfig
            The configuration object for the specified LLM.
        """
        llm_type, model_name = model_provider_model_name.split(":")

        if llm_type.lower() == "openai":
            return OpenAIConfig(model = model_name)
        elif llm_type.lower() == "groq":
            return GroqConfig(model = model_name)
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")