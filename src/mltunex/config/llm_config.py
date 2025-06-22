from dataclasses import dataclass
from mltunex.ai_handler.prompt import LLMPrompts
from typing import Literal

@dataclass
class OpenAIConfig:
    model: str = Literal["gpt-4o"] #type: ignore
    temperature: float = 0
    SYSTEM_PROMPT: str = LLMPrompts.OpenAIPrompt

    def __post_init__(self):
        """
        Post-initialization to ensure the model is set correctly.
        """
        if self.model not in ["gpt-4o"]:
            raise ValueError(f"Unsupported OpenAI model: {self.model}. Supported models: ['gpt-4o']")

@dataclass
class GroqConfig:
    model: str = Literal["deepseek-r1-distill-llama-70b", "qwen/qwen3-32b"] # type: ignore  # This should be set to the Groq model name, e.g., "groq-1" 
    temperature: float = 0
    SYSTEM_PROMPT: str = LLMPrompts.OpenAIPrompt

    def __post_init__(self):
        """
        Post-initialization to ensure the model is set correctly.
        """
        if self.model not in ["deepseek-r1-distill-llama-70b", "qwen/qwen3-32b"]:
            raise ValueError(f"Unsupported Groq model: {self.model}. Supported models: ['deepseek-r1-distill-llama-70b', 'qwen/qwen3-32b']")

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
            llm = OpenAIConfig()
            llm.model = model_name
            return llm
        elif llm_type.lower() == "groq":
            llm = GroqConfig()
            llm.model = model_name
            return llm
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")