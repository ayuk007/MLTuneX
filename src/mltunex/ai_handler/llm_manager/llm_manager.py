from mltunex.ai_handler.llm_manager.openai_handler import OpenAIHyperparamGenerator
from mltunex.ai_handler.llm_manager.groq_handler import GroqHyperparamGenerator
from mltunex.config.llm_config import OpenAIConfig, GroqConfig, LLMConfig
from typing import Optional

class LLMManager:

    @staticmethod
    def get_llm_instance(model_provider_model_name: str):
        """
        Get the LLM instance based on the specified type and model name.

        Parameters
        ----------
        llm_type_model_name : str
            The type and model name of the LLM, formatted as "LLMType:ModelName".

        Returns
        -------
        OpenAIHyperparamGenerator | GroqHyperparamGenerator
            An instance of the specified LLM.
        """
        llm_config = LLMConfig.get_llm_config(model_provider_model_name)
        
        if isinstance(llm_config, OpenAIConfig):
            return OpenAIHyperparamGenerator(config=llm_config)
        elif isinstance(llm_config, GroqConfig):
            return GroqHyperparamGenerator(config=llm_config)
        else:
            raise ValueError(f"Unsupported LLM configuration: {llm_config}")