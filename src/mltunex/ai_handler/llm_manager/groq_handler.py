"""Groq concrete LLM handler — extends BaseLLMHandler."""
from __future__ import annotations
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from mltunex.ai_handler.llm_handler_base import BaseLLMHandler, LLMHandlerConfig

load_dotenv()


class GroqHyperparamGenerator(BaseLLMHandler):
    provider_name = "Groq"

    def __init__(self, config: LLMHandlerConfig, **_kw) -> None:
        super().__init__(config)
        self._llm = ChatGroq(model=config.model_name, temperature=config.temperature)
        self._prompt = PromptTemplate(
            template=config.system_prompt,
            input_variables=["Data_Profile", "Top_Models",
                             "ModelHyperparameter_Schema", "HyperparameterResponsePrompt"],
        )

    # Return the full AIMessage so the base can extract token metadata
    def _call_llm_with_obj(self, prompt_vars: dict):
        response = (self._prompt | self._llm).invoke(prompt_vars)
        return response.content, response  # type: ignore

    # Fallback for the abstract contract (used by subclasses that skip _with_obj)
    def _call_llm(self, prompt_vars: dict) -> str:
        return (self._prompt | self._llm).invoke(prompt_vars).content  # type: ignore

    def generate_response(self, data_profile, top_models, schema):
        return self.suggest_search_spaces(data_profile, top_models, schema)