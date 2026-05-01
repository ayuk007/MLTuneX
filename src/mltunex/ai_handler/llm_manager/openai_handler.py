from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from mltunex.config.llm_config import OpenAIConfig
from mltunex.ai_handler.prompt import HyperparameterResponsePrompt
import json
import logging

load_dotenv()

logger = logging.getLogger(__name__)


class OpenAIHyperparamGenerator:
    """Generator for AI-powered hyperparameter optimization suggestions via OpenAI."""

    def __init__(self, hyperparameter_framework: str = "Optuna",
                 config: OpenAIConfig = OpenAIConfig) -> None:
        self.hyperparameter_framework = hyperparameter_framework
        self.llm = ChatOpenAI(model=config.model, temperature=config.temperature)
        self.output_parser = JsonOutputParser()
        self.prompt_template = PromptTemplate(
            template=config.SYSTEM_PROMPT,
            input_variables=["Data_Profile", "Top_Models",
                             "ModelHyperparameter_Schema",
                             "HyperparameterResponsePrompt"]
        )
        # OpenAI chain pipes through the output parser directly
        self.chain = self.prompt_template | self.llm | self.output_parser

    def generate_response(self, data_profile: str, top_models: str,
                          model_hyperparameter_schema: str) -> list:
        parsed = self.chain.invoke({
            "Data_Profile": data_profile,
            "Top_Models": top_models,
            "ModelHyperparameter_Schema": model_hyperparameter_schema,
            "HyperparameterResponsePrompt":
                HyperparameterResponsePrompt.get_hyperparameter_response_prompt(
                    self.hyperparameter_framework
                ),
        })
        return self._normalise(parsed)

    @staticmethod
    def _normalise(parsed) -> list:
        """Normalise LLM output to a guaranteed non-empty list of model dicts."""
        # Handle dict-wrapped responses e.g. {"models": [...]}
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list) and len(v) > 0:
                    parsed = v
                    break
            else:
                raise ValueError(
                    f"OpenAIHyperparamGenerator: expected a JSON array, got a dict "
                    f"with no list value. Keys: {list(parsed.keys())}"
                )

        if not isinstance(parsed, list):
            raise ValueError(
                f"OpenAIHyperparamGenerator: expected a JSON array, got "
                f"{type(parsed).__name__}."
            )

        if len(parsed) == 0:
            raise ValueError(
                "OpenAIHyperparamGenerator: LLM returned an empty JSON array."
            )

        # Validate entries
        valid = []
        for item in parsed:
            if not isinstance(item, dict) or "model_name" not in item:
                logger.warning("Skipping invalid item: %r", item)
                continue
            if "suggested_hyperparameters" not in item:
                item["suggested_hyperparameters"] = {}
            valid.append(item)

        if not valid:
            raise ValueError(
                "OpenAIHyperparamGenerator: no valid model entries found in "
                "LLM response. Each entry needs at least 'model_name'."
            )
        return valid