from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from mltunex.config.llm_config import GroqConfig
from mltunex.ai_handler.prompt import HyperparameterResponsePrompt

from json_repair import repair_json
import json
import logging

load_dotenv()

logger = logging.getLogger(__name__)


class GroqHyperparamGenerator:
    def __init__(self, hyperparameter_framework: str = "Optuna", config: GroqConfig = GroqConfig) -> None:
        self.hyperparameter_framework = hyperparameter_framework
        self.llm = ChatGroq(model=config.model, temperature=config.temperature)
        self.output_parser = JsonOutputParser()
        self.prompt_template = PromptTemplate(
            template=config.SYSTEM_PROMPT,
            input_variables=["Data_Profile", "Top_Models", "ModelHyperparameter_Schema", "HyperparameterResponsePrompt"]
        )
        # Fix 1: Groq chain was missing the output parser; OpenAI had it, Groq didn't.
        # We intentionally keep the parser OFF the chain here so we can apply
        # response_formatter (which strips <think> tags first) before parsing.
        self.chain = self.prompt_template | self.llm

    def generate_response(self, data_profile: str, top_models: str, model_hyperparameter_schema: str) -> list:
        raw = self.chain.invoke(
            {
                "Data_Profile": data_profile,
                "Top_Models": top_models,
                "ModelHyperparameter_Schema": model_hyperparameter_schema,
                "HyperparameterResponsePrompt": HyperparameterResponsePrompt.get_hyperparameter_response_prompt(
                    self.hyperparameter_framework
                ),
            }
        )
        return self.response_formatter(response=raw.content)  # type: ignore

    def response_formatter(self, response: str) -> list:
        """
        Strip <think> reasoning blocks, repair and parse the JSON, then
        normalise the result to a guaranteed non-empty list.

        Raises
        ------
        ValueError
            If no valid JSON array can be extracted from the response.
        """
        # ── Step 1: strip chain-of-thought reasoning block ────────────
        if "</think>" in response:
            response = response.split("</think>")[-1]

        # ── Step 2: extract the JSON array even if wrapped in markdown ─
        # LLMs often wrap output in ```json ... ``` fences
        response = response.strip()
        if "```" in response:
            # Take the content between the first and last fence
            parts = response.split("```")
            # parts[1] is the fenced block; strip a leading "json" language tag
            fenced = parts[1].strip()
            if fenced.lower().startswith("json"):
                fenced = fenced[4:].strip()
            response = fenced

        # ── Step 3: repair and parse ───────────────────────────────────
        try:
            repaired = repair_json(response)
            # repair_json returns a string; parse it into a Python object
            parsed = json.loads(repaired) if isinstance(repaired, str) else repaired
        except Exception as exc:
            raise ValueError(
                f"GroqHyperparamGenerator: JSON repair/parse failed.\n"
                f"Raw (post-think strip): {response[:400]}\n"
                f"Error: {exc}"
            ) from exc

        # ── Step 4: normalise to a list ────────────────────────────────
        # LLMs sometimes return {"models": [...]} or {"hyperparameters": [...]}
        # instead of a bare list.
        if isinstance(parsed, dict):
            # Find the first value that is a non-empty list
            for v in parsed.values():
                if isinstance(v, list) and len(v) > 0:
                    parsed = v
                    break
            else:
                raise ValueError(
                    f"GroqHyperparamGenerator: LLM returned a dict but no list "
                    f"value was found inside it. Keys: {list(parsed.keys())}\n"
                    f"Raw response (truncated): {response[:400]}"
                )

        if not isinstance(parsed, list):
            raise ValueError(
                f"GroqHyperparamGenerator: expected a JSON array, got "
                f"{type(parsed).__name__}.\nRaw (truncated): {response[:400]}"
            )

        if len(parsed) == 0:
            raise ValueError(
                "GroqHyperparamGenerator: LLM returned an empty JSON array. "
                "This usually means the model produced no hyperparameter suggestions. "
                "Check the prompt template and model response."
            )

        # ── Step 5: validate each entry has required keys ─────────────
        valid = []
        for item in parsed:
            if not isinstance(item, dict):
                logger.warning("Skipping non-dict item in LLM response: %r", item)
                continue
            if "model_name" not in item:
                logger.warning("Skipping item missing 'model_name': %r", item)
                continue
            if "suggested_hyperparameters" not in item:
                item["suggested_hyperparameters"] = {}
            valid.append(item)

        if not valid:
            raise ValueError(
                "GroqHyperparamGenerator: LLM response parsed successfully but "
                "no valid model entries were found. Each entry must have at least "
                "'model_name' and 'suggested_hyperparameters'.\n"
                f"Parsed list (first item): {parsed[0] if parsed else 'empty'}"
            )

        return valid