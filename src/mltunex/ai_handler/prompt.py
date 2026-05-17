"""
mltunex.ai_handler.prompt
──────────────────────────
Prompt templates for MLTuneX AI components.

Changes from original
─────────────────────
* HyperparameterResponsePrompt.get_hyperparameter_response_prompt() now
  delegates to ResponseSchemaRegistry so the format instructions are
  driven by the registered schema for the framework, not hard-coded here.
  The OptunaPrompt class attribute is kept for backward compatibility but
  is no longer the authoritative source — ResponseSchemaRegistry is.

* All original placeholders are preserved:
  {Data_Profile}, {Top_Models}, {ModelHyperparameter_Schema},
  {HyperparameterResponsePrompt}

* The system prompt body is tightened — removed emoji arrows that break
  Windows cp1252 terminals, kept all semantic instructions unchanged.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class HyperparameterResponsePrompt:
    """
    Collection of response-format prompts for different optimisation frameworks.

    ``get_hyperparameter_response_prompt`` is the canonical way to obtain the
    format instructions.  It delegates to ResponseSchemaRegistry so any
    registered framework is automatically supported.

    The ``OptunaPrompt`` class attribute is retained as a fallback for code
    that accesses it directly.
    """

    # Kept for direct-access backward compatibility
    OptunaPrompt: str = """
    - Return a JSON array, one object per model, structured for Optuna.
    - Each object must have "model_name" (string) and
      "suggested_hyperparameters" (object mapping param names to definitions).
    - Supported types: "int", "float", "categorical", "bool", "fixed".
    - For "int"/"float": provide "low" and "high" (low < high); optionally "step" and "log".
    - For "categorical": provide a non-empty "values" list.
    - For "fixed": provide a "value" key.
    - Do NOT use null or None as parameter values.
    - Output valid JSON only — no markdown fences, no prose, no comments.

    Example:
    [
      {
        "model_name": "RandomForestClassifier",
        "suggested_hyperparameters": {
          "n_estimators": {"type": "int",  "low": 100, "high": 500},
          "max_depth":    {"type": "categorical", "values": [3, 5, 7, 10]},
          "max_features": {"type": "categorical", "values": ["sqrt", "log2"]}
        }
      }
    ]
    """

    @staticmethod
    def get_hyperparameter_response_prompt(hyperparameter_framework: str) -> str:
        """
        Return the format-instructions prompt fragment for *hyperparameter_framework*.

        Delegates to ResponseSchemaRegistry so registered custom frameworks
        are automatically supported without modifying this file.

        Parameters
        ----------
        hyperparameter_framework : str
            Name of the tuning framework (e.g. ``"Optuna"``).

        Returns
        -------
        str
            The format instructions string injected as
            ``{HyperparameterResponsePrompt}`` into the system prompt.

        Raises
        ------
        ValueError
            If the framework is not registered.
        """
        # Lazy import to avoid circular dependency at module load time
        from mltunex.ai_handler.response_schema_registry import ResponseSchemaRegistry
        return ResponseSchemaRegistry.get(hyperparameter_framework).format_instructions()


@dataclass
class LLMPrompts:
    """
    System prompts for LLM interactions.

    All four placeholders are preserved exactly as in the original so the
    existing data-flow (orchestrator -> LLMManager -> handler -> chain.invoke)
    is unaffected.
    """

    OpenAIPrompt: str = """
    You are an expert machine learning engineer specialized in hyperparameter optimization.

    Below are:
    1. The metadata and statistical insights about the dataset,
    2. The top-performing models selected based on evaluation,
    3. The hyperparameters each model supports (with data types and value hints).

    Your task:
    - Suggest optimized hyperparameter ranges or values for each of the top models
      based on the dataset and its properties.
    - Use your understanding of the data (variance, correlation, distribution,
      skewness, etc.) to tailor the suggestions.
    - Only include hyperparameters that are meaningful for this dataset.
      Do not include every available hyperparameter.
    - Do not explain the hyperparameters or their significance.
    - Do not use null or None as a parameter value.
    - Provide the output in the format specified below.

    ---

    <DataProfile>
    {Data_Profile}
    </DataProfile>

    ---

    <TopModels>
    {Top_Models}
    </TopModels>

    ---

    <ModelHyperparameterSchema>
    {ModelHyperparameter_Schema}
    </ModelHyperparameterSchema>

    ---

    Instructions for Output:
    {HyperparameterResponsePrompt}
    """