"""
Prompt templates module for MLTuneX's AI components.

This module provides structured prompt templates for different hyperparameter
optimization frameworks and LLM interactions. It ensures consistent prompt
formatting and response structures across the framework.

Examples:
    >>> prompt = HyperparameterResponsePrompt.get_hyperparameter_response_prompt("Optuna")
    >>> system_prompt = LLMPrompts.OpenAIPrompt
"""

from dataclasses import dataclass


@dataclass
class HyperparameterResponsePrompt:
    """
    Collection of response format prompts for different optimization frameworks.

    This class provides template strings that specify the expected response
    format for different hyperparameter optimization frameworks. Currently
    supports Optuna with extensibility for other frameworks.

    Attributes
    ----------
    OptunaPrompt : str
        Template string for Optuna-compatible hyperparameter suggestions.
    """

    OptunaPrompt: str = """
    - Use this format for each model:
    - Return a list of dictionaries, one per model, with hyperparameters structured for use with Optuna:
    [
    {{
        "model_name": "RandomForestClassifier",
        "suggested_hyperparameters": {{
        "n_estimators": {{
            "type": "int",
            "low": 100,
            "high": 300,
            "step": 100
        }},
        "max_depth": {{
            "type": "int",
            "values": [null, 10, 20]
        }},
        "min_samples_split": {{
            "type": "int",
            "values": [2, 5]
        }},
        "max_features": {{
            "type": "categorical",
            "values": ["sqrt", "log2"]
        }}
        }}
    }},
    ...
    ]
    """    

    @staticmethod
    def get_hyperparameter_response_prompt(hyperparameter_framework: str) -> str:
        """
        Get the response format prompt for specified framework.

        Parameters
        ----------
        hyperparameter_framework : str
            Name of the hyperparameter optimization framework.
            Currently supported: ["Optuna"]

        Returns
        -------
        str
            Template string for the specified framework.

        Raises
        ------
        ValueError
            If the specified framework is not supported.

        Examples
        --------
        >>> prompt = HyperparameterResponsePrompt.get_hyperparameter_response_prompt("Optuna")
        """
        if hyperparameter_framework == "Optuna":
            return HyperparameterResponsePrompt.OptunaPrompt
        else:
            raise ValueError(f"Unsupported hyperparameter framework: {hyperparameter_framework}")
        

@dataclass
class LLMPrompts:
    """
    Collection of system prompts for different LLM interactions.

    This class provides carefully crafted system prompts for different
    LLM-based tasks in the framework. The prompts include context setup,
    task description, and expected output format.

    Attributes
    ----------
    OpenAIPrompt : str
        System prompt for OpenAI models with hyperparameter optimization focus.
    """

    OpenAIPrompt: str = """
    You are an expert machine learning engineer specialized in hyperparameter optimization.

    Below are:
    1. The metadata and statistical insights about the dataset,
    2. The top-performing models selected based on evaluation,
    3. The hyperparameters each model supports (with data types and value hints).

    Your task:
    ➡️ Suggest optimized hyperparameter **ranges** or **values** for each of the top models based on the dataset and its properties.  
    ➡️ Use your understanding of the data (variance, correlation, distribution, skewness, etc.) to tailor the suggestions.  
    ➡️ Only use **hyperparameters** that are suitable no need to use all of them if not required.
    ➡️ No need to explain the hyperparameters or their significance.
    ➡️ Provide the output in the specified format below.
    ➡️ You should not give any parameter values as null or None. If you don't know the value, don't include it in the output.

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
    ✅ **Instructions for Output**:
    {HyperparameterResponsePrompt}
    """