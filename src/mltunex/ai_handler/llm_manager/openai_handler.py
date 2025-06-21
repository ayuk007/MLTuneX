"""
OpenAI-powered hyperparameter generator module for MLTuneX.

This module provides functionality to generate optimized hyperparameter search
spaces using OpenAI's language models. It supports different hyperparameter
optimization frameworks and integrates with LangChain for prompt engineering
and response parsing.

Examples:
    >>> generator = OpenAIHyperparamGenerator()
    >>> response = generator.generate_response(
    ...     data_profile="Binary classification dataset with 1000 samples",
    ...     top_models="RandomForestClassifier",
    ...     model_hyperparameter_schema="n_estimators: int, max_depth: int"
    ... )
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from mltunex.config.llm_config import OpenAIConfig
from mltunex.ai_handler.prompt import HyperparameterResponsePrompt

# Load environment variables for OpenAI configuration
load_dotenv()


class OpenAIHyperparamGenerator:
    """
    Generator for AI-powered hyperparameter optimization suggestions.

    This class uses OpenAI's language models to generate optimized
    hyperparameter search spaces based on data characteristics and
    model types. It supports different hyperparameter optimization
    frameworks through configurable prompts.

    Parameters
    ----------
    hyperparameter_framework : str, optional (default="Optuna")
        Name of the hyperparameter optimization framework to use.
    config : OpenAIConfig, optional (default=OpenAIConfig)
        Configuration for OpenAI API settings.

    Attributes
    ----------
    hyperparameter_framework : str
        Name of the active hyperparameter optimization framework.
    llm : ChatOpenAI
        LangChain's OpenAI chat model interface.
    output_parser : JsonOutputParser
        Parser for converting LLM responses to JSON format.
    prompt_template : PromptTemplate
        Template for generating structured prompts.
    chain : Chain
        LangChain processing chain combining prompt, LLM, and parser.
    """

    def __init__(self, hyperparameter_framework: str = "Optuna", 
                 config: OpenAIConfig = OpenAIConfig) -> None:
        """
        Initialize the hyperparameter generator with specified framework and config.

        Parameters
        ----------
        hyperparameter_framework : str, optional (default="Optuna")
            Name of the hyperparameter optimization framework.
        config : OpenAIConfig, optional (default=OpenAIConfig)
            OpenAI configuration settings.
        """
        # Store the hyperparameter framework choice
        self.hyperparameter_framework = hyperparameter_framework
        
        # Initialize LangChain components
        self.llm = ChatOpenAI(model=config.model, temperature=config.temperature)
        self.output_parser = JsonOutputParser()
        
        # Create prompt template with required variables
        self.prompt_template = PromptTemplate(
            template=config.SYSTEM_PROMPT,
            input_variables=["Data_Profile", "Top_Models", 
                           "ModelHyperparameter_Schema", 
                           "HyperparameterResponsePrompt"]
        )
        
        # Build the LangChain processing chain
        self.chain = self.prompt_template | self.llm | self.output_parser

    def generate_response(self, data_profile: str, top_models: str, 
                         model_hyperparameter_schema: str) -> str:
        """
        Generate hyperparameter optimization suggestions.

        This method takes information about the dataset, models, and their
        hyperparameters to generate optimized search spaces using LLM.

        Parameters
        ----------
        data_profile : str
            Description of the dataset characteristics.
        top_models : str
            Names of the models to optimize.
        model_hyperparameter_schema : str
            Schema of available hyperparameters for the models.

        Returns
        -------
        str
            JSON-formatted string containing hyperparameter optimization
            suggestions.

        Examples
        --------
        >>> response = generator.generate_response(
        ...     "Binary classification, 1000 samples, 10 features",
        ...     "RandomForestClassifier",
        ...     "n_estimators: int, max_depth: int"
        ... )
        """
        # Invoke the LangChain processing chain with inputs
        response = self.chain.invoke({
            "Data_Profile": data_profile,
            "Top_Models": top_models,
            "ModelHyperparameter_Schema": model_hyperparameter_schema,
            "HyperparameterResponsePrompt": 
                HyperparameterResponsePrompt.get_hyperparameter_response_prompt(
                    self.hyperparameter_framework
                )
        })
        return response