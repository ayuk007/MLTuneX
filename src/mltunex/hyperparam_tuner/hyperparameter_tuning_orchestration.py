"""
Hyperparameter Tuning Orchestration module for MLTuneX.

This module provides high-level orchestration of hyperparameter optimization
workflows, combining AI-powered suggestion generation with concrete tuning
implementations. It coordinates between LLM-based parameter suggestions and
optimization frameworks like Optuna.

Examples:
    >>> orchestrator = HyperparameterTuningOrchestration()
    >>> best_model, params = orchestrator.tune(
    ...     data_profile="Binary classification dataset",
    ...     top_models="RandomForestClassifier",
    ...     model_hyperparameter_schema="n_estimators: int",
    ...     x_train=train_data,
    ...     y_train=train_labels
    ... )
"""

import pandas as pd
from typing import Union, Tuple, Dict, Any, List
from mltunex.model_registry.model_registry import Model_Registry
from mltunex.hyperparam_tuner.optuna_tuner import OptunaHyperparameterTuner
from mltunex.ai_handler.llm_manager.llm_manager import LLMManager



class HyperparameterTuningOrchestration:
    """
    Orchestrator for AI-powered hyperparameter optimization workflows.

    This class coordinates the interaction between AI-based hyperparameter
    suggestion generation and concrete optimization implementations. It manages
    the workflow of getting suggestions from LLMs and applying them through
    frameworks like Optuna.

    Parameters
    ----------
    hyperparameter_framework : str, optional (default="Optuna")
        Name of the hyperparameter optimization framework to use.
    models_library : str, optional (default="sklearn")
        Name of the machine learning library being used.
    training_results : List, optional (default=None)
        Previous training results for initialization.

    Attributes
    ----------
    hyperparameter_framework : str
        Active hyperparameter optimization framework.
    models_library : str
        Active machine learning library.
    model_registry : BaseModelRegistry
        Registry for accessing machine learning models.
    llm_chain : OpenAIHyperparamGenerator
        Interface for AI-powered parameter suggestion.
    hyperparameter_tuner : BaseHyperparameterTuner
        Concrete implementation of parameter optimization.
    """

    def __init__(self, task_type: str, hyperparameter_framework: str = "Optuna", 
                 models_library: str = "sklearn", 
                 training_results: List = None, model_provider_model_name: str = "Groq:qwen/qwen3-32b") -> None:
        """Initialize the hyperparameter tuning orchestration."""
        self.hyperparameter_framework = hyperparameter_framework
        self.models_library = models_library
        self.tuner = None
        
        # Initialize components
        self.model_registry = Model_Registry.get_model_registry(
            models_library=models_library
        )
        self.llm_chain = LLMManager.get_llm_instance(model_provider_model_name=model_provider_model_name)
        
        # Initialize appropriate tuner based on framework
        self.hyperparameter_tuner = (
            OptunaHyperparameterTuner(
                training_results=training_results,
                task_type = task_type
            ) if hyperparameter_framework == "Optuna" else None
        )
    
    def tune(self, data_profile: str, top_models: str, 
             model_hyperparameter_schema: str, x_train: pd.DataFrame, 
             y_train: pd.Series) -> List[Dict[str, Any]]:
        """
        Execute complete hyperparameter optimization workflow.

        This method coordinates the full optimization process:
        1. Generate AI-powered parameter suggestions
        2. Convert suggestions to search space configuration
        3. Execute concrete optimization using selected framework

        Parameters
        ----------
        data_profile : str
            Description of dataset characteristics.
        top_models : str
            JSON string of top performing models.
        model_hyperparameter_schema : str
            Schema of model hyperparameters.
        x_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training labels.

        Returns
        -------
        Tuple[str, Dict[str, Any]]
            Tuple containing:
            - Name of best performing model
            - Dictionary of optimal hyperparameters

        Examples
        --------
        >>> best_model, params = orchestrator.tune(
        ...     data_profile="Binary classification, 1000 samples",
        ...     top_models="{'model': 'RandomForest'}",
        ...     model_hyperparameter_schema="n_estimators: int",
        ...     x_train=X_train,
        ...     y_train=y_train
        ... )
        """
        # Generate hyperparameter suggestions using LLM
        response = self.llm_chain.generate_response(
            data_profile=data_profile,
            top_models=top_models,
            model_hyperparameter_schema=model_hyperparameter_schema
        )
        
        # Parse the response to get the hyperparameter search spaces
        model_search_spaces = response
        
        # Run Optuna for hyperparameter tuning
        best_model, best_params = self.hyperparameter_tuner.run_optuna(
            model_search_spaces=model_search_spaces,
            x_train=x_train,
            y_train=y_train
        )
        
        return best_model, best_params