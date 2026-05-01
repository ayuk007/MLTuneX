"""
Scikit-learn model registry implementation for MLTuneX.

This module provides a registry for scikit-learn models, including both classifiers
and regressors. It also supports additional models from XGBoost and LightGBM.
The registry handles model filtering, hyperparameter extraction, and model selection
based on task type.

Attributes:
    ModelType: Type alias for tuple of (model_name: str, model_class: Type)
"""

from os import stat
from sklearn.utils import all_estimators
from sklearn.base import ClassifierMixin, RegressorMixin
import xgboost
import lightgbm
from typing import List, Tuple, Type, Dict, Union

from mltunex.model_registry.base import BaseModelRegistry

# Type aliases for clarity
ModelType = Tuple[str, Type]

class SkLearn_Model_Registry(BaseModelRegistry):
    """
    Registry for Scikit-Learn models.

    This class manages the available scikit-learn models, providing methods to
    access classification and regression models, their hyperparameters, and
    filtering out models that are not suitable for direct use.

    Attributes
    ----------
    REMOVED_CLASSIFIERS : List[str]
        List of classifier names to exclude from the registry.
    REMOVED_REGRESSORS : List[str]
        List of regressor names to exclude from the registry.
    ADDITIONAL_CLASSIFIERS : List[ModelType]
        Additional classifiers from other libraries (XGBoost, LightGBM).
    ADDITIONAL_REGRESSORS : List[ModelType]
        Additional regressors from other libraries (XGBoost, LightGBM).
    """

    # Lists of models to exclude with explanatory comments
    REMOVED_CLASSIFIERS: List[str] = [
        # Meta-estimators and complex models that require special handling
        "ClassifierChain", "MultiOutputClassifier", "OneVsOneClassifier",
        "OneVsRestClassifier", "VotingClassifier",
        # Models with specific requirements or computational constraints
        "ComplementNB", "GradientBoostingClassifier", "GaussianProcessClassifier",
        "HistGradientBoostingClassifier", "MLPClassifier", "LogisticRegressionCV",
        "MultinomialNB", "OutputCodeClassifier", "RadiusNeighborsClassifier", 
        "FixedThresholdClassifier", "NuSVC",
        # Semi-supervised and specialized models
        "LabelPropagation", "LabelSpreading", "SelfTrainingClassifier", 
        "StackingClassifier", "TunedThresholdClassifierCV",
    ]

    REMOVED_REGRESSORS: List[str] = [
        # Meta-estimators and ensemble methods
        "StackingRegressor", "MultiOutputRegressor", "RegressorChain", "VotingRegressor",
        # Specialized regression models
        "TheilSenRegressor", "ARDRegression", "CCA", "IsotonicRegression",
        # Multi-task learning models
        "MultiTaskElasticNet", "MultiTaskElasticNetCV", "MultiTaskLasso", 
        "MultiTaskLassoCV",
        # Partial least squares models
        "PLSCanonical", "PLSRegression", "RadiusNeighborsRegressor",
    ]

    # Additional model configurations from other libraries
    ADDITIONAL_CLASSIFIERS: List[ModelType] = [
        ("XGBClassifier", xgboost.XGBClassifier),
        ("LGBMClassifier", lightgbm.LGBMClassifier),
    ]

    ADDITIONAL_REGRESSORS: List[ModelType] = [
        ("XGBRegressor", xgboost.XGBRegressor),
        ("LGBMRegressor", lightgbm.LGBMRegressor),
    ]

    @staticmethod
    def get_classification_models() -> List[ModelType]:
        """
        Get classification models from scikit-learn, excluding the removed ones.

        Returns
        -------
        List[ModelType]
            List of tuples containing (model_name, model_class) for classifiers
            from scikit-learn and additional sources.
        """
        # Get all estimators and filter out the removed classifiers
        models = [
            model for model in all_estimators()
            if issubclass(model[1], ClassifierMixin)
            and model[0] not in SkLearn_Model_Registry.REMOVED_CLASSIFIERS
        ]

        # Add additional classifiers (like XGBoost, LightGBM, etc.)
        models.extend(SkLearn_Model_Registry.ADDITIONAL_CLASSIFIERS)
        return models

    @staticmethod
    def get_regression_models() -> List[ModelType]:
        """
        Get regression models from scikit-learn, excluding the removed ones.

        Returns
        -------
        List[ModelType]
            List of tuples containing (model_name, model_class) for regressors
            from scikit-learn and additional sources.
        """
        # Get all estimators and filter out the removed regressors
        models = [
            model for model in all_estimators()
            if issubclass(model[1], RegressorMixin)
            and model[0] not in SkLearn_Model_Registry.REMOVED_REGRESSORS
        ]

        # Add additional regressors (like XGBoost, LightGBM, etc.)
        models.extend(SkLearn_Model_Registry.ADDITIONAL_REGRESSORS)
        return models
    
    @staticmethod
    def get_hyperparameters(model_name: str, model: object) -> List:
        """
        Get hyperparameters for a given model.

        Parameters
        ----------
        model_name : str
            Name of the model to get hyperparameters for.
        model : object
            Model instance to extract hyperparameters from.

        Returns
        -------
        List
            List of parameter names for the model.

        Raises
        ------
        ValueError
            If the model doesn't support parameter extraction.
        """
        if hasattr(model, "_get_param_names"):
            params = model._get_param_names()
            return params
        else:
            raise ValueError(f"Model {model_name} does not have _get_param_names method.")
        
    @staticmethod
    def get_all_hyperparameters(
        top_models: List[str], 
        models: Dict[str, ModelType]
    ) -> Dict[str, Dict[str, Union[str, List[str]]]]:
        """
        Get hyperparameters for a list of models.

        Parameters
        ----------
        top_models : List[str]
            List of model names to get hyperparameters for.
        models : Dict[str, ModelType]
            Dictionary mapping model names to their implementations.

        Returns
        -------
        Dict[str, Dict[str, Union[str, List[str]]]]
            Dictionary mapping model names to their hyperparameters.
        """
        hyperparameters = {}
        for model_name in top_models:
            hyperparameters[model_name] = SkLearn_Model_Registry.get_hyperparameters(
                model_name, 
                models[model_name][1]
            )
        return hyperparameters

    @staticmethod
    def get_models(task_type: str) -> List[ModelType]:
        """
        Get models based on the specified task type.

        Parameters
        ----------
        task_type : str
            Type of machine learning task ('classification' or 'regression').

        Returns
        -------
        List[ModelType]
            List of models appropriate for the specified task type.
        """
        if task_type == "classification":
            return SkLearn_Model_Registry.get_classification_models()
        elif task_type == "regression":
            return SkLearn_Model_Registry.get_regression_models()
        else:
            raise ValueError(f"Unsupported task type: {task_type}")