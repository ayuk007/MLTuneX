"""
MLTuneX Main Interface module.

This module provides the high-level interface for the MLTuneX framework,
coordinating data ingestion, model training, metadata profiling, and
hyperparameter optimization workflows.

Examples:
    >>> tuner = MLTuneX(data=df, target_column="target", task_type="classification")
    >>> tuner.run(
    ...     result_csv_path="results/",
    ...     model_dir_path="models/",
    ...     tune_models="yes"
    ... )
"""

import os
import pandas as pd
from typing import Union, Tuple, Dict, Any
from mltunex.data.ingestion import Data_Ingestion
from mltunex.trainer.trainer import ModelTrainer
from mltunex.hyperparam_tuner.hyperparameter_tuning_orchestration import HyperparameterTuningOrchestration
from mltunex.ai_handler.metadata_profiler import MetaDataProfiler
from mltunex.utils.model_utils import ModelUtils


class MLTuneX:
    """
    Main interface for MLTuneX automated machine learning framework.

    This class coordinates all major components of the MLTuneX framework:
    - Data ingestion and preprocessing
    - Model training and evaluation
    - Metadata profiling for AI-powered optimization
    - Hyperparameter tuning with AI suggestions

    Parameters
    ----------
    data : Union[Tuple, str, pd.DataFrame]
        Input data as DataFrame, file path, or train-test split tuple.
    target_column : str
        Name of the target variable column.
    task_type : str
        Type of ML task ('classification' or 'regression').
    hyperparameter_framework : str, optional (default="Optuna")
        Framework to use for hyperparameter optimization.
    models_library : str, optional (default="sklearn")
        Machine learning library to use for models.

    Attributes
    ----------
    x_train : pd.DataFrame
        Training features.
    x_test : pd.DataFrame
        Testing features.
    y_train : pd.Series
        Training labels.
    y_test : pd.Series
        Testing labels.
    trainer : ModelTrainer
        Model training orchestrator.
    metadata_profiler : MetaDataProfiler
        Dataset analysis component.
    tuner : HyperparameterTuningOrchestration
        Hyperparameter optimization component.
    """

    def __init__(self, data: Union[Tuple, str, pd.DataFrame], target_column: str, 
                 task_type: str, model_provider_model_name: str, hyperparameter_framework: str = "Optuna", 
                 models_library: str = "sklearn"):
        """Initialize MLTuneX with data and configuration."""
        # Store configuration
        self.models_library = models_library
        self.hyperparameter_framework = hyperparameter_framework
        
        # Ingest and split data
        self.x_train, self.x_test, self.y_train, self.y_test = Data_Ingestion().ingest_data(
            source=data, target_column=target_column
        )
        self.task_type = task_type
        
        # Initialize trainer component
        self.trainer = ModelTrainer(
            models_library=self.models_library,
            cross_validation_strategy="kfold",
            task_type=self.task_type,
            train_parallelization=False
        )
        
        # Initialize metadata profiler
        self.metadata_profiler = MetaDataProfiler(
            data=(self.x_train, self.x_test, self.y_train, self.y_test),
            target_column=target_column,
            task_type=task_type,
            is_preprocessed=False
        )
        
        # Initialize tuning orchestrator
        self.tuner = HyperparameterTuningOrchestration(
            hyperparameter_framework=self.hyperparameter_framework,
            models_library=self.models_library,
            task_type = self.task_type,
            training_results=None,  # Will be set after training
            model_provider_model_name = model_provider_model_name

        )
    
    def run(self, result_csv_path: str, model_dir_path: str, 
            tune_models: str = "yes") -> None:
        """
        Execute complete MLTuneX workflow.

        This method coordinates the entire automated ML process:
        1. Train and evaluate baseline models
        2. Extract dataset metadata
        3. Generate AI-powered hyperparameter suggestions
        4. Optimize hyperparameters
        5. Train final model with best parameters
        6. Save results and model artifacts

        Parameters
        ----------
        result_csv_path : str
            Path to save evaluation results.
        model_dir_path : str
            Path to save trained models.
        tune_models : str, optional (default="yes")
            Whether to perform hyperparameter tuning.

        Examples
        --------
        >>> tuner.run("results/", "models/", tune_models="yes")
        """
        # Train initial models and evaluate
        training_results = self.trainer._run(
            X_train=self.x_train, 
            y_train=self.y_train,
            X_test=self.x_test, 
            y_test=self.y_test
        )
        
        # Extract dataset metadata for AI suggestions
        metadata = self.metadata_profiler.extract()
        
        # Save evaluation results
        evaluation_df = ModelUtils.save_results(
            evaluation_results=training_results[-1],
            evaluation_results_path=result_csv_path
        )
        
        # Skip tuning if not requested
        if tune_models != "yes":
            print("Skipping hyperparameter tuning.")
            return
            
        # Select top performing models
        top_models = ModelUtils.get_topK_models(
            results_csv=evaluation_df, 
            k=3,
            task_type = self.task_type
        )
        
        # Update tuner with training results
        self.tuner.hyperparameter_tuner.training_results = training_results
        
        # Run hyperparameter optimization
        best_model, best_params = self.tuner.tune(
            data_profile=metadata,
            top_models=top_models.to_json(),
            model_hyperparameter_schema=str(
                self.trainer.model_registry.get_all_hyperparameters(
                    top_models=top_models["Model"].tolist(),
                    models=training_results[0]
                )
            ),
            x_train=self.x_train,
            y_train=self.y_train
        )
        
        # Print optimization results
        print(f"Best Model: {best_model}, Best Hyperparameters: {best_params}")
        
        # Train final model with best parameters
        best_model_object = self.train_best_model(
            model_name=best_model,
            x_train=self.x_train,
            y_train=self.y_train,
            best_params=best_params
        )
        
        # Save the optimized model
        self.save_best_model(
            model_name=best_model,
            model=best_model_object,
            model_dir_path=model_dir_path
        )

    def train_best_model(self, model_name: str, x_train: pd.DataFrame, 
                        y_train: pd.Series, 
                        best_params: Dict[str, Any] = None) -> object:
        """
        Train model with optimized hyperparameters.

        Parameters
        ----------
        model_name : str
            Name of the model to train.
        x_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training labels.
        best_params : Dict[str, Any], optional
            Optimized hyperparameters.

        Returns
        -------
        object
            Trained model instance.

        Raises
        ------
        ValueError
            If model_name not found in registry.
        """
        # Get available models from registry
        model_tuple = self.trainer._load_models()
        model_dict = {model[0]: model[1] for model in model_tuple}
        
        # Validate model exists
        if model_name not in list(model_dict.keys()):
            raise ValueError(f"Model {model_name} not found in the registry.")
        
        # Train model with optimized parameters
        trained_model = self.trainer.library_trainer.train_model(
            model=model_dict[model_name],
            params=best_params,
            X_train=x_train,
            y_train=y_train,
            task_type=self.task_type
        )
        return trained_model

    def save_best_model(self, model_name: str, model: object, 
                       model_dir_path: str) -> None:
        """
        Save optimized model to disk.

        Parameters
        ----------
        model_name : str
            Name to use for saved model.
        model : object
            Trained model instance to save.
        model_dir_path : str
            Directory path to save model.
        """
        ModelUtils.save_model(
            model, 
            os.path.join(model_dir_path, f"{model_name}.joblib")
        )
        print(f"Best model '{model_name}' saved to {model_dir_path}")