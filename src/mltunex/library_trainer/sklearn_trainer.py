"""
Scikit-learn model trainer implementation for MLTuneX.

This module provides concrete implementation for training scikit-learn models.
It handles model initialization, parallel processing configuration, and
error handling during the training process.

Examples:
    >>> trainer = SklearnTrainer()
    >>> trained_model = trainer.train_model(RandomForestClassifier, X_train, y_train)
"""

import pandas as pd
from sklearn.base import BaseEstimator
from mltunex.library_trainer.base import BaseLibraryTrainer


class SklearnTrainer(BaseLibraryTrainer):
    """
    Trainer class for Scikit-Learn models.

    This class provides functionality to train scikit-learn models with
    automatic parallel processing configuration where supported.

    Methods
    -------
    train_model(model, X_train, y_train, task_type) -> BaseEstimator
        Train a scikit-learn model with the given data.
    """
    
    def __init__(self):
        """Initialize the SklearnTrainer."""
        pass

    def train_model(self, model: BaseEstimator, X_train: pd.DataFrame, 
                   y_train: pd.Series, task_type: str = None) -> BaseEstimator:
        """
        Train a scikit-learn model with the given training data.

        This method handles model initialization, configures parallel processing
        if supported by the model, and performs the training process.

        Parameters
        ----------
        model : BaseEstimator
            The scikit-learn model class to train.
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training labels.
        task_type : str, optional
            Type of machine learning task ('classification' or 'regression').

        Returns
        -------
        BaseEstimator
            Trained model instance if successful, None if training fails.

        Raises
        ------
        Exception
            If any error occurs during model training.

        Notes
        -----
        Models supporting parallel processing will be configured to use
        all available CPU cores (n_jobs=-1).
        """
        try:
            # Initialize the model instance
            model_instance = model()

            # Configure parallel processing if supported
            if hasattr(model_instance, "n_jobs"):
                model_instance.set_params(n_jobs=-1)
            
            # Train the model with provided data
            model_instance.fit(X_train, y_train)

            return model_instance
        except Exception as e:
            print(f"Error training model {model}: {e}")
            return None