"""
Model iteration strategy module for MLTuneX.

This module defines strategies for iterating over machine learning models
during training. It provides different iteration approaches based on the
machine learning library being used, with support for parallel processing
where available.

Examples:
    >>> strategy = IterationStrategy.get_iteration_strategy("sklearn")
    >>> models, results = strategy(train_fn, eval_fn, model_list, X, y)
"""

from typing import Callable, List, Tuple, Dict, Any


class IterationStrategy:
    """
    Defines iteration strategies for different machine learning libraries.

    This class provides static methods that implement different approaches
    to iterate over and train multiple models. Each strategy is optimized
    for specific machine learning libraries.

    Methods
    -------
    iterate_sklearn_models
        Strategy for training scikit-learn models sequentially.
    get_iteration_strategy
        Factory method to get appropriate iteration strategy.
    """

    @staticmethod
    def iterate_sklearn_models(
        train_function: Callable, 
        evaluate_function: Callable,
        models: List[Tuple[str, object]], 
        x_train: Any,
        y_train: Any
    ) -> Tuple[Dict, List]:
        """
        Iterate over Scikit-Learn models for training and evaluation.

        This method implements sequential training of scikit-learn models,
        collecting both trained models and their evaluation results.

        Parameters
        ----------
        train_function : Callable
            Function to train individual models.
        evaluate_function : Callable
            Function to evaluate trained models.
        models : List[Tuple[str, object]]
            List of tuples containing (model_name, model_class).
        x_train : array-like
            Training features.
        y_train : array-like
            Training target values.

        Returns
        -------
        Tuple[Dict, List]
            - Dict mapping model names to trained model instances
            - List of evaluation results for each model

        Notes
        -----
        Models that fail to train (return None) are skipped in the results.
        """
        # Initialize containers for results
        result_models = {}
        evaluate_results = []

        # Iterate over each model
        for model in models:
            model_name, _ = model
            # Train the model
            trained_model = train_function(model, x_train, y_train)
            
            # Store results if training was successful
            if trained_model is not None:
                result_models[model_name] = trained_model
                evaluate_results.append(
                    evaluate_function(trained_model, x_train, y_train)
                )
        
        return result_models, evaluate_results
    
    @staticmethod
    def get_iteration_strategy(library: str) -> Callable:
        """
        Get the appropriate iteration strategy for a given library.

        Factory method that returns the correct iteration strategy based
        on the machine learning library being used.

        Parameters
        ----------
        library : str
            Name of the machine learning library.
            Currently supported: "sklearn"

        Returns
        -------
        Callable
            Function implementing the iteration strategy.

        Notes
        -----
        Falls back to a simple sequential execution if no specific
        strategy is found for the given library.
        """
        # Define mapping of libraries to their strategies
        strategies = {
            "sklearn": IterationStrategy.iterate_sklearn_models,
            # Add other libraries here if needed
        }

        # Return appropriate strategy or fallback
        if library not in strategies:
            print(f"⚠️ Warning: No iteration strategy found for '{library}'. "
                  "Defaulting to sequential execution.")
            return lambda models, trainer, x_train, y_train: [
                (model[0], trainer.train_model(model[1], x_train, y_train))
                for model in models
            ]

        return strategies[library]