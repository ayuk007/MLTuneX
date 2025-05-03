"""
Parallelization strategy module for MLTuneX.

This module provides different strategies for parallel model training based on
the machine learning library being used. It supports both CPU-based (scikit-learn)
and GPU-based (cuML) parallel processing.

Examples:
    >>> strategy = ParallelizationStrategy.get_parallel_strategy("sklearn")
    >>> results = strategy(train_fn, models, X, y, n_workers=4)
"""

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from typing import Callable, List, Tuple, Dict
from joblib import Parallel, delayed


def train_model(model: Tuple[str, object], X_train: object, y_train: object) -> object:
    """
    Train a scikit-learn model with the given training data.

    Parameters
    ----------
    model : Tuple[str, object]
        Tuple containing (model_name, model_instance).
    X_train : array-like
        Training features.
    y_train : array-like
        Training labels.

    Returns
    -------
    object
        Trained model instance.

    Notes
    -----
    This function extracts the model instance from the tuple and
    fits it to the provided training data.
    """
    model_name, estimator = model
    estimator.fit(X_train, y_train)
    return estimator


class ParallelizationStrategy:
    """
    Defines parallel execution strategies for different ML libraries.

    This class provides static methods implementing different approaches to
    parallel model training, optimized for specific machine learning libraries
    and hardware configurations.

    Methods
    -------
    parallel_sklearn : Parallel execution for CPU-based scikit-learn models
    parallel_cuml : Parallel execution for GPU-accelerated cuML models
    get_parallel_strategy : Factory method to get appropriate strategy
    """

    @staticmethod
    def parallel_sklearn(
        train_function: Callable, 
        models: List[Tuple[str, object]], 
        X_train: object, 
        y_train: object, 
        num_workers: int
    ) -> Dict[str, object]:
        """
        Parallel execution for Scikit-Learn models using thread pooling.

        Parameters
        ----------
        train_function : Callable
            Function to train individual models.
        models : List[Tuple[str, object]]
            List of (model_name, model_instance) tuples.
        X_train : array-like
            Training features.
        y_train : array-like
            Training labels.
        num_workers : int
            Number of parallel workers to use.

        Returns
        -------
        Dict[str, object]
            Dictionary mapping model names to trained models.

        Notes
        -----
        Uses ThreadPoolExecutor for parallel processing as scikit-learn
        models typically release the GIL during fitting.
        """
        # Create thread pool and execute training in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(
                train_function, 
                models, 
                [X_train] * len(models), 
                [y_train] * len(models)
            ))

        # Map results back to model names
        return {model[0]: result for model, result in zip(models, results)}

    @staticmethod
    def parallel_cuml(
        train_function: Callable, 
        models: List[Tuple[str, object]], 
        X_train: object, 
        y_train: object, 
        num_workers: int
    ) -> Dict[str, object]:
        """
        Parallel execution for cuML models using GPU acceleration.

        Parameters
        ----------
        train_function : Callable
            Function to train individual models.
        models : List[Tuple[str, object]]
            List of (model_name, model_instance) tuples.
        X_train : array-like
            Training features.
        y_train : array-like
            Training labels.
        num_workers : int
            Number of parallel workers to use.

        Returns
        -------
        Dict[str, object]
            Dictionary mapping model names to trained models.

        Notes
        -----
        Uses ThreadPoolExecutor as cuML models are GPU-accelerated and
        Python threading is sufficient for GPU task management.
        """
        def train_wrapper(model_info):
            """Helper function to maintain model names during parallel execution."""
            model_name, model = model_info
            return model_name, train_function(model, X_train, y_train)

        # Execute training in parallel using thread pool
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = dict(executor.map(train_wrapper, models))

        return results

    @staticmethod
    def get_parallel_strategy(library: str) -> Callable:
        """
        Get the appropriate parallelization strategy for a given library.

        Parameters
        ----------
        library : str
            Name of the machine learning library.
            Currently supported: ["sklearn", "cuml"]

        Returns
        -------
        Callable
            Function implementing the parallelization strategy.

        Notes
        -----
        Falls back to sequential execution if no specific strategy
        is found for the given library.
        """
        # Define mapping of libraries to their parallel strategies
        strategies = {
            "sklearn": ParallelizationStrategy.parallel_sklearn,
            "cuml": ParallelizationStrategy.parallel_cuml,
        }

        # Return appropriate strategy or fall back to sequential
        if library not in strategies:
            print(f"⚠️ Warning: No parallelization strategy found for '{library}'. "
                  "Defaulting to sequential execution.")
            return lambda train_function, models, X_train, y_train, _: {
                m[0]: train_function(m[1], X_train, y_train) for m in models
            }

        return strategies[library]