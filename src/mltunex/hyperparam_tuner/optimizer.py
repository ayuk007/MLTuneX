"""
Optimizer abstraction layer for MLTuneX.

Design
------
* Optimizer           — abstract interface every optimizer must fulfil
* AIAdvisor           — abstract interface for AI-based search-space advisors
* LLMAdvisorAdapter   — Adapter Pattern: bridges the existing LLMManager API
                        into the AIAdvisor interface so the optimizer remains
                        decoupled from any LLM backend
* OptunaOptimizer     — concrete Optimizer implementation using Optuna;
                        accepts an optional AIAdvisor to seed search spaces

SOLID compliance
----------------
* SRP  — each class has one reason to change
* OCP  — new advisors / optimizers register without touching core code
* LSP  — any AIAdvisor or Optimizer subclass is a valid substitution
* ISP  — AIAdvisor and Optimizer are small, focused interfaces
* DIP  — OptunaOptimizer depends on AIAdvisor abstract; not on LLMManager
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore", category=UserWarning, module="optuna")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# AIAdvisor interface
# ---------------------------------------------------------------------------

class AIAdvisor(ABC):
    """
    Abstract interface for AI-based hyperparameter advisors.

    An advisor receives dataset profile information and candidate model
    names, then returns a list of search-space dictionaries in the format
    that the Optimizer understands.

    The Optimizer depends on *this* interface — never on a concrete LLM.
    """

    @abstractmethod
    def suggest_search_spaces(
        self,
        data_profile: str,
        top_models: str,
        model_hyperparameter_schema: str,
    ) -> List[Dict[str, Any]]:
        """
        Suggest hyperparameter search spaces for the given models.

        Returns
        -------
        List[Dict[str, Any]]
            Each element is a dict with keys:
            - "model_name"               : str
            - "suggested_hyperparameters": dict of param_name → param_def
        """


# ---------------------------------------------------------------------------
# Adapter: bridges LLMManager → AIAdvisor
# ---------------------------------------------------------------------------

class LLMAdvisorAdapter(AIAdvisor):
    """
    Adapter that wraps the existing LLMManager-based generators into the
    AIAdvisor interface.

    This keeps the Optimizer free of any LLM-specific imports and allows
    the AI component to be swapped (e.g., rule-based advisor) without
    modifying OptunaOptimizer.

    Parameters
    ----------
    llm_instance : Any
        An object that implements ``generate_response(data_profile, top_models,
        model_hyperparameter_schema) -> list``.
    """

    def __init__(self, llm_instance: Any) -> None:
        self._llm = llm_instance

    def suggest_search_spaces(
        self,
        data_profile: str,
        top_models: str,
        model_hyperparameter_schema: str,
    ) -> List[Dict[str, Any]]:
        response = self._llm.generate_response(
            data_profile=data_profile,
            top_models=top_models,
            schema=model_hyperparameter_schema,
        )
        # The LLM chain already returns a parsed list; validate shape.
        if not isinstance(response, list):
            raise ValueError(
                "LLMAdvisorAdapter: expected a list from generate_response, "
                f"got {type(response).__name__}."
            )
        return response


# ---------------------------------------------------------------------------
# Optimizer interface
# ---------------------------------------------------------------------------

class Optimizer(ABC):
    """
    Abstract interface for hyperparameter optimizers.

    Responsibilities
    ----------------
    Accept a model + search space definition, run the optimization
    process, and return the best model name and its hyperparameters.
    """

    @abstractmethod
    def optimize(
        self,
        model_search_spaces: List[Dict[str, Any]],
        x_train: pd.DataFrame,
        y_train: pd.Series,
        trained_models: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Run the optimization process.

        Parameters
        ----------
        model_search_spaces : List[Dict]
            AI-suggested or manually defined search spaces.
        x_train : pd.DataFrame
        y_train : pd.Series
        trained_models : Dict[str, Any]
            ``{model_name: (model_name, fitted_estimator)}`` produced by the
            training loop.  The optimizer re-instantiates models from this dict
            during each trial.

        Returns
        -------
        Tuple[str, Dict[str, Any]]
            (best_model_name, best_hyperparameters)
        """


# ---------------------------------------------------------------------------
# Concrete optimizer — Optuna
# ---------------------------------------------------------------------------

class OptunaOptimizer(Optimizer):
    """
    Optuna-based hyperparameter optimizer.

    Optionally accepts an AIAdvisor to generate search spaces on the fly.
    When no advisor is supplied, the caller must provide *model_search_spaces*
    directly to :meth:`optimize`.

    Parameters
    ----------
    task_type : str
        "classification" → optimise accuracy; "regression" → R²
    n_trials : int
        Number of Optuna trials.
    advisor : AIAdvisor, optional
        AI advisor for dynamic search-space generation.
    library : str
        ML library backend (currently "sklearn").
    """

    def __init__(
        self,
        task_type: str,
        n_trials: int = 25,
        advisor: Optional[AIAdvisor] = None,
        library: str = "sklearn",
    ) -> None:
        from mltunex.library_trainer.library_trainer import LibraryTrainer

        self._task_type = task_type
        self._n_trials = n_trials
        self._advisor = advisor
        self._model_trainer = LibraryTrainer.get_trainer(library=library)
        self._scoring = "accuracy" if task_type == "classification" else "r2"

    # ------------------------------------------------------------------
    # Optimizer interface
    # ------------------------------------------------------------------

    def optimize(
        self,
        model_search_spaces: List[Dict[str, Any]],
        x_train: pd.DataFrame,
        y_train: pd.Series,
        trained_models: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        self._trial_history: List[Dict[str, Any]] = []

        study = optuna.create_study(direction="maximize")
        objective = self._build_objective(
            model_search_spaces, x_train, y_train, trained_models
        )

        def _record_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            model_cfg = trial.params.get("model_config", {})
            model_name = model_cfg.get("model_name", "?") if isinstance(model_cfg, dict) else "?"
            self._trial_history.append({
                "trial":  trial.number,
                "model":  model_name,
                "score":  trial.value if trial.value is not None else 0.0,
                "params": self._extract_params(trial.params, model_name),
            })

        study.optimize(objective, n_trials=self._n_trials, callbacks=[_record_trial])

        best_trial = study.best_trial
        best_model_name = best_trial.params["model_config"]["model_name"]
        best_params = self._extract_params(best_trial.params, best_model_name)
        self._best_score = best_trial.value or 0.0
        return best_model_name, best_params

    @property
    def trial_history(self) -> List[Dict[str, Any]]:
        """All recorded trials after :meth:`optimize` has been called."""
        return list(getattr(self, "_trial_history", []))

    @property
    def best_score(self) -> float:
        """Best cross-val score found during optimisation."""
        return getattr(self, "_best_score", 0.0)

    # ------------------------------------------------------------------
    # Public helper — let orchestration layer pass AI profile if needed
    # ------------------------------------------------------------------

    def get_search_spaces_from_advisor(
        self,
        data_profile: str,
        top_models: str,
        model_hyperparameter_schema: str,
    ) -> List[Dict[str, Any]]:
        """Delegate to the registered AIAdvisor."""
        if self._advisor is None:
            raise RuntimeError(
                "No AIAdvisor is registered with this OptunaOptimizer."
            )
        return self._advisor.suggest_search_spaces(
            data_profile, top_models, model_hyperparameter_schema
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_objective(
        self,
        model_search_spaces: List[Dict[str, Any]],
        x_train: pd.DataFrame,
        y_train: pd.Series,
        trained_models: Dict[str, Any],
    ):
        scoring = self._scoring
        model_trainer = self._model_trainer

        def objective(trial: optuna.Trial) -> float:
            model_config = trial.suggest_categorical("model_config", model_search_spaces)
            model_name = model_config["model_name"]
            param_defs = model_config.get("suggested_hyperparameters", {})

            params: Dict[str, Any] = {}
            try:
                for param_name, param_def in param_defs.items():
                    full_name = f"{model_name}_{param_name}"
                    params[param_name] = _suggest_param(trial, full_name, param_def)

                # trained_models is {model_name: (name, fitted_estimator)}
                model_tuple = trained_models[model_name]
                model = model_trainer.train_model(
                    model=model_tuple[-1], params=params, tune=True
                )
                if model is None:
                    return 0.0
                return float(
                    cross_val_score(model, x_train, y_train, cv=3, scoring=scoring).mean()
                )
            except Exception as exc:
                print(f"[OptunaOptimizer] trial failed for {model_name}: {exc}")
                return 0.0

        return objective

    @staticmethod
    def _extract_params(
        best_params: Dict[str, Any], model_name: str
    ) -> Dict[str, Any]:
        prefix = f"{model_name}_"
        return {
            k.replace(prefix, ""): v
            for k, v in best_params.items()
            if k.startswith(prefix)
        }


# ---------------------------------------------------------------------------
# Shared utility — used by both OptunaOptimizer and legacy tuner
# ---------------------------------------------------------------------------

def _suggest_param(
    trial: optuna.Trial, param_name: str, param_def: Dict[str, Any]
) -> Any:
    """Suggest a parameter value from its definition dict."""
    type_ = param_def["type"]

    if type_ == "int":
        if "values" in param_def:
            return trial.suggest_categorical(param_name, param_def["values"])
        return trial.suggest_int(
            param_name,
            param_def["low"],
            param_def["high"],
            step=param_def.get("step", 1),
        )
    if type_ == "float":
        return trial.suggest_float(
            param_name,
            param_def["low"],
            param_def["high"],
            step=param_def.get("step"),
            log=param_def.get("log", False),
        )
    if type_ == "categorical":
        return trial.suggest_categorical(param_name, param_def["values"])
    if type_ == "bool":
        return trial.suggest_categorical(param_name, [True, False])
    if type_ == "fixed":
        return param_def["value"]

    raise ValueError(f"Unsupported parameter type: {type_}")


# ---------------------------------------------------------------------------
# OptimizerFactory
# ---------------------------------------------------------------------------

class OptimizerFactory:
    """
    Factory for Optimizer instances.

    Currently registered
    --------------------
    "optuna" → OptunaOptimizer
    """

    _registry: dict[str, type[Optimizer]] = {
        "optuna": OptunaOptimizer,
    }

    @classmethod
    def register(cls, name: str, optimizer_class: type[Optimizer]) -> None:
        cls._registry[name.lower()] = optimizer_class

    @classmethod
    def create(
        cls,
        method: str = "optuna",
        **kwargs: Any,
    ) -> Optimizer:
        key = method.lower()
        if key not in cls._registry:
            raise ValueError(
                f"Unknown optimizer '{method}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[key](**kwargs)
