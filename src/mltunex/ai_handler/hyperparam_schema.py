"""
mltunex.ai_handler.hyperparam_schema
──────────────────────────────────────
Two responsibilities:

1. HyperparamSchema — validates and normalises LLM output against the
   expected search-space schema so bad responses are caught early with
   clear error messages.

2. FallbackHyperparams — predefined search spaces for every registered
   model.  Used when the LLM fails after MAX_RETRIES attempts.

Schema for a single model entry
────────────────────────────────
{
  "model_name": "<str>",
  "suggested_hyperparameters": {
    "<param_name>": {
      "type": "int" | "float" | "categorical" | "bool" | "fixed",
      // int / float
      "low":  <number>,
      "high": <number>,
      "step": <number>,       // optional
      "log":  <bool>,         // optional, float only
      // categorical
      "values": [<any>, ...],
      // fixed
      "value": <any>
    },
    ...
  }
}
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_RETRIES = 3

# ── Valid types ───────────────────────────────────────────────────────────────
_VALID_TYPES = {"int", "float", "categorical", "bool", "fixed"}


# ─────────────────────────────────────────────────────────────────────────────
# Schema validator
# ─────────────────────────────────────────────────────────────────────────────

class HyperparamSchema:
    """
    Validates and normalises a list of model search-space dicts.

    Raises
    ------
    ValueError
        On any structural violation so the caller can retry the LLM call.
    """

    @staticmethod
    def validate(raw: Any) -> List[Dict[str, Any]]:
        """
        Validate *raw* against the expected schema.

        Parameters
        ----------
        raw : Any
            The parsed Python object returned by the LLM handler.

        Returns
        -------
        List[Dict[str, Any]]
            The validated (and lightly normalised) list.
        """
        if not isinstance(raw, list):
            raise ValueError(
                f"Schema violation: expected a JSON array at the top level, "
                f"got {type(raw).__name__}."
            )
        if len(raw) == 0:
            raise ValueError("Schema violation: top-level array is empty.")

        validated: List[Dict[str, Any]] = []
        for i, item in enumerate(raw):
            validated.append(HyperparamSchema._validate_entry(i, item))
        return validated

    @staticmethod
    def _validate_entry(idx: int, item: Any) -> Dict[str, Any]:
        if not isinstance(item, dict):
            raise ValueError(
                f"Schema violation: entry [{idx}] must be a JSON object, "
                f"got {type(item).__name__}."
            )

        # ── model_name ───────────────────────────────────────────────
        if "model_name" not in item or not isinstance(item["model_name"], str):
            raise ValueError(
                f"Schema violation: entry [{idx}] missing or non-string 'model_name'."
            )

        # ── suggested_hyperparameters ────────────────────────────────
        if "suggested_hyperparameters" not in item:
            item = dict(item, suggested_hyperparameters={})
        hp = item["suggested_hyperparameters"]
        if not isinstance(hp, dict):
            raise ValueError(
                f"Schema violation: entry [{idx}] 'suggested_hyperparameters' "
                f"must be a JSON object, got {type(hp).__name__}."
            )

        normalised_hp: Dict[str, Any] = {}
        for param_name, defn in hp.items():
            normalised_hp[param_name] = HyperparamSchema._validate_param(
                idx, item["model_name"], param_name, defn
            )

        return {"model_name": item["model_name"],
                "suggested_hyperparameters": normalised_hp}

    @staticmethod
    def _validate_param(
        entry_idx: int, model_name: str, param: str, defn: Any
    ) -> Dict[str, Any]:
        loc = f"entry [{entry_idx}] model '{model_name}' param '{param}'"

        if not isinstance(defn, dict):
            raise ValueError(
                f"Schema violation: {loc} — parameter definition must be a "
                f"JSON object, got {type(defn).__name__}."
            )

        ptype = defn.get("type")
        if ptype not in _VALID_TYPES:
            raise ValueError(
                f"Schema violation: {loc} — 'type' must be one of "
                f"{sorted(_VALID_TYPES)}, got {ptype!r}."
            )

        if ptype in ("int", "float"):
            if "values" in defn:
                # Treat as categorical shorthand — accepted
                return {"type": "categorical", "values": defn["values"]}
            if "low" not in defn or "high" not in defn:
                raise ValueError(
                    f"Schema violation: {loc} — type '{ptype}' requires "
                    f"'low' and 'high' keys."
                )
            if not isinstance(defn["low"], (int, float)) or \
               not isinstance(defn["high"], (int, float)):
                raise ValueError(
                    f"Schema violation: {loc} — 'low' and 'high' must be numbers."
                )
            if defn["low"] >= defn["high"]:
                raise ValueError(
                    f"Schema violation: {loc} — 'low' ({defn['low']}) must be "
                    f"less than 'high' ({defn['high']})."
                )

        elif ptype == "categorical":
            if "values" not in defn or not isinstance(defn["values"], list):
                raise ValueError(
                    f"Schema violation: {loc} — type 'categorical' requires "
                    f"a 'values' list."
                )
            if len(defn["values"]) == 0:
                raise ValueError(
                    f"Schema violation: {loc} — 'values' list is empty."
                )

        elif ptype == "fixed":
            if "value" not in defn:
                raise ValueError(
                    f"Schema violation: {loc} — type 'fixed' requires a 'value' key."
                )

        return defn


# ─────────────────────────────────────────────────────────────────────────────
# Fallback hyperparameter registry
# ─────────────────────────────────────────────────────────────────────────────

# Format identical to the LLM output schema so the optimizer consumes it
# without any special-casing.

_FALLBACK: Dict[str, Dict[str, Any]] = {

    # ── Tree / Ensemble ───────────────────────────────────────────────────────

    "RandomForestClassifier": {
        "n_estimators":      {"type": "int",         "low": 50,   "high": 500},
        "max_depth":         {"type": "categorical",  "values": [3, 5, 7, 10, None]},
        "min_samples_split": {"type": "int",          "low": 2,    "high": 20},
        "min_samples_leaf":  {"type": "int",          "low": 1,    "high": 10},
        "max_features":      {"type": "categorical",  "values": ["sqrt","log2", None]},
        "criterion":         {"type": "categorical",  "values": ["gini","entropy"]},
    },
    "RandomForestRegressor": {
        "n_estimators":      {"type": "int",         "low": 50,   "high": 500},
        "max_depth":         {"type": "categorical",  "values": [3, 5, 7, 10, None]},
        "min_samples_split": {"type": "int",          "low": 2,    "high": 20},
        "min_samples_leaf":  {"type": "int",          "low": 1,    "high": 10},
        "max_features":      {"type": "categorical",  "values": ["sqrt","log2", None]},
    },
    "ExtraTreesClassifier": {
        "n_estimators":      {"type": "int",         "low": 50,   "high": 500},
        "max_depth":         {"type": "categorical",  "values": [3, 5, 7, 10, None]},
        "min_samples_split": {"type": "int",          "low": 2,    "high": 20},
        "max_features":      {"type": "categorical",  "values": ["sqrt","log2", None]},
        "criterion":         {"type": "categorical",  "values": ["gini","entropy"]},
    },
    "ExtraTreesRegressor": {
        "n_estimators":      {"type": "int",         "low": 50,   "high": 500},
        "max_depth":         {"type": "categorical",  "values": [3, 5, 7, 10, None]},
        "min_samples_split": {"type": "int",          "low": 2,    "high": 20},
        "max_features":      {"type": "categorical",  "values": ["sqrt","log2", None]},
    },
    "DecisionTreeClassifier": {
        "max_depth":         {"type": "categorical",  "values": [3, 5, 7, 10, 15, None]},
        "min_samples_split": {"type": "int",          "low": 2,    "high": 30},
        "min_samples_leaf":  {"type": "int",          "low": 1,    "high": 15},
        "criterion":         {"type": "categorical",  "values": ["gini","entropy"]},
        "max_features":      {"type": "categorical",  "values": ["sqrt","log2", None]},
    },
    "DecisionTreeRegressor": {
        "max_depth":         {"type": "categorical",  "values": [3, 5, 7, 10, 15, None]},
        "min_samples_split": {"type": "int",          "low": 2,    "high": 30},
        "min_samples_leaf":  {"type": "int",          "low": 1,    "high": 15},
        "max_features":      {"type": "categorical",  "values": ["sqrt","log2", None]},
    },

    # ── Boosting ──────────────────────────────────────────────────────────────

    "XGBClassifier": {
        "n_estimators":    {"type": "int",   "low": 50,   "high": 500},
        "max_depth":       {"type": "int",   "low": 2,    "high": 10},
        "learning_rate":   {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "subsample":       {"type": "float", "low": 0.5,  "high": 1.0},
        "colsample_bytree":{"type": "float", "low": 0.5,  "high": 1.0},
        "reg_alpha":       {"type": "float", "low": 1e-8, "high": 10.0,"log": True},
        "reg_lambda":      {"type": "float", "low": 1e-8, "high": 10.0,"log": True},
        "min_child_weight":{"type": "int",   "low": 1,    "high": 10},
    },
    "XGBRegressor": {
        "n_estimators":    {"type": "int",   "low": 50,   "high": 500},
        "max_depth":       {"type": "int",   "low": 2,    "high": 10},
        "learning_rate":   {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "subsample":       {"type": "float", "low": 0.5,  "high": 1.0},
        "colsample_bytree":{"type": "float", "low": 0.5,  "high": 1.0},
        "reg_alpha":       {"type": "float", "low": 1e-8, "high": 10.0,"log": True},
        "reg_lambda":      {"type": "float", "low": 1e-8, "high": 10.0,"log": True},
    },
    "LGBMClassifier": {
        "n_estimators":    {"type": "int",   "low": 50,   "high": 500},
        "max_depth":       {"type": "int",   "low": 3,    "high": 12},
        "learning_rate":   {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "num_leaves":      {"type": "int",   "low": 20,   "high": 150},
        "subsample":       {"type": "float", "low": 0.5,  "high": 1.0},
        "colsample_bytree":{"type": "float", "low": 0.5,  "high": 1.0},
        "reg_alpha":       {"type": "float", "low": 1e-8, "high": 10.0,"log": True},
        "reg_lambda":      {"type": "float", "low": 1e-8, "high": 10.0,"log": True},
    },
    "LGBMRegressor": {
        "n_estimators":    {"type": "int",   "low": 50,   "high": 500},
        "max_depth":       {"type": "int",   "low": 3,    "high": 12},
        "learning_rate":   {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "num_leaves":      {"type": "int",   "low": 20,   "high": 150},
        "subsample":       {"type": "float", "low": 0.5,  "high": 1.0},
        "colsample_bytree":{"type": "float", "low": 0.5,  "high": 1.0},
    },
    "AdaBoostClassifier": {
        "n_estimators":  {"type": "int",   "low": 50,  "high": 300},
        "learning_rate": {"type": "float", "low": 0.01,"high": 2.0, "log": True},
        "algorithm":     {"type": "categorical", "values": ["SAMME"]},
    },
    "AdaBoostRegressor": {
        "n_estimators":  {"type": "int",   "low": 50,  "high": 300},
        "learning_rate": {"type": "float", "low": 0.01,"high": 2.0, "log": True},
        "loss":          {"type": "categorical", "values": ["linear","square","exponential"]},
    },
    "BaggingClassifier": {
        "n_estimators":    {"type": "int",   "low": 10, "high": 200},
        "max_samples":     {"type": "float", "low": 0.5,"high": 1.0},
        "max_features":    {"type": "float", "low": 0.5,"high": 1.0},
    },

    # ── Linear ────────────────────────────────────────────────────────────────

    "LogisticRegression": {
        "C":           {"type": "float", "low": 1e-4, "high": 100.0, "log": True},
        "solver":      {"type": "categorical", "values": ["lbfgs","liblinear","saga"]},
        "max_iter":    {"type": "int",   "low": 100,  "high": 1000},
        "penalty":     {"type": "categorical", "values": ["l2", None]},
    },
    "RidgeClassifier": {
        "alpha":    {"type": "float", "low": 1e-4, "high": 100.0, "log": True},
        "solver":   {"type": "categorical",
                     "values": ["auto","svd","cholesky","lsqr","sparse_cg","sag","saga"]},
    },
    "Ridge": {
        "alpha":   {"type": "float", "low": 1e-4, "high": 100.0, "log": True},
        "solver":  {"type": "categorical",
                    "values": ["auto","svd","cholesky","lsqr","sparse_cg","sag","saga"]},
    },
    "Lasso": {
        "alpha":    {"type": "float", "low": 1e-4, "high": 10.0, "log": True},
        "max_iter": {"type": "int",   "low": 500,  "high": 5000},
    },
    "ElasticNet": {
        "alpha":    {"type": "float", "low": 1e-4, "high": 10.0, "log": True},
        "l1_ratio": {"type": "float", "low": 0.0,  "high": 1.0},
        "max_iter": {"type": "int",   "low": 500,  "high": 5000},
    },
    "SGDClassifier": {
        "alpha":        {"type": "float", "low": 1e-5, "high": 1.0, "log": True},
        "loss":         {"type": "categorical",
                         "values": ["hinge","log_loss","modified_huber","perceptron"]},
        "penalty":      {"type": "categorical", "values": ["l2","l1","elasticnet"]},
        "max_iter":     {"type": "int",   "low": 100, "high": 1000},
        "learning_rate":{"type": "categorical",
                         "values": ["constant","optimal","invscaling","adaptive"]},
    },
    "SGDRegressor": {
        "alpha":        {"type": "float", "low": 1e-5, "high": 1.0, "log": True},
        "loss":         {"type": "categorical",
                         "values": ["squared_error","huber","epsilon_insensitive"]},
        "penalty":      {"type": "categorical", "values": ["l2","l1","elasticnet"]},
        "max_iter":     {"type": "int",   "low": 100, "high": 1000},
    },

    # ── SVM ───────────────────────────────────────────────────────────────────

    "SVC": {
        "C":      {"type": "float", "low": 1e-3, "high": 100.0, "log": True},
        "kernel": {"type": "categorical", "values": ["rbf","linear","poly","sigmoid"]},
        "gamma":  {"type": "categorical", "values": ["scale","auto"]},
        "degree": {"type": "int",   "low": 2, "high": 5},
    },
    "SVR": {
        "C":       {"type": "float", "low": 1e-3, "high": 100.0, "log": True},
        "kernel":  {"type": "categorical", "values": ["rbf","linear","poly"]},
        "gamma":   {"type": "categorical", "values": ["scale","auto"]},
        "epsilon": {"type": "float", "low": 1e-4, "high": 1.0, "log": True},
    },
    "LinearSVC": {
        "C":        {"type": "float", "low": 1e-3, "high": 100.0, "log": True},
        "max_iter": {"type": "int",   "low": 500,  "high": 5000},
        "penalty":  {"type": "categorical", "values": ["l2"]},
    },

    # ── Neighbours ────────────────────────────────────────────────────────────

    "KNeighborsClassifier": {
        "n_neighbors": {"type": "int",  "low": 1,  "high": 30},
        "weights":     {"type": "categorical", "values": ["uniform","distance"]},
        "metric":      {"type": "categorical",
                        "values": ["euclidean","manhattan","chebyshev","minkowski"]},
        "p":           {"type": "int",  "low": 1, "high": 4},
    },
    "KNeighborsRegressor": {
        "n_neighbors": {"type": "int",  "low": 1,  "high": 30},
        "weights":     {"type": "categorical", "values": ["uniform","distance"]},
        "metric":      {"type": "categorical",
                        "values": ["euclidean","manhattan","minkowski"]},
    },

    # ── Naive Bayes ───────────────────────────────────────────────────────────

    "GaussianNB": {
        "var_smoothing": {"type": "float", "low": 1e-10, "high": 1e-1, "log": True},
    },
    "BernoulliNB": {
        "alpha":     {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
        "binarize":  {"type": "float", "low": 0.0,  "high": 1.0},
    },

    # ── Linear Discriminant / Quadratic ───────────────────────────────────────

    "LinearDiscriminantAnalysis": {
        "solver":    {"type": "categorical", "values": ["svd","lsqr","eigen"]},
        "shrinkage": {"type": "categorical", "values": ["auto", None]},
    },
    "QuadraticDiscriminantAnalysis": {
        "reg_param": {"type": "float", "low": 0.0, "high": 1.0},
    },

    # ── Other regressors ──────────────────────────────────────────────────────

    "GradientBoostingRegressor": {
        "n_estimators":    {"type": "int",   "low": 50,  "high": 400},
        "learning_rate":   {"type": "float", "low": 0.01,"high": 0.3, "log": True},
        "max_depth":       {"type": "int",   "low": 2,   "high": 8},
        "subsample":       {"type": "float", "low": 0.5, "high": 1.0},
        "min_samples_split":{"type":"int",   "low": 2,   "high": 20},
    },
    "LinearRegression": {
        "fit_intercept": {"type": "bool"},
        "positive":      {"type": "bool"},
    },
    "HuberRegressor": {
        "epsilon":  {"type": "float", "low": 1.01, "high": 3.0},
        "alpha":    {"type": "float", "low": 1e-5, "high": 1.0, "log": True},
        "max_iter": {"type": "int",   "low": 100,  "high": 1000},
    },
}


class FallbackHyperparams:
    """
    Returns predefined search spaces for known models.

    Used when the LLM fails to produce a valid response after MAX_RETRIES.
    The returned format is identical to what the LLM is expected to produce,
    so the optimizer consumes it without any special-casing.
    """

    @staticmethod
    def get(model_names: List[str]) -> List[Dict[str, Any]]:
        """
        Build a fallback search-space list for *model_names*.

        Models with no predefined entry get an empty hyperparameter dict,
        which causes the optimizer to skip tuning for that model.

        Parameters
        ----------
        model_names : List[str]
            Names of models that need search spaces.

        Returns
        -------
        List[Dict[str, Any]]
            One entry per model, in the standard schema format.
        """
        result = []
        for name in model_names:
            hp = _FALLBACK.get(name, {})
            if not hp:
                logger.warning(
                    "FallbackHyperparams: no predefined search space for '%s'. "
                    "It will be included with empty hyperparameters.", name
                )
            result.append({
                "model_name": name,
                "suggested_hyperparameters": copy.deepcopy(hp),
            })
        return result

    @staticmethod
    def has(model_name: str) -> bool:
        """Return True if a predefined space exists for *model_name*."""
        return model_name in _FALLBACK

    @staticmethod
    def list_supported() -> List[str]:
        """Return all model names with a predefined fallback space."""
        return sorted(_FALLBACK.keys())