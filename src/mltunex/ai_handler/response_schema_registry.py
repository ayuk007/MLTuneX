"""
mltunex.ai_handler.response_schema_registry
─────────────────────────────────────────────
Registry of response-format schemas keyed by hyperparameter-tuning framework.

Design
------
* TuningResponseSchema   — abstract interface every schema must implement
* OptunaResponseSchema   — concrete schema for Optuna (replaces hard-coded
                           OptunaPrompt in prompt.py)
* ResponseSchemaRegistry — open registry; users register custom schemas
                           for new tuning backends without touching core code

SOLID notes
-----------
* OCP  — new frameworks registered via ResponseSchemaRegistry.register()
* ISP  — TuningResponseSchema only exposes what the caller needs:
         format_instructions() and validate()
* DIP  — LLM handlers depend on TuningResponseSchema, never on concrete classes

Usage
-----
>>> schema = ResponseSchemaRegistry.get("Optuna")
>>> prompt_fragment = schema.format_instructions()   # injected into LLM prompt
>>> validated = schema.validate(llm_output)          # raises ValueError on bad output
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from mltunex.ai_handler.hyperparam_schema import HyperparamSchema


# ─────────────────────────────────────────────────────────────────────────────
# Abstract interface
# ─────────────────────────────────────────────────────────────────────────────

class TuningResponseSchema(ABC):
    """
    Abstract interface for a tuning-framework response schema.

    Responsibilities
    ----------------
    1. Provide a prompt fragment that instructs the LLM on the exact
       output format required by this tuning framework.
    2. Validate and normalise a parsed LLM response against that format.
    """

    @abstractmethod
    def format_instructions(self) -> str:
        """
        Return the prompt fragment that describes the expected output format.

        This string is injected into the LLM prompt as
        ``{HyperparameterResponsePrompt}``.  It must be self-contained and
        include a concrete JSON example so the LLM can follow it reliably.
        """

    @abstractmethod
    def validate(self, raw: Any) -> List[Dict[str, Any]]:
        """
        Validate and normalise *raw* (the parsed LLM output).

        Parameters
        ----------
        raw : Any
            The Python object produced by parsing the LLM response.

        Returns
        -------
        List[Dict[str, Any]]
            The validated, normalised list of model search-space dicts.

        Raises
        ------
        ValueError
            If *raw* does not conform to the expected schema.
        """

    @property
    @abstractmethod
    def framework_name(self) -> str:
        """Human-readable name of the tuning framework this schema targets."""


# ─────────────────────────────────────────────────────────────────────────────
# Optuna schema
# ─────────────────────────────────────────────────────────────────────────────

class OptunaResponseSchema(TuningResponseSchema):
    """
    Response schema for Optuna-compatible hyperparameter search spaces.

    Produces the format-instructions prompt fragment and delegates
    structural validation to HyperparamSchema (the pydantic-style
    validator in hyperparam_schema.py).
    """

    @property
    def framework_name(self) -> str:
        return "Optuna"

    def format_instructions(self) -> str:
        return """
Return a **JSON array** — one object per model — using the following schema.
Do NOT include any explanatory text, markdown fences, or keys outside this structure.

[
  {
    "model_name": "<EstimatorClassName>",
    "suggested_hyperparameters": {
      "<param_name>": {
        "type": "int",
        "low": <int>,
        "high": <int>,
        "step": <int>          // optional
      },
      "<param_name>": {
        "type": "float",
        "low": <float>,
        "high": <float>,
        "log": true            // optional — use for learning-rate style params
      },
      "<param_name>": {
        "type": "categorical",
        "values": [<val>, ...]  // non-empty list; no null/None values
      },
      "<param_name>": {
        "type": "bool"
      },
      "<param_name>": {
        "type": "fixed",
        "value": <val>
      }
    }
  },
  ...
]

Rules:
- Only include hyperparameters that are meaningful for the dataset profile.
- Do NOT use null or None as a parameter value.
- "low" must be strictly less than "high".
- "values" lists must be non-empty.
- Output valid JSON only — no markdown, no prose, no comments.
"""

    def validate(self, raw: Any) -> List[Dict[str, Any]]:
        """Delegate to HyperparamSchema which enforces the full Optuna contract."""
        return HyperparamSchema.validate(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

class ResponseSchemaRegistry:
    """
    Open registry mapping framework names → TuningResponseSchema instances.

    Built-in registrations
    ----------------------
    "Optuna"  →  OptunaResponseSchema

    Extending
    ---------
    To support a new tuning backend (e.g. Ray Tune, HyperOpt):

    >>> from mltunex.ai_handler.response_schema_registry import (
    ...     TuningResponseSchema, ResponseSchemaRegistry
    ... )
    >>> class RayTuneSchema(TuningResponseSchema):
    ...     framework_name = "RayTune"
    ...     def format_instructions(self): return "... your format ..."
    ...     def validate(self, raw): return raw   # add validation
    ...
    >>> ResponseSchemaRegistry.register(RayTuneSchema())
    >>> schema = ResponseSchemaRegistry.get("RayTune")
    """

    _registry: Dict[str, TuningResponseSchema] = {}

    @classmethod
    def register(cls, schema: TuningResponseSchema) -> None:
        """
        Register a TuningResponseSchema instance.

        Parameters
        ----------
        schema : TuningResponseSchema
            An instance whose ``framework_name`` property is used as the key.
            Registering a name that already exists overwrites the previous entry.
        """
        cls._registry[schema.framework_name] = schema

    @classmethod
    def get(cls, framework: str) -> TuningResponseSchema:
        """
        Retrieve the schema for *framework*.

        Parameters
        ----------
        framework : str
            The tuning framework name (case-sensitive, e.g. ``"Optuna"``).

        Returns
        -------
        TuningResponseSchema

        Raises
        ------
        ValueError
            If *framework* is not registered.
        """
        if framework not in cls._registry:
            raise ValueError(
                f"No response schema registered for framework '{framework}'. "
                f"Registered: {list(cls._registry.keys())}. "
                f"Register a new schema with ResponseSchemaRegistry.register(MySchema())."
            )
        return cls._registry[framework]

    @classmethod
    def list_frameworks(cls) -> List[str]:
        """Return the names of all registered frameworks."""
        return list(cls._registry.keys())


# ── Register built-ins at import time ────────────────────────────────────────
ResponseSchemaRegistry.register(OptunaResponseSchema())