"""
mltunex.ai_handler.llm_handler_base
─────────────────────────────────────
Abstract base class and registry for LLM hyperparameter generators.

Design
------
* LLMHandlerConfig    — lightweight dataclass carrying everything a handler
                        needs at construction time (model name, temperature,
                        system prompt, framework name).  Decouples handler
                        construction from provider-specific config classes.

* BaseLLMHandler      — abstract interface every handler must implement.
                        Extends AIAdvisor so it can be passed wherever an
                        AIAdvisor is expected (LSP).

* LLMHandlerRegistry  — open registry; maps provider_name → handler class.
                        Users attach custom LLMs by registering a subclass.

SOLID notes
-----------
* OCP  — new providers registered via LLMHandlerRegistry.register()
* DIP  — orchestrator depends on BaseLLMHandler, never on concrete classes
* LSP  — every registered handler is a valid BaseLLMHandler / AIAdvisor

Usage — attaching a custom LLM
-------------------------------
>>> from mltunex.ai_handler.llm_handler_base import (
...     BaseLLMHandler, LLMHandlerConfig, LLMHandlerRegistry
... )
>>> class MyVendorHandler(BaseLLMHandler):
...     provider_name = "MyVendor"
...
...     def _call_llm(self, prompt_vars: dict) -> str:
...         # call your own API, return raw string response
...         return my_api.complete(**prompt_vars)
...
>>> LLMHandlerRegistry.register(MyVendorHandler)
>>> # Now "MyVendor:my-model-name" works everywhere in MLTuneX
"""

from __future__ import annotations

import json
import logging
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Type

from mltunex.ai_handler.prompt import LLMPrompts
from mltunex.ai_handler.response_schema_registry import ResponseSchemaRegistry
from mltunex.ai_handler.hyperparam_schema import FallbackHyperparams, MAX_RETRIES
from mltunex.hyperparam_tuner.optimizer import AIAdvisor

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Token usage tracking
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TokenUsage:
    """
    Accumulated token usage across all LLM calls made by one handler instance.

    Attributes
    ----------
    provider : str
        Provider name (e.g. "Groq", "OpenAI").
    model_name : str
        Model identifier (e.g. "qwen/qwen3-32b").
    prompt_tokens : int
        Total tokens consumed by prompts.
    completion_tokens : int
        Total tokens in generated completions.
    total_tokens : int
        prompt_tokens + completion_tokens.
    calls : int
        Number of successful LLM calls recorded.
    fallback_used : bool
        True if the fallback hyperparameter registry was used.
    """
    provider: str = ""
    model_name: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0
    fallback_used: bool = False

    def record(self, response_obj: Any) -> None:
        """
        Extract and accumulate token counts from a LangChain AIMessage.

        Works for both Groq and OpenAI responses which surface usage metadata
        as ``response.usage_metadata`` (LangChain >=0.2) or
        ``response.response_metadata["token_usage"]`` (older).
        Silently skips if metadata is unavailable (e.g. streaming, mocked).
        """
        self.calls += 1
        try:
            # LangChain 0.2+ unified interface
            if hasattr(response_obj, "usage_metadata") and response_obj.usage_metadata:
                meta = response_obj.usage_metadata
                self.prompt_tokens     += int(meta.get("input_tokens",  0))
                self.completion_tokens += int(meta.get("output_tokens", 0))
                self.total_tokens      += int(meta.get("total_tokens",
                                              meta.get("input_tokens",0) +
                                              meta.get("output_tokens",0)))
                return
            # Older LangChain: response_metadata["token_usage"] or ["usage"]
            if hasattr(response_obj, "response_metadata"):
                rm = response_obj.response_metadata
                tu = rm.get("token_usage") or rm.get("usage") or {}
                pt = int(tu.get("prompt_tokens",      tu.get("input_tokens",  0)))
                ct = int(tu.get("completion_tokens",  tu.get("output_tokens", 0)))
                tt = int(tu.get("total_tokens",       pt + ct))
                self.prompt_tokens     += pt
                self.completion_tokens += ct
                self.total_tokens      += tt
        except Exception:
            pass  # never crash on missing metadata


# ─────────────────────────────────────────────────────────────────────────────
# Config dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMHandlerConfig:
    """
    Provider-agnostic configuration for any LLM handler.

    Parameters
    ----------
    model_name : str
        The model identifier string passed to the provider API.
    temperature : float
        Sampling temperature (0 = deterministic).
    system_prompt : str
        The full system/user prompt template.  Must contain the placeholders
        ``{Data_Profile}``, ``{Top_Models}``, ``{ModelHyperparameter_Schema}``,
        and ``{HyperparameterResponsePrompt}``.
    framework : str
        Name of the tuning framework whose schema is used for validation
        and format instructions (looked up in ResponseSchemaRegistry).
    extra : dict
        Any provider-specific extra kwargs forwarded to the underlying client.
    """

    model_name: str
    temperature: float = 0.0
    system_prompt: str = field(default_factory=lambda: LLMPrompts.OpenAIPrompt)
    framework: str = "Optuna"
    extra: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base handler
# ─────────────────────────────────────────────────────────────────────────────

class BaseLLMHandler(AIAdvisor):
    """
    Abstract base class for all LLM hyperparameter generators.

    Subclasses only need to implement ``_call_llm`` — the low-level method
    that sends a prompt to the provider and returns the raw string response.

    The base class handles:
    - Building the full prompt from ``LLMHandlerConfig``
    - Injecting the correct format instructions from ``ResponseSchemaRegistry``
    - Retry loop (up to MAX_RETRIES) with exponential back-off
    - JSON parsing and normalisation
    - Schema validation via the registered ``TuningResponseSchema``
    - Fallback to ``FallbackHyperparams`` when all retries are exhausted
    """

    #: Subclasses must set this to the provider name used in registry lookup
    provider_name: ClassVar[str] = ""

    def __init__(self, config: LLMHandlerConfig) -> None:
        self._config = config
        self._schema = ResponseSchemaRegistry.get(config.framework)
        self.token_usage = TokenUsage(
            provider   = self.__class__.provider_name,
            model_name = config.model_name,
        )

    # ── AIAdvisor interface ───────────────────────────────────────────────────

    def suggest_search_spaces(
        self,
        data_profile: str,
        top_models: str,
        model_hyperparameter_schema: str,
    ) -> List[Dict[str, Any]]:
        """
        Generate search spaces with retry + schema validation + fallback.

        Implements AIAdvisor.suggest_search_spaces so this handler can be
        passed directly wherever an AIAdvisor is expected.
        """
        prompt_vars = {
            "Data_Profile":               data_profile,
            "Top_Models":                 top_models,
            "ModelHyperparameter_Schema": model_hyperparameter_schema,
            "HyperparameterResponsePrompt": self._schema.format_instructions(),
        }

        last_error: Optional[Exception] = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw_text, response_obj = self._call_llm_tracked(prompt_vars)
                self.token_usage.record(response_obj)
                parsed    = self._parse(raw_text)
                validated = self._schema.validate(parsed)
                logger.info(
                    "%s: success on attempt %d/%d",
                    type(self).__name__, attempt, MAX_RETRIES,
                )
                return validated

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "%s: attempt %d/%d failed — %s",
                    type(self).__name__, attempt, MAX_RETRIES, exc,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(1.5 * attempt)

        # All retries exhausted → use predefined fallback
        model_names = self._extract_model_names(top_models)
        logger.warning(
            "%s: all %d attempts failed (%s). "
            "Using predefined fallback hyperparameters for: %s",
            type(self).__name__, MAX_RETRIES, last_error, model_names,
        )
        self.token_usage.fallback_used = True
        return FallbackHyperparams.get(model_names)

    # ── Subclass contract ─────────────────────────────────────────────────────

    def _call_llm_tracked(self, prompt_vars: Dict[str, str]):
        """
        Call the LLM and return ``(raw_text, response_object)``.

        The response object is passed to ``token_usage.record()`` to extract
        usage metadata.  Subclasses that override ``_call_llm`` still work
        because this method falls back to calling ``_call_llm`` and wrapping
        the string in a dummy object.
        """
        # If the subclass overrides _call_llm_with_obj, use that
        if hasattr(self, "_call_llm_with_obj"):
            return self._call_llm_with_obj(prompt_vars)  # type: ignore
        raw = self._call_llm(prompt_vars)
        # Subclass only returns str — no metadata available; that's fine
        return raw, None

    @abstractmethod
    def _call_llm(self, prompt_vars: Dict[str, str]) -> str:
        """
        Send the prompt to the provider and return the raw text response.

        Parameters
        ----------
        prompt_vars : dict
            The four standard variables already populated with values.

        Returns
        -------
        str
            Raw response string (may contain thinking blocks, code fences, etc.).
        """

    # ── Shared parsing helpers ────────────────────────────────────────────────

    @staticmethod
    def _parse(response: str) -> Any:
        """Strip think-blocks / fences, repair JSON, return Python object."""
        from json_repair import repair_json

        # Strip <think> chain-of-thought (Groq reasoning models)
        if "</think>" in response:
            response = response.split("</think>")[-1]

        response = response.strip()

        # Strip markdown code fences
        if "```" in response:
            parts  = response.split("```")
            fenced = parts[1].strip()
            if fenced.lower().startswith("json"):
                fenced = fenced[4:].strip()
            response = fenced

        try:
            repaired = repair_json(response)
            parsed   = json.loads(repaired) if isinstance(repaired, str) else repaired
        except Exception as exc:
            raise ValueError(
                f"JSON repair/parse failed.\n"
                f"Raw (first 400 chars): {response[:400]}\n"
                f"Error: {exc}"
            ) from exc

        # Normalise dict-wrapped responses {"models": [...]} → [...]
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list) and v:
                    return v
            raise ValueError(
                f"LLM returned a dict with no list value. Keys: {list(parsed.keys())}"
            )

        if not isinstance(parsed, list):
            raise ValueError(
                f"Expected a JSON array, got {type(parsed).__name__}."
            )
        if not parsed:
            raise ValueError("LLM returned an empty JSON array.")

        # Drop entries missing model_name
        result = []
        for item in parsed:
            if not isinstance(item, dict) or "model_name" not in item:
                logger.warning("Skipping invalid entry: %r", item)
                continue
            if "suggested_hyperparameters" not in item:
                item["suggested_hyperparameters"] = {}
            result.append(item)

        if not result:
            raise ValueError("No valid model entries in LLM response.")

        return result

    @staticmethod
    def _extract_model_names(top_models_json: str) -> List[str]:
        """Best-effort extraction of model names from pandas .to_json() output."""
        try:
            data = json.loads(top_models_json)
            if isinstance(data, dict) and "Model" in data:
                return [str(v) for v in data["Model"].values()]
        except Exception:
            pass
        return []

    # ── Legacy compatibility: generate_response ───────────────────────────────

    def generate_response(
        self,
        data_profile: str,
        top_models: str,
        model_hyperparameter_schema: str,
    ) -> List[Dict[str, Any]]:
        """
        Alias of ``suggest_search_spaces`` for backward compatibility with
        code that calls ``llm.generate_response(...)``.
        """
        return self.suggest_search_spaces(
            data_profile, top_models, model_hyperparameter_schema
        )


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

class LLMHandlerRegistry:
    """
    Open registry mapping provider names → BaseLLMHandler subclasses.

    Built-in registrations happen in llm_manager.py at import time.

    Extending
    ---------
    >>> from mltunex.ai_handler.llm_handler_base import (
    ...     BaseLLMHandler, LLMHandlerConfig, LLMHandlerRegistry
    ... )
    >>> class MyVendorHandler(BaseLLMHandler):
    ...     provider_name = "MyVendor"
    ...     def _call_llm(self, prompt_vars):
    ...         return my_client.complete(prompt_vars["Data_Profile"])
    ...
    >>> LLMHandlerRegistry.register(MyVendorHandler)
    >>> # Now usable as: OrchestratorConfig(model_provider_model_name="MyVendor:my-model")
    """

    _registry: Dict[str, Type[BaseLLMHandler]] = {}

    @classmethod
    def register(cls, handler_class: Type[BaseLLMHandler]) -> None:
        """
        Register a handler class.

        Parameters
        ----------
        handler_class : Type[BaseLLMHandler]
            Must have a non-empty ``provider_name`` class variable.
        """
        if not handler_class.provider_name:
            raise ValueError(
                f"{handler_class.__name__} must define a non-empty 'provider_name' class variable."
            )
        cls._registry[handler_class.provider_name.lower()] = handler_class

    @classmethod
    def create(
        cls,
        provider: str,
        model_name: str,
        framework: str = "Optuna",
        temperature: float = 0.0,
        **extra: Any,
    ) -> BaseLLMHandler:
        """
        Instantiate a handler for *provider* / *model_name*.

        Parameters
        ----------
        provider : str
            Provider name (case-insensitive), e.g. ``"Groq"``, ``"OpenAI"``.
        model_name : str
            Model identifier passed to the provider API.
        framework : str
            Tuning framework whose schema is used (default: ``"Optuna"``).
        temperature : float
            Sampling temperature.
        **extra : Any
            Additional kwargs forwarded to ``LLMHandlerConfig.extra``.

        Returns
        -------
        BaseLLMHandler

        Raises
        ------
        ValueError
            If *provider* is not registered.
        """
        key = provider.lower()
        if key not in cls._registry:
            raise ValueError(
                f"No LLM handler registered for provider '{provider}'. "
                f"Registered: {list(cls._registry.keys())}. "
                f"Register a new handler with LLMHandlerRegistry.register(MyHandler)."
            )
        config = LLMHandlerConfig(
            model_name=model_name,
            temperature=temperature,
            framework=framework,
            extra=extra,
        )
        return cls._registry[key](config=config)

    @classmethod
    def list_providers(cls) -> List[str]:
        """Return registered provider names."""
        return list(cls._registry.keys())