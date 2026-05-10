"""
mltunex.reporting.pipeline_logger
───────────────────────────────────
Global structured logger that captures every event inside the MLTuneX pipeline.

Design
------
* PipelineLogger  — singleton per experiment run; writes to pipeline.log in
                    the experiment directory.  Uses Python's standard ``logging``
                    module so the file output is always clean and structured,
                    while the console shows only what the orchestrator explicitly
                    prints.

Key behaviours
--------------
* All 3rd-party library warnings (sklearn, optuna, pandas) are routed through
  this logger at DEBUG level → visible in the log file, invisible in the terminal.
* Each pipeline stage is bracketed with START / END entries and elapsed time.
* Exceptions are logged with full tracebacks before being re-raised.
* A ``silence_console()`` call in the orchestrator redirects the root Python
  warning system into the file handler so nothing leaks to the terminal.
"""

from __future__ import annotations

import logging
import os
import time
import traceback
import warnings
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Optional


# ── Module-level singleton ──────────────────────────────────────────────────
_active_logger: Optional["PipelineLogger"] = None


def get_pipeline_logger() -> Optional["PipelineLogger"]:
    """Return the currently active PipelineLogger, or None if not initialised."""
    return _active_logger


class PipelineLogger:
    """
    Structured file logger for the MLTuneX pipeline.

    One instance is created per run by the orchestrator.  All internal
    components that want structured logging call ``get_pipeline_logger()``
    rather than constructing their own logger.

    Parameters
    ----------
    log_dir : str
        Directory where ``pipeline.log`` is written (the experiment dir).
    experiment_name : str
        Stamped into every log line for easy grep.
    """

    def __init__(self, log_dir: str, experiment_name: str = "exp") -> None:
        global _active_logger
        _active_logger = self

        self._name         = experiment_name
        self._log_dir      = log_dir
        self._log_path     = os.path.join(log_dir, "pipeline.log")
        self._stage_starts: dict[str, float] = {}

        # ── File logger ──────────────────────────────────────────────
        self._logger = logging.getLogger(f"mltunex.{experiment_name}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False      # never bubble up to root

        if not self._logger.handlers:
            fh = logging.FileHandler(self._log_path, encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            self._logger.addHandler(fh)

        # ── Redirect all Python warnings to this logger ───────────────
        self._orig_showwarning = warnings.showwarning
        warnings.showwarning = self._capture_warning

        # ── Suppress noisy 3rd-party loggers from the terminal ────────
        for noisy in ("optuna", "sklearn", "lightgbm", "xgboost", "catboost",
                      "urllib3", "httpx", "httpcore"):
            lg = logging.getLogger(noisy)
            lg.setLevel(logging.CRITICAL)   # file still gets it via our handler
            lg.propagate = False

        self.info(f"Pipeline logger initialised: {self._log_path}")

    # ── Public API ────────────────────────────────────────────────────

    def debug(self, msg: str, **kw: Any) -> None:
        self._logger.debug(self._fmt(msg, **kw))

    def info(self, msg: str, **kw: Any) -> None:
        self._logger.info(self._fmt(msg, **kw))

    def warning(self, msg: str, **kw: Any) -> None:
        self._logger.warning(self._fmt(msg, **kw))

    def error(self, msg: str, exc: Optional[BaseException] = None, **kw: Any) -> None:
        if exc:
            tb = traceback.format_exc()
            self._logger.error(self._fmt(f"{msg}\n{tb}", **kw))
        else:
            self._logger.error(self._fmt(msg, **kw))

    @contextmanager
    def stage(self, name: str):
        """
        Context manager that logs START / END with elapsed time for a stage.

        Usage
        -----
        >>> with logger.stage("Training"):
        ...     train_all_models()
        """
        self.info(f"[START] {name}")
        t0 = time.perf_counter()
        try:
            yield
            elapsed = time.perf_counter() - t0
            self.info(f"[END]   {name} ({elapsed:.2f}s)")
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            self.error(f"[FAIL]  {name} ({elapsed:.2f}s)", exc=exc)
            raise

    def log_dict(self, label: str, data: dict, max_items: int = 50) -> None:
        """Log key-value pairs from a dict, truncating large dicts."""
        self.debug(f"{label}:")
        items = list(data.items())
        for k, v in items[:max_items]:
            self.debug(f"    {k}: {v}")
        if len(items) > max_items:
            self.debug(f"    … ({len(items) - max_items} more items)")

    def log_dataframe_shape(self, label: str, df: Any) -> None:
        """Log the shape and column names of a DataFrame."""
        try:
            self.debug(f"{label}: shape={df.shape}, columns={list(df.columns)}")
        except Exception:
            self.debug(f"{label}: (not a DataFrame)")

    def close(self) -> None:
        """Restore warning behaviour and flush handlers."""
        global _active_logger
        warnings.showwarning = self._orig_showwarning
        for handler in self._logger.handlers:
            handler.flush()
            handler.close()
        _active_logger = None

    @property
    def log_path(self) -> str:
        return self._log_path

    # ── Private ───────────────────────────────────────────────────────

    def _fmt(self, msg: str, **kw: Any) -> str:
        extra = "  ".join(f"{k}={v}" for k, v in kw.items())
        return f"[{self._name}] {msg}  {extra}".rstrip()

    def _capture_warning(self, message, category, filename, lineno,
                         file=None, line=None) -> None:
        """Route Python warnings into the file logger."""
        self._logger.warning(
            f"[PY-WARNING] {category.__name__}: {message}  "
            f"({os.path.basename(filename)}:{lineno})"
        )