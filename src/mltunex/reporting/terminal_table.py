# """
# mltunex.reporting.terminal_table
# ─────────────────────────────────
# Live terminal table that updates in-place as each model finishes training.

# Uses ``rich`` for a polished, YOLO-style output.  Falls back gracefully to
# plain ``print`` if rich is not installed so the rest of the system is never
# broken by an optional display dependency.

# Design
# ------
# *  TrainingTable   — stateful Live table; call ``add_result()`` after each
#                      model finishes, ``close()`` when the loop is done.
# *  plain_row()     — single-line fallback if rich is unavailable.
# """

# from __future__ import annotations

# from typing import Any, Dict, Optional

# _RICH_AVAILABLE = False
# try:
#     from rich.console import Console
#     from rich.live import Live
#     from rich.table import Table
#     from rich import box
#     _RICH_AVAILABLE = True
# except ImportError:
#     pass


# # ── colour thresholds ──────────────────────────────────────────────────────
# _METRIC_STYLE = {
#     # classification
#     "Accuracy": [(0.90, "bold green"), (0.75, "yellow"), (0.0, "red")],
#     "f1":       [(0.90, "bold green"), (0.75, "yellow"), (0.0, "red")],
#     "AUC":      [(0.90, "bold green"), (0.75, "yellow"), (0.0, "red")],
#     "AUCPR":    [(0.90, "bold green"), (0.75, "yellow"), (0.0, "red")],
#     "LogLoss":  [(0.20, "bold green"), (0.50, "yellow"), (0.0, "red")],   # lower = better
#     # regression
#     "R2":       [(0.90, "bold green"), (0.70, "yellow"), (0.0, "red")],
#     "MAE":      [(0.10, "bold green"), (0.30, "yellow"), (0.0, "red")],   # lower = better
#     "MSE":      [(0.01, "bold green"), (0.10, "yellow"), (0.0, "red")],
#     "RMSE":     [(0.10, "bold green"), (0.30, "yellow"), (0.0, "red")],
# }
# _LOWER_IS_BETTER = {"LogLoss", "MAE", "MSE", "RMSE"}


# def _style_for(metric: str, value: float) -> str:
#     thresholds = _METRIC_STYLE.get(metric)
#     if thresholds is None:
#         return "white"
#     if metric in _LOWER_IS_BETTER:
#         for threshold, style in thresholds:
#             if value <= threshold:
#                 return style
#     else:
#         for threshold, style in thresholds:
#             if value >= threshold:
#                 return style
#     return "white"


# class TrainingTable:
#     """
#     Live-updating terminal table for model training progress.

#     Usage
#     -----
#     >>> table = TrainingTable(metric_names=["Accuracy", "f1", "AUC"])
#     >>> table.start()
#     >>> table.add_result("RandomForest", {"Accuracy": 0.91, "f1": 0.90, "AUC": 0.95},
#     ...                  status="done", elapsed=3.2)
#     >>> table.add_result("SVC", None, status="failed", elapsed=0.1)
#     >>> table.close()
#     """

#     def __init__(self, metric_names: list[str], total_models: int = 0) -> None:
#         self._metric_names = metric_names
#         self._total = total_models
#         self._rows: list[dict] = []

#         if _RICH_AVAILABLE:
#             self._console = Console()
#             self._live: Optional[Live] = None
#         else:
#             self._console = None
#             self._live = None

#     # ------------------------------------------------------------------
#     # Lifecycle
#     # ------------------------------------------------------------------

#     def start(self) -> "TrainingTable":
#         if _RICH_AVAILABLE:
#             self._live = Live(
#                 self._build_rich_table(),
#                 console=self._console,
#                 refresh_per_second=8,
#                 vertical_overflow="visible",
#             )
#             self._live.start()
#         else:
#             header = "  {:<35s}".format("Model")
#             for m in self._metric_names:
#                 header += f"  {m:>10s}"
#             header += "  {:>8s}  {:>7s}".format("Time(s)", "Status")
#             print("\n" + header)
#             print("  " + "─" * (len(header) - 2))
#         return self

#     def add_result(
#         self,
#         model_name: str,
#         metrics: Optional[Dict[str, float]],
#         status: str = "done",
#         elapsed: float = 0.0,
#     ) -> None:
#         self._rows.append(
#             {"model": model_name, "metrics": metrics or {}, "status": status, "elapsed": elapsed}
#         )
#         if _RICH_AVAILABLE and self._live:
#             self._live.update(self._build_rich_table())
#         else:
#             self._plain_row(model_name, metrics or {}, status, elapsed)

#     def close(self) -> None:
#         if _RICH_AVAILABLE and self._live:
#             self._live.stop()
#         else:
#             total = len(self._rows)
#             done  = sum(1 for r in self._rows if r["status"] == "done")
#             print(f"\n  {done}/{total} models trained successfully.\n")

#     # ------------------------------------------------------------------
#     # Rich rendering
#     # ------------------------------------------------------------------

#     def _build_rich_table(self) -> "Table":
#         table = Table(
#             title=f"[bold cyan]MLTuneX — Model Training[/bold cyan]  "
#                   f"[dim]({len(self._rows)}/{self._total or '?'})[/dim]",
#             box=box.ROUNDED,
#             show_header=True,
#             header_style="bold magenta",
#             border_style="bright_black",
#             expand=False,
#             padding=(0, 1),
#         )
#         table.add_column("#",          style="dim",         width=4,  justify="right")
#         table.add_column("Model",      style="bold white",  min_width=28)
#         for m in self._metric_names:
#             table.add_column(m, justify="right", min_width=max(len(m), 8))
#         table.add_column("Time (s)",   justify="right", style="cyan",   width=9)
#         table.add_column("Status",     justify="center",                width=8)

#         for idx, row in enumerate(self._rows, 1):
#             status_str = (
#                 "[bold green]✓ done[/bold green]"   if row["status"] == "done"
#                 else "[bold red]✗ fail[/bold red]"
#             )
#             cells: list[str] = [str(idx), row["model"]]
#             for m in self._metric_names:
#                 val = row["metrics"].get(m)
#                 if val is None:
#                     cells.append("[dim]—[/dim]")
#                 else:
#                     style = _style_for(m, val)
#                     cells.append(f"[{style}]{val:.4f}[/{style}]")
#             cells.append(f"{row['elapsed']:.1f}")
#             cells.append(status_str)
#             table.add_row(*cells)

#         return table

#     # ------------------------------------------------------------------
#     # Plain-text fallback
#     # ------------------------------------------------------------------

#     def _plain_row(
#         self,
#         model_name: str,
#         metrics: Dict[str, float],
#         status: str,
#         elapsed: float,
#     ) -> None:
#         row = "  {:<35s}".format(model_name)
#         for m in self._metric_names:
#             val = metrics.get(m)
#             row += f"  {val:>10.4f}" if val is not None else f"  {'—':>10s}"
#         row += f"  {elapsed:>8.1f}  {status:>7s}"
#         print(row)

"""
mltunex.reporting.terminal_table
─────────────────────────────────
Live terminal table that updates in-place as each model finishes training.

Environment detection
---------------------
Rich's Live display requires a real TTY.  In Jupyter / Colab / plain scripts
the Live context manager causes:
  - Broken ANSI escape codes in output cells
  - `Exception ignored in sys.unraisablehook` from rich.console._exit_buffer
  - LightGBM [Info] lines interleaved with table output

This module detects the runtime environment and selects the appropriate renderer:

  * Real TTY (terminal)  → Rich Live table (animated, colour-coded)
  * Jupyter / Colab      → Plain print with ipython display (clean, no ANSI)
  * Non-TTY script       → Plain print rows (CI-safe)

Rich is always imported lazily so missing it never crashes the pipeline.
"""
from __future__ import annotations

import io
import os
import sys
from typing import Any, Dict, List, Optional

# ── Environment detection ─────────────────────────────────────────────────────

def _in_jupyter() -> bool:
    """Return True when running inside a Jupyter / Colab / IPython kernel."""
    try:
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
        return shell in ("ZMQInteractiveShell",   # Jupyter notebook / lab
                         "google.colab._shell",   # Colab
                         "TerminalInteractiveShell")
    except NameError:
        return False


def _is_real_tty() -> bool:
    """Return True only when stdout is a real interactive terminal."""
    if _in_jupyter():
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _rich_available() -> bool:
    try:
        import rich  # noqa: F401
        return True
    except ImportError:
        return False


# ── Colour thresholds ─────────────────────────────────────────────────────────
_LOWER_IS_BETTER = {"LogLoss", "MAE", "MSE", "RMSE"}

_METRIC_THRESHOLDS: Dict[str, List] = {
    "Accuracy": [(0.90, "bold green"), (0.75, "yellow"), (0.0, "red")],
    "f1":       [(0.90, "bold green"), (0.75, "yellow"), (0.0, "red")],
    "AUC":      [(0.90, "bold green"), (0.75, "yellow"), (0.0, "red")],
    "AUCPR":    [(0.90, "bold green"), (0.75, "yellow"), (0.0, "red")],
    "LogLoss":  [(0.20, "bold green"), (0.50, "yellow"), (0.0, "red")],
    "R2":       [(0.90, "bold green"), (0.70, "yellow"), (0.0, "red")],
    "MAE":      [(0.10, "bold green"), (0.30, "yellow"), (0.0, "red")],
    "MSE":      [(0.01, "bold green"), (0.10, "yellow"), (0.0, "red")],
    "RMSE":     [(0.10, "bold green"), (0.30, "yellow"), (0.0, "red")],
}


def _style_for(metric: str, value: float) -> str:
    thresholds = _METRIC_THRESHOLDS.get(metric)
    if thresholds is None:
        return "white"
    lower = metric in _LOWER_IS_BETTER
    for threshold, style in thresholds:
        if lower and value <= threshold:
            return style
        if not lower and value >= threshold:
            return style
    return "white"


# ── Main class ────────────────────────────────────────────────────────────────

class TrainingTable:
    """
    Training progress table with automatic environment adaptation.

    Usage
    -----
    >>> table = TrainingTable(metric_names=["Accuracy", "f1", "AUC"], total_models=5)
    >>> table.start()
    >>> table.add_result("RandomForest", {"Accuracy": 0.91, "f1": 0.90}, elapsed=2.1)
    >>> table.add_result("SVC", None, status="failed", elapsed=0.1)
    >>> table.close()
    """

    def __init__(
        self,
        metric_names: List[str],
        total_models: int = 0,
    ) -> None:
        self._metric_names  = metric_names
        self._total         = total_models
        self._rows: List[Dict[str, Any]] = []

        # Choose renderer at construction time — stable for the life of the table
        self._use_rich = _is_real_tty() and _rich_available()
        self._use_jupyter = _in_jupyter()

        self._live   = None   # Rich Live instance (TTY only)
        self._console = None  # Rich Console instance (TTY only)

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> "TrainingTable":
        if self._use_rich:
            self._start_rich()
        else:
            self._start_plain()
        return self

    def add_result(
        self,
        model_name: str,
        metrics: Optional[Dict[str, float]],
        status: str = "done",
        elapsed: float = 0.0,
    ) -> None:
        self._rows.append({
            "model":   model_name,
            "metrics": metrics or {},
            "status":  status,
            "elapsed": elapsed,
        })
        if self._use_rich and self._live is not None:
            try:
                self._live.update(self._build_rich_table())
            except Exception:
                # Rich Live failed mid-way — fall back to plain for remaining rows
                self._use_rich = False
                self._close_rich()
                self._plain_row(model_name, metrics or {}, status, elapsed)
        else:
            self._plain_row(model_name, metrics or {}, status, elapsed)

    def close(self) -> None:
        if self._use_rich and self._live is not None:
            self._close_rich()
        else:
            done  = sum(1 for r in self._rows if r["status"] == "done")
            total = len(self._rows)
            msg   = f"\n[MLTuneX] {done}/{total} models trained successfully.\n"
            if self._use_jupyter:
                print(msg)
            else:
                print(msg)

    # ── Rich rendering ────────────────────────────────────────────────

    def _start_rich(self) -> None:
        try:
            from rich.console import Console
            from rich.live import Live
            # Force_terminal prevents Rich from disabling colours on non-TTY
            # file= must be the real stdout, not any captured stream
            self._console = Console(highlight=False, force_terminal=True)
            self._live = Live(
                self._build_rich_table(),
                console=self._console,
                refresh_per_second=8,
                vertical_overflow="visible",
            )
            self._live.__enter__()
        except Exception:
            # Any Rich initialisation failure → fall through to plain
            self._use_rich = False
            self._live = None
            self._start_plain()

    def _close_rich(self) -> None:
        try:
            if self._live is not None:
                self._live.__exit__(None, None, None)
        except Exception:
            pass
        self._live    = None
        self._console = None

    def _build_rich_table(self):
        from rich.table import Table
        from rich import box

        table = Table(
            title=(
                f"[bold cyan]MLTuneX — Model Training[/bold cyan]  "
                f"[dim]({len(self._rows)}/{self._total or '?'})[/dim]"
            ),
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
            border_style="bright_black",
            expand=False,
            padding=(0, 1),
        )
        table.add_column("#",         style="dim",        width=4,  justify="right")
        table.add_column("Model",     style="bold white", min_width=28)
        for m in self._metric_names:
            table.add_column(m, justify="right", min_width=max(len(m), 8))
        table.add_column("Time (s)", justify="right", style="cyan",  width=9)
        table.add_column("Status",   justify="center",               width=8)

        for idx, row in enumerate(self._rows, 1):
            status_str = (
                "[bold green]done[/bold green]"
                if row["status"] == "done"
                else "[bold red]fail[/bold red]"
            )
            cells = [str(idx), row["model"]]
            for m in self._metric_names:
                val = row["metrics"].get(m)
                if val is None:
                    cells.append("[dim]-[/dim]")
                else:
                    s = _style_for(m, val)
                    cells.append(f"[{s}]{val:.4f}[/{s}]")
            cells.append(f"{row['elapsed']:.1f}")
            cells.append(status_str)
            table.add_row(*cells)

        return table

    # ── Plain rendering ───────────────────────────────────────────────

    def _start_plain(self) -> None:
        """Print a plain-text table header."""
        header = f"  {'Model':<35s}"
        for m in self._metric_names:
            header += f"  {m:>10s}"
        header += f"  {'Time(s)':>8s}  {'Status':>7s}"
        sep = "  " + "-" * (len(header) - 2)

        if self._use_jupyter:
            # In Jupyter, flush immediately so the header appears before rows
            print(f"\n{header}")
            print(sep)
            sys.stdout.flush()
        else:
            print(f"\n{header}")
            print(sep)

    def _plain_row(
        self,
        model_name: str,
        metrics: Dict[str, float],
        status: str,
        elapsed: float,
    ) -> None:
        row = f"  {model_name:<35s}"
        for m in self._metric_names:
            val = metrics.get(m)
            row += f"  {val:>10.4f}" if val is not None else f"  {'--':>10s}"
        row += f"  {elapsed:>8.1f}  {status:>7s}"
        print(row)
        if self._use_jupyter:
            sys.stdout.flush()