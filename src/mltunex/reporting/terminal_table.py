"""
mltunex.reporting.terminal_table
─────────────────────────────────
Live terminal table that updates in-place as each model finishes training.

Uses ``rich`` for a polished, YOLO-style output.  Falls back gracefully to
plain ``print`` if rich is not installed so the rest of the system is never
broken by an optional display dependency.

Design
------
*  TrainingTable   — stateful Live table; call ``add_result()`` after each
                     model finishes, ``close()`` when the loop is done.
*  plain_row()     — single-line fallback if rich is unavailable.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

_RICH_AVAILABLE = False
try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich import box
    _RICH_AVAILABLE = True
except ImportError:
    pass


# ── colour thresholds ──────────────────────────────────────────────────────
_METRIC_STYLE = {
    # classification
    "Accuracy": [(0.90, "bold green"), (0.75, "yellow"), (0.0, "red")],
    "f1":       [(0.90, "bold green"), (0.75, "yellow"), (0.0, "red")],
    "AUC":      [(0.90, "bold green"), (0.75, "yellow"), (0.0, "red")],
    "AUCPR":    [(0.90, "bold green"), (0.75, "yellow"), (0.0, "red")],
    "LogLoss":  [(0.20, "bold green"), (0.50, "yellow"), (0.0, "red")],   # lower = better
    # regression
    "R2":       [(0.90, "bold green"), (0.70, "yellow"), (0.0, "red")],
    "MAE":      [(0.10, "bold green"), (0.30, "yellow"), (0.0, "red")],   # lower = better
    "MSE":      [(0.01, "bold green"), (0.10, "yellow"), (0.0, "red")],
    "RMSE":     [(0.10, "bold green"), (0.30, "yellow"), (0.0, "red")],
}
_LOWER_IS_BETTER = {"LogLoss", "MAE", "MSE", "RMSE"}


def _style_for(metric: str, value: float) -> str:
    thresholds = _METRIC_STYLE.get(metric)
    if thresholds is None:
        return "white"
    if metric in _LOWER_IS_BETTER:
        for threshold, style in thresholds:
            if value <= threshold:
                return style
    else:
        for threshold, style in thresholds:
            if value >= threshold:
                return style
    return "white"


class TrainingTable:
    """
    Live-updating terminal table for model training progress.

    Usage
    -----
    >>> table = TrainingTable(metric_names=["Accuracy", "f1", "AUC"])
    >>> table.start()
    >>> table.add_result("RandomForest", {"Accuracy": 0.91, "f1": 0.90, "AUC": 0.95},
    ...                  status="done", elapsed=3.2)
    >>> table.add_result("SVC", None, status="failed", elapsed=0.1)
    >>> table.close()
    """

    def __init__(self, metric_names: list[str], total_models: int = 0) -> None:
        self._metric_names = metric_names
        self._total = total_models
        self._rows: list[dict] = []

        if _RICH_AVAILABLE:
            self._console = Console()
            self._live: Optional[Live] = None
        else:
            self._console = None
            self._live = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "TrainingTable":
        if _RICH_AVAILABLE:
            self._live = Live(
                self._build_rich_table(),
                console=self._console,
                refresh_per_second=8,
                vertical_overflow="visible",
            )
            self._live.start()
        else:
            header = "  {:<35s}".format("Model")
            for m in self._metric_names:
                header += f"  {m:>10s}"
            header += "  {:>8s}  {:>7s}".format("Time(s)", "Status")
            print("\n" + header)
            print("  " + "─" * (len(header) - 2))
        return self

    def add_result(
        self,
        model_name: str,
        metrics: Optional[Dict[str, float]],
        status: str = "done",
        elapsed: float = 0.0,
    ) -> None:
        self._rows.append(
            {"model": model_name, "metrics": metrics or {}, "status": status, "elapsed": elapsed}
        )
        if _RICH_AVAILABLE and self._live:
            self._live.update(self._build_rich_table())
        else:
            self._plain_row(model_name, metrics or {}, status, elapsed)

    def close(self) -> None:
        if _RICH_AVAILABLE and self._live:
            self._live.stop()
        else:
            total = len(self._rows)
            done  = sum(1 for r in self._rows if r["status"] == "done")
            print(f"\n  {done}/{total} models trained successfully.\n")

    # ------------------------------------------------------------------
    # Rich rendering
    # ------------------------------------------------------------------

    def _build_rich_table(self) -> "Table":
        table = Table(
            title=f"[bold cyan]MLTuneX — Model Training[/bold cyan]  "
                  f"[dim]({len(self._rows)}/{self._total or '?'})[/dim]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
            border_style="bright_black",
            expand=False,
            padding=(0, 1),
        )
        table.add_column("#",          style="dim",         width=4,  justify="right")
        table.add_column("Model",      style="bold white",  min_width=28)
        for m in self._metric_names:
            table.add_column(m, justify="right", min_width=max(len(m), 8))
        table.add_column("Time (s)",   justify="right", style="cyan",   width=9)
        table.add_column("Status",     justify="center",                width=8)

        for idx, row in enumerate(self._rows, 1):
            status_str = (
                "[bold green]✓ done[/bold green]"   if row["status"] == "done"
                else "[bold red]✗ fail[/bold red]"
            )
            cells: list[str] = [str(idx), row["model"]]
            for m in self._metric_names:
                val = row["metrics"].get(m)
                if val is None:
                    cells.append("[dim]—[/dim]")
                else:
                    style = _style_for(m, val)
                    cells.append(f"[{style}]{val:.4f}[/{style}]")
            cells.append(f"{row['elapsed']:.1f}")
            cells.append(status_str)
            table.add_row(*cells)

        return table

    # ------------------------------------------------------------------
    # Plain-text fallback
    # ------------------------------------------------------------------

    def _plain_row(
        self,
        model_name: str,
        metrics: Dict[str, float],
        status: str,
        elapsed: float,
    ) -> None:
        row = "  {:<35s}".format(model_name)
        for m in self._metric_names:
            val = metrics.get(m)
            row += f"  {val:>10.4f}" if val is not None else f"  {'—':>10s}"
        row += f"  {elapsed:>8.1f}  {status:>7s}"
        print(row)
