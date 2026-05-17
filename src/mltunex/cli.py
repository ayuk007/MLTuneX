"""
mltunex.cli
────────────
Command-line interface for MLTuneX.

Usage
-----
    python -m mltunex.cli --data titanic.csv --target Survived --task classification \\
        --llm "Groq:qwen/qwen3-32b"

    # Regression with parallel training and custom log dir
    python -m mltunex.cli --data housing.parquet --target price --task regression \\
        --llm "OpenAI:gpt-4o" --parallel --jobs 4 --trials 50 --log-dir runs/

All arguments are documented in ``mltunex --help``.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import traceback


# ── Banner ──────────────────────────────────────────────────────────────────

_BANNER = r"""
 ███╗   ███╗██╗  ████████╗██╗   ██╗███╗   ██╗███████╗██╗  ██╗
 ████╗ ████║██║  ╚══██╔══╝██║   ██║████╗  ██║██╔════╝╚██╗██╔╝
 ██╔████╔██║██║     ██║   ██║   ██║██╔██╗ ██║█████╗   ╚███╔╝
 ██║╚██╔╝██║██║     ██║   ██║   ██║██║╚██╗██║██╔══╝   ██╔██╗
 ██║ ╚═╝ ██║███████╗██║   ╚██████╔╝██║ ╚████║███████╗██╔╝ ██╗
 ╚═╝     ╚═╝╚══════╝╚═╝    ╚═════╝ ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝
 Automated Machine Learning Fine-Tuning System
"""


def _print_banner() -> None:
    try:
        from rich.console import Console
        from rich.text import Text
        console = Console()
        console.print(f"[bold cyan]{_BANNER}[/bold cyan]")
    except ImportError:
        print(_BANNER)


# ── Argument parser ──────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mltunex",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            MLTuneX — Automated Machine Learning Fine-Tuning System
            ─────────────────────────────────────────────────────────
            Train multiple models, select top candidates, and optimise
            hyperparameters with AI guidance — all from the terminal.

            Examples
            --------
            # Minimal: classification with defaults
            mltunex --data titanic.csv --target Survived --task classification \\
                    --llm "Groq:qwen/qwen3-32b"

            # Regression, parallel, 50 trials, custom output dirs
            mltunex --data housing.csv --target SalePrice --task regression \\
                    --llm "OpenAI:gpt-4o" --parallel --jobs 4 --trials 50 \\
                    --results results/ --models models/ --log-dir logs/

            # Skip tuning, use stability-aware selector
            mltunex --data data.csv --target label --task classification \\
                    --llm "Groq:qwen/qwen3-32b" --no-tune \\
                    --selector stability --top-k 5
        """),
    )

    # ── Required ──────────────────────────────────────────────────────
    req = parser.add_argument_group("required arguments")
    req.add_argument(
        "--data", "-d",
        required=True,
        metavar="PATH",
        help="Path to the dataset file (.csv, .xlsx, .parquet, .feather).",
    )
    req.add_argument(
        "--target", "-t",
        required=True,
        metavar="COLUMN",
        help="Name of the target column.",
    )
    req.add_argument(
        "--task",
        required=True,
        choices=["classification", "regression"],
        help="Machine learning task type.",
    )
    req.add_argument(
        "--llm",
        required=False,
        default=None,
        metavar="PROVIDER:MODEL",
        help=(
            "LLM for AI-guided hyperparameter advice. "
            "Format: Provider:ModelName  (e.g. 'Groq:qwen/qwen3-32b', "
            "'OpenAI:gpt-4o'). "
            "Required unless --no-tune is set."
        ),
    )

    # ── Training ──────────────────────────────────────────────────────
    train = parser.add_argument_group("training options")
    train.add_argument(
        "--test-size", type=float, default=0.2, metavar="FLOAT",
        help="Fraction of data held out for evaluation (default: 0.2).",
    )
    train.add_argument(
        "--no-preprocess", action="store_true",
        help="Skip the adaptive preprocessing pipeline.",
    )
    train.add_argument(
        "--parallel", action="store_true",
        help="Train models in parallel using multiprocessing.",
    )
    train.add_argument(
        "--jobs", "-j", type=int, default=-1, metavar="N",
        help="Worker count for parallel training (-1 = all CPUs, default: -1).",
    )
    train.add_argument(
        "--library", default="sklearn", metavar="LIB",
        help="Model library backend (default: sklearn).",
    )

    # ── Tuning ────────────────────────────────────────────────────────
    tune = parser.add_argument_group("tuning options")
    tune.add_argument(
        "--no-tune", action="store_true",
        help="Skip AI-guided hyperparameter optimisation.",
    )
    tune.add_argument(
        "--trials", type=int, default=25, metavar="N",
        help="Number of Optuna trials (default: 25).",
    )
    tune.add_argument(
        "--optimizer", default="optuna", metavar="NAME",
        help="Optimizer backend (default: optuna).",
    )

    # ── Selection ─────────────────────────────────────────────────────
    sel = parser.add_argument_group("model selection options")
    sel.add_argument(
        "--selector", default="topk",
        choices=["topk", "stability", "generalization"],
        help="Model selection strategy (default: topk).",
    )
    sel.add_argument(
        "--top-k", type=int, default=3, metavar="N",
        help="Number of candidate models forwarded to the optimizer (default: 3).",
    )
    sel.add_argument(
        "--primary-metric", metavar="METRIC",
        help=(
            "Override the primary ranking metric. "
            "Defaults to 'Accuracy' (classification) or 'R2' (regression)."
        ),
    )
    sel.add_argument(
        "--stability-weight", type=float, default=0.2, metavar="FLOAT",
        help="[stability selector] Instability penalty weight (default: 0.2).",
    )
    sel.add_argument(
        "--train-metric", metavar="COLUMN",
        help="[generalization selector] Train-set metric column name.",
    )
    sel.add_argument(
        "--gap-penalty", type=float, default=0.5, metavar="FLOAT",
        help="[generalization selector] Train/test gap penalty weight (default: 0.5).",
    )
    sel.add_argument(
        "--profiling", default="extended",
        choices=["basic", "extended"],
        help="Data profiling strategy (default: extended).",
    )

    # ── Output ────────────────────────────────────────────────────────
    out = parser.add_argument_group("output options")
    out.add_argument(
        "--results", default="results/", metavar="DIR",
        help="Directory for evaluation CSV (default: results/).",
    )
    out.add_argument(
        "--models", default="models/", metavar="DIR",
        help="Directory for saved model artefact (default: models/).",
    )
    out.add_argument(
        "--log-dir", default="logs/", metavar="DIR",
        help="Root directory for experiment logs (default: logs/).",
    )
    out.add_argument(
        "--exp-name", default="exp", metavar="NAME",
        help="Tag prepended to the experiment log folder (default: exp).",
    )

    return parser


# ── Validation ───────────────────────────────────────────────────────────────

def _validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Raise parser errors for invalid argument combinations."""
    import os

    if not os.path.isfile(args.data):
        parser.error(
            f"Data file not found: '{args.data}'\n"
            f"  Tip: use an absolute path or check your working directory."
        )

    if not 0.0 < args.test_size < 1.0:
        parser.error(f"--test-size must be between 0 and 1, got {args.test_size}.")

    if args.trials < 1:
        parser.error(f"--trials must be ≥ 1, got {args.trials}.")

    if args.top_k < 1:
        parser.error(f"--top-k must be ≥ 1, got {args.top_k}.")

    if not args.no_tune:
        if not args.llm:
            parser.error(
                "--llm is required when AI tuning is enabled.\n"
                "  Provide: --llm 'Groq:qwen/qwen3-32b'\n"
                "  Or skip tuning with: --no-tune"
            )
        if ":" not in args.llm:
            parser.error(
                f"--llm must be in 'Provider:ModelName' format, got '{args.llm}'.\n"
                f"  Examples: 'Groq:qwen/qwen3-32b'  or  'OpenAI:gpt-4o'"
            )

    if args.selector == "generalization" and not args.train_metric:
        parser.error(
            "--train-metric is required when using the 'generalization' selector.\n"
            "  Provide the name of the train-set metric column in your results."
        )


# ── Main ─────────────────────────────────────────────────────────────────────

def _run_ui(argv: list[str] | None = None) -> int:
    """Launch the Streamlit UI."""
    import subprocess, sys as _sys, os as _os
    app_path = _os.path.join(_os.path.dirname(__file__), "ui", "app.py")
    cmd = [_sys.executable, "-m", "streamlit", "run", app_path]
    if argv:
        cmd += argv
    try:
        result = subprocess.run(cmd)
        return result.returncode
    except FileNotFoundError:
        print("Streamlit not found. Install it with:  pip install streamlit")
        return 1


def main(argv: list[str] | None = None) -> int:
    # ── Check for 'ui' subcommand before building the full parser ─────────
    raw = argv if argv is not None else sys.argv[1:]
    if raw and raw[0] == "ui":
        _print_banner()
        return _run_ui(raw[1:])

    parser = _build_parser()
    args   = parser.parse_args(argv)
    _validate(args, parser)

    _print_banner()

    try:
        from mltunex.orchestrator import MLTuneXOrchestrator, OrchestratorConfig

        config = OrchestratorConfig(
            source                    = args.data,
            target_column             = args.target,
            task_type                 = args.task,
            model_provider_model_name = args.llm or "Groq:none",

            result_csv_path           = args.results,
            model_dir_path            = args.models,
            log_dir                   = args.log_dir,
            experiment_name           = args.exp_name,

            test_size                 = args.test_size,
            preprocess                = not args.no_preprocess,
            parallel_training         = args.parallel,
            n_jobs                    = args.jobs,

            tune_models               = not args.no_tune,
            optimizer_method          = args.optimizer,
            n_trials                  = args.trials,

            profiling_strategy        = args.profiling,
            models_library            = args.library,
            selection_strategy        = args.selector,
            top_k                     = args.top_k,

            selector_primary_metric   = args.primary_metric,
            selector_stability_weight = args.stability_weight,
            selector_train_metric     = args.train_metric,
            selector_gap_penalty      = args.gap_penalty,
        )

        MLTuneXOrchestrator(config).run()
        return 0

    except KeyboardInterrupt:
        print("\n\n[MLTuneX] Interrupted by user.")
        return 130

    except Exception as exc:
        print(f"\n[MLTuneX] ✘ Fatal error: {exc}")
        print("\nFull traceback:")
        traceback.print_exc()
        print(
            "\nIf this looks like a bug, please open an issue at "
            "https://github.com/ayuk007/MLTuneX with the traceback above."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())