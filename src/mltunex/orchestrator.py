# """
# MLTuneX Orchestrator — central workflow coordinator (Facade Pattern).
# """

# from __future__ import annotations

# import io
# import logging
# import os
# import sys
# import time
# import warnings
# import multiprocessing as mp
# from contextlib import contextmanager
# from dataclasses import dataclass
# from typing import Any, Dict, List, Optional, Tuple, Union

# import pandas as pd

# # Suppress C-level library noise (LightGBM, XGBoost) before any imports
# os.environ.setdefault("LIGHTGBM_VERBOSITY", "-1")
# os.environ.setdefault("LIGHTGBM_VERBOSITY", "-1")
# os.environ.setdefault("VERBOSITY", "0")

# from mltunex.data.sources import DataSourceFactory
# from mltunex.data.profiler import DataProfilerFactory
# from mltunex.data.splitter import Data_Splitter
# from mltunex.preprocessing import AdaptivePipelineDirector
# from mltunex.model_registry.model_registry import Model_Registry
# from mltunex.model_registry.selector import ModelSelectorFactory, SelectorConfig
# from mltunex.library_trainer.library_trainer import LibraryTrainer
# from mltunex.evaluate.evaluator import EvaluatorFactory
# from mltunex.hyperparam_tuner.optimizer import (
#     LLMAdvisorAdapter,
#     Optimizer,
#     OptimizerFactory,
# )
# from mltunex.ai_handler.llm_manager.llm_manager import LLMManager
# from mltunex.utils.model_utils import ModelUtils
# from mltunex.reporting.terminal_table import TrainingTable
# from mltunex.reporting.experiment_logger import ExperimentLogger
# from mltunex.reporting.pipeline_logger import PipelineLogger


# # ─────────────────────────────────────────────────────────────────────────────
# # Silence helper — routes all 3rd-party noise to the file log only
# # ─────────────────────────────────────────────────────────────────────────────

# @contextmanager
# def _silent():
#     """
#     Suppress all terminal output from 3rd-party libraries.

#     Three layers:
#       1. Python warnings module
#       2. Python logging (sklearn, optuna, lightgbm, xgboost, ...)
#       3. C-level stdout/stderr (LightGBM "[LightGBM] [Warning]" etc.)
#          via fd-level redirect to /dev/null.
#     """
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")

#         noisy = ["optuna", "sklearn", "lightgbm", "xgboost", "catboost",
#                  "urllib3", "httpx", "httpcore"]
#         old_levels = {}
#         for name in noisy:
#             lg = logging.getLogger(name)
#             old_levels[name] = lg.level
#             lg.setLevel(logging.CRITICAL)

#         # Redirect file descriptors 1 (stdout) and 2 (stderr) to /dev/null
#         # This catches C-extension output that bypasses Python's sys.stdout
#         devnull_fd = os.open(os.devnull, os.O_WRONLY)
#         saved_fds: Dict[int, int] = {}
#         for fd in (1, 2):
#             try:
#                 saved_fds[fd] = os.dup(fd)
#                 os.dup2(devnull_fd, fd)
#             except Exception:
#                 pass

#         try:
#             yield
#         finally:
#             for fd, saved in saved_fds.items():
#                 try:
#                     os.dup2(saved, fd)
#                     os.close(saved)
#                 except Exception:
#                     pass
#             try:
#                 os.close(devnull_fd)
#             except Exception:
#                 pass
#             for name, level in old_levels.items():
#                 logging.getLogger(name).setLevel(level)


# # ─────────────────────────────────────────────────────────────────────────────
# # Configuration
# # ─────────────────────────────────────────────────────────────────────────────

# @dataclass
# class OrchestratorConfig:
#     """All configuration for a single MLTuneX run."""

#     # Required
#     source: Union[str, pd.DataFrame]
#     target_column: str
#     task_type: str
#     model_provider_model_name: str

#     # I/O
#     result_csv_path: str = "results/"
#     model_dir_path: str  = "models/"
#     log_dir: str         = "logs/"
#     experiment_name: str = "exp"

#     # Training
#     test_size: float        = 0.2
#     preprocess: bool        = True
#     parallel_training: bool = False
#     n_jobs: int             = -1     # -1 = all CPUs

#     # Tuning
#     tune_models: bool             = True
#     optimizer_method: str         = "optuna"
#     n_trials: int                 = 25
#     hyperparameter_framework: str = "Optuna"

#     # Profiling / selection
#     profiling_strategy: str       = "extended"
#     models_library: str           = "sklearn"
#     selection_strategy: str       = "topk"
#     top_k: int                    = 3

#     selector_primary_metric: Optional[str] = None
#     selector_stability_weight: float       = 0.2
#     selector_train_metric: Optional[str]   = None
#     selector_gap_penalty: float            = 0.5


# # ─────────────────────────────────────────────────────────────────────────────
# # Module-level worker (must be top-level for multiprocessing pickle)
# # ─────────────────────────────────────────────────────────────────────────────

# def _train_single_model_worker(
#     args: Tuple[str, Any, Any, Any, Any, Any, str],
# ) -> Tuple[str, Any, Dict[str, float], float, str]:
#     """
#     Train + evaluate one model in an isolated worker process.

#     Critical: the worker receives x_test AND y_test so it can produce
#     real metrics.  Previously y_test was not passed → all NaN metrics.
#     """
#     model_name, estimator_class, x_train, y_train, x_test, y_test, task_type = args

#     # Silence ALL output inside the worker process
#     import warnings as _w
#     _w.filterwarnings("ignore")
#     import logging as _l
#     for _n in ["optuna", "sklearn", "lightgbm", "xgboost", "catboost"]:
#         _l.getLogger(_n).setLevel(_l.CRITICAL)

#     trainer   = LibraryTrainer.get_trainer("sklearn")
#     evaluator = EvaluatorFactory.create(task_type)

#     t0 = time.perf_counter()
#     try:
#         model = trainer.train_model(
#             model=estimator_class,
#             X_train=x_train,
#             y_train=y_train,
#             task_type=task_type,
#         )
#         if model is None:
#             return model_name, None, {}, time.perf_counter() - t0, "failed"

#         # Pass y_test so the evaluator can actually compute metrics
#         result  = evaluator.evaluate(model_name, model, x_test, y_test)
#         metrics = result.get(model_name) or {}
#         # Strip NaN values so table renders cleanly
#         metrics = {k: v for k, v in metrics.items() if v == v}  # NaN != NaN
#         return model_name, model, metrics, time.perf_counter() - t0, "done"

#     except Exception as exc:
#         return model_name, None, {}, time.perf_counter() - t0, f"failed: {exc}"


# # ─────────────────────────────────────────────────────────────────────────────
# # Orchestrator
# # ─────────────────────────────────────────────────────────────────────────────

# class MLTuneXOrchestrator:
#     """Facade that coordinates the complete MLTuneX AutoML workflow."""

#     def __init__(self, config: OrchestratorConfig) -> None:
#         self._cfg      = config
#         self._x_train: Optional[pd.DataFrame] = None
#         self._x_test:  Optional[pd.DataFrame] = None
#         self._y_train: Optional[pd.Series]    = None
#         self._y_test:  Optional[pd.Series]    = None
#         self._pipeline = None
#         self._logger:  Optional[ExperimentLogger]  = None
#         self._plog:    Optional[PipelineLogger]     = None
#         # Optional callback for Streamlit / external progress tracking
#         self._progress_callback = None

#     # ── Public API ────────────────────────────────────────────────────

#     def set_progress_callback(self, fn) -> None:
#         """
#         Register a callable that receives progress events.

#         Signature: fn(stage: str, payload: dict) -> None
#         Used by the Streamlit UI to update live widgets.
#         """
#         self._progress_callback = fn

#     def _emit(self, stage: str, payload: dict) -> None:
#         if self._progress_callback:
#             self._progress_callback(stage, payload)

#     def run(self) -> None:
#         """Execute the complete AutoML workflow."""
#         cfg = self._cfg

#         # ── Initialise loggers ────────────────────────────────────────
#         self._logger = ExperimentLogger(
#             base_log_dir=cfg.log_dir,
#             experiment_name=cfg.experiment_name,
#         )
#         # Override model save dir to be inside the experiment folder
#         # so every artefact for a run lives together.
#         self._model_dir = os.path.join(
#             self._logger.experiment_dir, "models"
#         )
#         os.makedirs(self._model_dir, exist_ok=True)
#         self._plog = PipelineLogger(
#             log_dir=self._logger.experiment_dir,
#             experiment_name=cfg.experiment_name,
#         )
#         self._plog.info("Run started", task=cfg.task_type,
#                         source=str(cfg.source)[:120])

#         try:
#             self._run_pipeline()
#         finally:
#             self._plog.close()

#     def _run_pipeline(self) -> None:
#         cfg = self._cfg
#         pl  = self._plog

#         # [1] Load
#         print("\n[1/8] Loading data ...")
#         with pl.stage("Load Data"):
#             df = self._load_data()
#             pl.log_dataframe_shape("Loaded dataframe", df)
#         self._emit("data_loaded", {"rows": len(df), "cols": len(df.columns)})

#         # [2] Profile
#         print("[2/8] Profiling data ...")
#         with pl.stage("Profile Data"):
#             profile = self._profile_data(df)
#             pl.log_dict("Profile", {k: v for k, v in profile.items()
#                                     if not isinstance(v, dict)})
#         self._emit("profiled", {"profile": profile})

#         # [3] Split
#         print("[3/8] Splitting data ...")
#         with pl.stage("Split Data"):
#             X = df.drop(columns=[cfg.target_column])
#             y = df[cfg.target_column]
#             splitter = Data_Splitter()
#             self._x_train, self._x_test, self._y_train, self._y_test = (
#                 splitter.split_data(X, y, test_size=cfg.test_size)
#             )
#             pl.info(f"Train: {len(self._x_train)} rows  Test: {len(self._x_test)} rows")

#         # [4] Preprocess
#         if cfg.preprocess:
#             print("[4/8] Building and applying preprocessing pipeline ...")
#             with pl.stage("Preprocess"):
#                 self._x_train, self._x_test = self._preprocess(profile)
#                 steps = self._pipeline.steps if self._pipeline else []
#                 pl.info(f"Pipeline steps: {[s for s, _ in steps]}")
#         else:
#             print("[4/8] Preprocessing skipped (preprocess=False).")
#             pl.info("Preprocessing skipped")

#         # Save the fitted pipeline for inference reuse
#         if cfg.preprocess:
#             self._save_pipeline()

#         # Log preprocessing artefact
#         pipeline_steps = self._pipeline.steps if self._pipeline else []
#         self._logger.log_preprocessing(
#             profile=profile,
#             pipeline_steps=pipeline_steps,
#             profiling_strategy=cfg.profiling_strategy,
#             preprocessed_train=self._x_train,
#             preprocessed_test=self._x_test,
#         )

#         # [5] Train
#         # Count models so UI can show progress bar total
#         _registry_preview = Model_Registry.get_model_registry(cfg.models_library)
#         _n_total = len(_registry_preview.get_models(task_type=cfg.task_type))
#         self._emit("n_models_total", {"total": _n_total})

#         mode = "parallel" if cfg.parallel_training else "sequential"
#         print(f"\n[5/8] Training initial models ({mode}) ...\n")
#         with pl.stage(f"Train Models ({mode})"):
#             trained_dict, all_eval_results = self._train_models()
#             pl.info(f"Trained {len(trained_dict)} models successfully")
#         # [6] Metrics
#         print("\n[6/8] Saving model metrics ...")
#         with pl.stage("Save Metrics"):
#             self._logger.log_model_metrics(
#                 all_results=all_eval_results,
#                 task_type=cfg.task_type,
#             )
#             evaluation_df = ModelUtils.save_results(
#                 evaluation_results=all_eval_results,
#                 evaluation_results_path=cfg.result_csv_path,
#             )
#             pl.log_dataframe_shape("Evaluation DataFrame", evaluation_df)
#         self._emit("metrics_ready", {"eval_df": evaluation_df})

#         if not cfg.tune_models:
#             print("[7/8] Hyperparameter tuning skipped (tune_models=False).")
#             # Still select and save top models so the user gets useful output
#             print("      Selecting and saving top models ...")
#             with pl.stage("Select Top Models (no-tune)"):
#                 selector_cfg, top_models_df = self._select_top_models(evaluation_df)
#                 pl.info(f"Selected: {top_models_df['Model'].tolist()}")
#                 self._logger.log_selection(
#                     all_results_df=evaluation_df,
#                     selected_df=top_models_df,
#                     selection_strategy=cfg.selection_strategy,
#                     selector_config=selector_cfg,
#                 )
#             self._emit("selection_done", {"top_models": top_models_df})

#             # Save each top model to disk
#             saved_paths = []
#             for _, row in top_models_df.iterrows():
#                 model_name = row["Model"]
#                 if model_name in trained_dict:
#                     _, model_obj = trained_dict[model_name]
#                     self._save_model(model_name, model_obj)
#                     saved_paths.append(
#                         os.path.abspath(
#                             os.path.join(cfg.model_dir_path, f"{model_name}.joblib")
#                         )
#                     )
#                     pl.info(f"Saved model: {model_name}")

#             print("\n[MLTuneX] ---- Saved Models ----")
#             for p in saved_paths:
#                 print(f"  {p}")
#             print("[MLTuneX] -------------------------\n")
#             self._emit("run_summary", {
#                 "saved_models": saved_paths,
#                 "pipeline_path": os.path.abspath(
#                     os.path.join(getattr(self,"_model_dir",cfg.model_dir_path), "preprocessing_pipeline.joblib")
#                 ) if cfg.preprocess else None,
#                 "log_dir": self._logger.experiment_dir,
#             })
#             print("[8/8] Done.\n")
#             pl.info("Pipeline completed (no tuning)")
#             return

#         # [7] Select
#         print("[7/8] Selecting top candidates ...")
#         with pl.stage("Select Top Models"):
#             selector_cfg, top_models_df = self._select_top_models(evaluation_df)
#             pl.info(f"Selected: {top_models_df['Model'].tolist()}")
#             self._logger.log_selection(
#                 all_results_df=evaluation_df,
#                 selected_df=top_models_df,
#                 selection_strategy=cfg.selection_strategy,
#                 selector_config=selector_cfg,
#             )
#         self._emit("selection_done", {"top_models": top_models_df})

#         # [8] Optimise
#         print("[8/8] Running AI-guided hyperparameter optimisation ...")
#         with pl.stage("Hyperparameter Optimisation"):
#             optimizer, best_model_name, best_params = self._optimize(
#                 trained_dict, top_models_df, profile
#             )
#             pl.info(f"Best model: {best_model_name}  params: {best_params}")

#         print(f"\n[MLTuneX] Best model : {best_model_name}")
#         print(f"  Best params : {best_params}\n")

#         self._logger.log_tuning(
#             ai_suggestions=getattr(optimizer, "_last_search_spaces", []),
#             trial_history=getattr(optimizer, "trial_history", []),
#             best_model_name=best_model_name,
#             best_params=best_params,
#             best_score=getattr(optimizer, "best_score", 0.0),
#             n_trials=cfg.n_trials,
#             optimizer_method=cfg.optimizer_method,
#             token_usage=getattr(optimizer, "_advisor_token_usage", None),
#         )
#         self._emit("tuning_done", {
#             "best_model": best_model_name,
#             "best_params": best_params,
#             "best_score": getattr(optimizer, "best_score", 0.0),
#             "trial_history": getattr(optimizer, "trial_history", []),
#         })

#         print("Retraining best model with optimal parameters ...")
#         with pl.stage("Retrain Best Model"):
#             final_model = self._retrain_best(best_model_name, best_params)

#         print("Saving final model ...")
#         self._save_model(best_model_name, final_model)
#         final_model_path = os.path.abspath(
#             os.path.join(getattr(self,"_model_dir",cfg.model_dir_path), f"{best_model_name}.joblib")
#         )
#         pipeline_path = os.path.abspath(
#             os.path.join(cfg.model_dir_path, "preprocessing_pipeline.joblib")
#         ) if cfg.preprocess else None

#         print("\n[MLTuneX] ========== Run Summary ==========")
#         print(f"  Best model   : {best_model_name}")
#         print(f"  Best score   : {getattr(optimizer, 'best_score', 0.0):.5f}")
#         print(f"  Model saved  : {final_model_path}")
#         if pipeline_path and os.path.exists(pipeline_path):
#             print(f"  Pipeline     : {pipeline_path}")
#         print(f"  Logs & reports: {self._logger.experiment_dir}")
#         print("[MLTuneX] ====================================\n")

#         self._emit("run_summary", {
#             "best_model":     best_model_name,
#             "best_score":     getattr(optimizer, "best_score", 0.0),
#             "saved_models":   [final_model_path],
#             "pipeline_path":  pipeline_path,
#             "log_dir":        self._logger.experiment_dir,
#         })
#         pl.info("Pipeline completed successfully",
#                 model=best_model_name, path=final_model_path,
#                 artefacts=self._logger.experiment_dir)
#         print("Done.\n")

#     # ── Steps ─────────────────────────────────────────────────────────

#     def _load_data(self) -> pd.DataFrame:
#         return DataSourceFactory.create(self._cfg.source).read()

#     def _profile_data(self, df: pd.DataFrame) -> Dict[str, Any]:
#         profiler = DataProfilerFactory.create(self._cfg.profiling_strategy)
#         return profiler.profile(df, self._cfg.target_column)

#     def _preprocess(self, profile: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
#         director = AdaptivePipelineDirector(task_type=self._cfg.task_type)
#         pipeline = director.build_from_profile(profile)
#         self._pipeline = pipeline
#         with _silent():
#             x_train_pp = pipeline.fit_transform(self._x_train)
#             x_test_pp  = pipeline.transform(self._x_test)
#         return x_train_pp, x_test_pp

#     # ── Training ──────────────────────────────────────────────────────

#     def _train_models(self) -> Tuple[Dict[str, Any], List[Dict]]:
#         cfg      = self._cfg
#         registry = Model_Registry.get_model_registry(cfg.models_library)
#         models   = registry.get_models(task_type=cfg.task_type)
#         evaluator = EvaluatorFactory.create(cfg.task_type)

#         metric_names = list(evaluator.metrics().keys())
#         table = TrainingTable(metric_names=metric_names, total_models=len(models))
#         table.start()

#         if cfg.parallel_training:
#             trained, eval_results = self._train_parallel(models, table)
#         else:
#             trained, eval_results = self._train_sequential(models, evaluator, table)

#         table.close()
#         return trained, eval_results

#     def _train_sequential(
#         self,
#         models: List[Tuple[str, Any]],
#         evaluator: Any,
#         table: TrainingTable,
#     ) -> Tuple[Dict[str, Any], List[Dict]]:
#         cfg     = self._cfg
#         trainer = LibraryTrainer.get_trainer(cfg.models_library)
#         trained: Dict[str, Any] = {}
#         eval_results: List[Dict] = []

#         for model_name, estimator_class in models:
#             t0 = time.perf_counter()
#             try:
#                 with _silent():
#                     model = trainer.train_model(
#                         model=estimator_class,
#                         X_train=self._x_train,
#                         y_train=self._y_train,
#                         task_type=cfg.task_type,
#                     )
#                 elapsed = time.perf_counter() - t0
#                 if model is None:
#                     table.add_result(model_name, None, status="failed", elapsed=elapsed)
#                     if self._plog:
#                         self._plog.warning(f"Model returned None: {model_name}")
#                     continue

#                 with _silent():
#                     result  = evaluator.evaluate(model_name, model,
#                                                  self._x_test, self._y_test)
#                 metrics = result.get(model_name) or {}
#                 trained[model_name] = (model_name, model)
#                 eval_results.append(result)
#                 table.add_result(model_name, metrics, status="done", elapsed=elapsed)
#                 if self._plog:
#                     self._plog.info(f"Trained {model_name}", elapsed=f"{elapsed:.1f}s",
#                                     **{k: f"{v:.4f}" for k, v in metrics.items()
#                                        if isinstance(v, float)})
#                 self._emit("model_done", {
#                     "model": model_name, "metrics": metrics, "elapsed": elapsed
#                 })

#             except Exception as exc:
#                 elapsed = time.perf_counter() - t0
#                 table.add_result(model_name, None, status="failed", elapsed=elapsed)
#                 if self._plog:
#                     self._plog.error(f"Training failed: {model_name}", exc=exc)

#         return trained, eval_results

#     def _train_parallel(
#         self,
#         models: List[Tuple[str, Any]],
#         table: TrainingTable,
#     ) -> Tuple[Dict[str, Any], List[Dict]]:
#         cfg    = self._cfg
#         n_jobs = cfg.n_jobs if cfg.n_jobs > 0 else mp.cpu_count()
#         n_jobs = min(n_jobs, len(models))

#         # Bug fix: pass y_test into the worker so metrics are not NaN
#         work_args = [
#             (name, cls,
#              self._x_train, self._y_train,
#              self._x_test,  self._y_test,   # <── y_test now included
#              cfg.task_type)
#             for name, cls in models
#         ]

#         trained: Dict[str, Any] = {}
#         eval_results: List[Dict] = []

#         ctx = mp.get_context("spawn")
#         with ctx.Pool(processes=n_jobs) as pool:
#             for result_tuple in pool.imap_unordered(
#                 _train_single_model_worker, work_args
#             ):
#                 model_name, model, metrics, elapsed, status = result_tuple
#                 if "done" in status and model is not None:
#                     trained[model_name] = (model_name, model)
#                     eval_results.append({model_name: metrics or None})
#                     table.add_result(model_name, metrics, status="done", elapsed=elapsed)
#                     if self._plog:
#                         self._plog.info(f"[parallel] Trained {model_name}",
#                                         elapsed=f"{elapsed:.1f}s")
#                     self._emit("model_done", {
#                         "model": model_name, "metrics": metrics, "elapsed": elapsed
#                     })
#                 else:
#                     table.add_result(model_name, None, status="failed", elapsed=elapsed)
#                     if self._plog:
#                         self._plog.warning(f"[parallel] Failed: {model_name}  status={status}")

#         return trained, eval_results

#     # ── Selection ─────────────────────────────────────────────────────

#     def _select_top_models(
#         self, evaluation_df: pd.DataFrame
#     ) -> Tuple[SelectorConfig, pd.DataFrame]:
#         cfg     = self._cfg
#         primary = cfg.selector_primary_metric or (
#             "Accuracy" if cfg.task_type == "classification" else "R2"
#         )
#         selector_config = SelectorConfig(
#             primary_metric   = primary,
#             stability_weight = cfg.selector_stability_weight,
#             train_metric     = cfg.selector_train_metric,
#             gap_penalty      = cfg.selector_gap_penalty,
#         )
#         selector = ModelSelectorFactory.create(cfg.selection_strategy,
#                                                config=selector_config)
#         return selector_config, selector.select(evaluation_df, k=cfg.top_k)

#     # ── Optimisation ──────────────────────────────────────────────────

#     def _optimize(
#         self,
#         trained_dict: Dict[str, Any],
#         top_models_df: pd.DataFrame,
#         profile: Dict[str, Any],
#     ) -> Tuple[Optimizer, str, Dict[str, Any]]:
#         cfg = self._cfg

#         llm     = LLMManager.get_llm_instance(cfg.model_provider_model_name)
#         advisor = LLMAdvisorAdapter(llm)

#         registry  = Model_Registry.get_model_registry(cfg.models_library)
#         top_names = top_models_df["Model"].tolist()
#         schema    = str(registry.get_all_hyperparameters(
#             top_models=top_names, models=trained_dict
#         ))

#         model_search_spaces = advisor.suggest_search_spaces(
#             data_profile=self._profile_to_xml(profile),
#             top_models=top_models_df.to_json(),
#             model_hyperparameter_schema=schema,
#         )
#         # Extract token_usage from the underlying LLM handler.
#         # LLMAdvisorAdapter wraps the handler in self._llm; BaseLLMHandler
#         # stores token_usage directly. Also handles direct BaseLLMHandler.
#         _advisor_token_usage = (
#             getattr(advisor, "token_usage", None) or
#             getattr(getattr(advisor, "_llm", None), "token_usage", None)
#         )

#         # Bug fix: guard against empty search spaces from LLM
#         if not model_search_spaces:
#             if self._plog:
#                 self._plog.error("AI advisor returned empty search spaces — "
#                                  "check LLM output. Falling back to first top model.")
#             raise ValueError(
#                 "AI advisor returned an empty search-space list. "
#                 "Check that the LLM response is a non-empty JSON array."
#             )

#         optimizer = OptimizerFactory.create(
#             method=cfg.optimizer_method,
#             task_type=cfg.task_type,
#             n_trials=cfg.n_trials,
#         )
#         optimizer._last_search_spaces    = model_search_spaces
#         optimizer._advisor_token_usage   = _advisor_token_usage

#         if self._plog:
#             self._plog.info(f"Running Optuna: {cfg.n_trials} trials")
#         with _silent():
#             best_model_name, best_params = optimizer.optimize(
#                 model_search_spaces=model_search_spaces,
#                 x_train=self._x_train,
#                 y_train=self._y_train,
#                 trained_models=trained_dict,
#             )
#         return optimizer, best_model_name, best_params

#     def _retrain_best(self, model_name: str, best_params: Dict[str, Any]) -> Any:
#         registry = Model_Registry.get_model_registry(self._cfg.models_library)
#         models   = dict(registry.get_models(task_type=self._cfg.task_type))
#         if model_name not in models:
#             raise ValueError(f"Model '{model_name}' not found in registry.")
#         trainer = LibraryTrainer.get_trainer(self._cfg.models_library)
#         with _silent():
#             return trainer.train_model(
#                 model=models[model_name],
#                 params=best_params,
#                 X_train=self._x_train,
#                 y_train=self._y_train,
#                 task_type=self._cfg.task_type,
#             )

#     def _save_model(self, model_name: str, model: Any) -> None:
#         _dir = getattr(self, "_model_dir", self._cfg.model_dir_path)
#         os.makedirs(_dir, exist_ok=True)
#         save_path = os.path.abspath(
#             os.path.join(_dir, f"{model_name}.joblib")
#         )
#         ModelUtils.save_model(model, save_path)
#         # Always show the user where the model landed — CLI, import, Streamlit
#         print(f"[MLTuneX] Model saved  : {save_path}")
#         if self._plog:
#             self._plog.info(f"Model saved", model=model_name, path=save_path)
#         self._emit("model_saved", {"model": model_name, "path": save_path})

#     def _save_pipeline(self) -> None:
#         """Persist the fitted preprocessing pipeline so it can be reused at inference time."""
#         if self._pipeline is None:
#             return
#         _dir = getattr(self, "_model_dir", self._cfg.model_dir_path)
#         os.makedirs(_dir, exist_ok=True)
#         save_path = os.path.abspath(
#             os.path.join(_dir, "preprocessing_pipeline.joblib")
#         )
#         try:
#             import joblib
#             joblib.dump(self._pipeline, save_path)
#             print(f"[MLTuneX] Pipeline saved: {save_path}")
#             if self._plog:
#                 self._plog.info("Preprocessing pipeline saved", path=save_path)
#             self._emit("pipeline_saved", {"path": save_path})
#         except Exception as exc:
#             if self._plog:
#                 self._plog.error("Failed to save preprocessing pipeline", exc=exc)

#     @staticmethod
#     def _profile_to_xml(profile: Dict[str, Any]) -> str:
#         lines = ["<DataProfile>"]
#         for key, value in profile.items():
#             lines.append(f"  <{key}>{value}</{key}>")
#         lines.append("</DataProfile>")
#         return "\n".join(lines)

"""
MLTuneX Orchestrator — central workflow coordinator (Facade Pattern).
"""

from __future__ import annotations

import io
import logging
import os
import sys
import time
import warnings
import multiprocessing as mp
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

# Suppress C-level library noise (LightGBM, XGBoost) before any imports
os.environ["LIGHTGBM_VERBOSITY"] = "-1"
os.environ["VERBOSITY"]           = "0"
# Suppress LightGBM [Info] thread-selection messages
os.environ["LIGHTGBM_SILENT"]     = "1"
# Redirect LightGBM C stdout (Jupyter / Colab compatible)
import logging as _log
for _lib in ("lightgbm","xgboost","catboost","optuna"):
    _l = _log.getLogger(_lib); _l.setLevel(_log.CRITICAL); _l.propagate = False

from mltunex.data.sources import DataSourceFactory
from mltunex.data.profiler import DataProfilerFactory
from mltunex.data.splitter import Data_Splitter
from mltunex.preprocessing import AdaptivePipelineDirector
from mltunex.model_registry.model_registry import Model_Registry
from mltunex.model_registry.selector import ModelSelectorFactory, SelectorConfig
from mltunex.library_trainer.library_trainer import LibraryTrainer
from mltunex.evaluate.evaluator import EvaluatorFactory
from mltunex.hyperparam_tuner.optimizer import (
    LLMAdvisorAdapter,
    Optimizer,
    OptimizerFactory,
)
from mltunex.ai_handler.llm_manager.llm_manager import LLMManager
from mltunex.utils.model_utils import ModelUtils
from mltunex.reporting.terminal_table import TrainingTable
from mltunex.reporting.experiment_logger import ExperimentLogger
from mltunex.reporting.pipeline_logger import PipelineLogger


# ─────────────────────────────────────────────────────────────────────────────
# Silence helper — routes all 3rd-party noise to the file log only
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def _silent():
    """
    Suppress all terminal output from 3rd-party libraries.

    Three layers:
      1. Python warnings module
      2. Python logging (sklearn, optuna, lightgbm, xgboost, ...)
      3. C-level stdout/stderr (LightGBM "[LightGBM] [Warning]" etc.)
         via fd-level redirect to /dev/null.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        noisy = ["optuna", "sklearn", "lightgbm", "xgboost", "catboost",
                 "urllib3", "httpx", "httpcore"]
        old_levels = {}
        for name in noisy:
            lg = logging.getLogger(name)
            old_levels[name] = lg.level
            lg.setLevel(logging.CRITICAL)

        # Redirect file descriptors 1 (stdout) and 2 (stderr) to /dev/null
        # This catches C-extension output that bypasses Python's sys.stdout
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_fds: Dict[int, int] = {}
        for fd in (1, 2):
            try:
                saved_fds[fd] = os.dup(fd)
                os.dup2(devnull_fd, fd)
            except Exception:
                pass

        try:
            yield
        finally:
            for fd, saved in saved_fds.items():
                try:
                    os.dup2(saved, fd)
                    os.close(saved)
                except Exception:
                    pass
            try:
                os.close(devnull_fd)
            except Exception:
                pass
            for name, level in old_levels.items():
                logging.getLogger(name).setLevel(level)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OrchestratorConfig:
    """All configuration for a single MLTuneX run."""

    # Required
    source: Union[str, pd.DataFrame]
    target_column: str
    task_type: str
    # Optional when tune_models=False — no API key needed
    model_provider_model_name: str = "Groq:none"

    # I/O
    result_csv_path: str = "results/"
    model_dir_path: str  = "models/"
    log_dir: str         = "logs/"
    experiment_name: str = "exp"

    # Training
    test_size: float        = 0.2
    preprocess: bool        = True
    parallel_training: bool = False
    n_jobs: int             = -1     # -1 = all CPUs

    # Tuning
    tune_models: bool             = True
    optimizer_method: str         = "optuna"
    n_trials: int                 = 25
    hyperparameter_framework: str = "Optuna"

    # Profiling / selection
    profiling_strategy: str       = "extended"
    models_library: str           = "sklearn"
    selection_strategy: str       = "topk"
    top_k: int                    = 3

    selector_primary_metric: Optional[str] = None
    selector_stability_weight: float       = 0.2
    selector_train_metric: Optional[str]   = None
    selector_gap_penalty: float            = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Module-level worker (must be top-level for multiprocessing pickle)
# ─────────────────────────────────────────────────────────────────────────────

def _train_single_model_worker(
    args: Tuple[str, Any, Any, Any, Any, Any, str],
) -> Tuple[str, Any, Dict[str, float], float, str]:
    """
    Train + evaluate one model in an isolated worker process.

    Critical: the worker receives x_test AND y_test so it can produce
    real metrics.  Previously y_test was not passed → all NaN metrics.
    """
    model_name, estimator_class, x_train, y_train, x_test, y_test, task_type = args

    # Silence ALL output inside the worker process
    import warnings as _w
    _w.filterwarnings("ignore")
    import logging as _l
    for _n in ["optuna", "sklearn", "lightgbm", "xgboost", "catboost"]:
        _l.getLogger(_n).setLevel(_l.CRITICAL)

    trainer   = LibraryTrainer.get_trainer("sklearn")
    evaluator = EvaluatorFactory.create(task_type)

    t0 = time.perf_counter()
    try:
        model = trainer.train_model(
            model=estimator_class,
            X_train=x_train,
            y_train=y_train,
            task_type=task_type,
        )
        if model is None:
            return model_name, None, {}, time.perf_counter() - t0, "failed"

        # Pass y_test so the evaluator can actually compute metrics
        result  = evaluator.evaluate(model_name, model, x_test, y_test)
        metrics = result.get(model_name) or {}
        # Strip NaN values so table renders cleanly
        metrics = {k: v for k, v in metrics.items() if v == v}  # NaN != NaN
        return model_name, model, metrics, time.perf_counter() - t0, "done"

    except Exception as exc:
        return model_name, None, {}, time.perf_counter() - t0, f"failed: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class MLTuneXOrchestrator:
    """Facade that coordinates the complete MLTuneX AutoML workflow."""

    def __init__(self, config: OrchestratorConfig) -> None:
        self._cfg      = config
        self._x_train: Optional[pd.DataFrame] = None
        self._x_test:  Optional[pd.DataFrame] = None
        self._y_train: Optional[pd.Series]    = None
        self._y_test:  Optional[pd.Series]    = None
        self._pipeline = None
        self._logger:  Optional[ExperimentLogger]  = None
        self._plog:    Optional[PipelineLogger]     = None
        # Optional callback for Streamlit / external progress tracking
        self._progress_callback = None

    # ── Public API ────────────────────────────────────────────────────

    def set_progress_callback(self, fn) -> None:
        """
        Register a callable that receives progress events.

        Signature: fn(stage: str, payload: dict) -> None
        Used by the Streamlit UI to update live widgets.
        """
        self._progress_callback = fn

    def _emit(self, stage: str, payload: dict) -> None:
        if self._progress_callback:
            self._progress_callback(stage, payload)

    def run(self) -> None:
        """Execute the complete AutoML workflow."""
        cfg = self._cfg

        # ── Initialise loggers ────────────────────────────────────────
        self._logger = ExperimentLogger(
            base_log_dir=cfg.log_dir,
            experiment_name=cfg.experiment_name,
        )
        # Override model save dir to be inside the experiment folder
        # so every artefact for a run lives together.
        self._model_dir = os.path.join(
            self._logger.experiment_dir, "models"
        )
        os.makedirs(self._model_dir, exist_ok=True)
        self._plog = PipelineLogger(
            log_dir=self._logger.experiment_dir,
            experiment_name=cfg.experiment_name,
        )
        self._plog.info("Run started", task=cfg.task_type,
                        source=str(cfg.source)[:120])

        try:
            self._run_pipeline()
        finally:
            self._plog.close()

    def _run_pipeline(self) -> None:
        cfg = self._cfg
        pl  = self._plog

        # [1] Load
        print("\n[1/8] Loading data ...")
        with pl.stage("Load Data"):
            df = self._load_data()
            pl.log_dataframe_shape("Loaded dataframe", df)
        self._emit("data_loaded", {"rows": len(df), "cols": len(df.columns)})

        # [2] Profile
        print("[2/8] Profiling data ...")
        with pl.stage("Profile Data"):
            profile = self._profile_data(df)
            pl.log_dict("Profile", {k: v for k, v in profile.items()
                                    if not isinstance(v, dict)})
        self._emit("profiled", {"profile": profile})

        # [3] Split
        print("[3/8] Splitting data ...")
        with pl.stage("Split Data"):
            X = df.drop(columns=[cfg.target_column])
            y = df[cfg.target_column]
            splitter = Data_Splitter()
            self._x_train, self._x_test, self._y_train, self._y_test = (
                splitter.split_data(X, y, test_size=cfg.test_size)
            )
            pl.info(f"Train: {len(self._x_train)} rows  Test: {len(self._x_test)} rows")

        # [4] Preprocess
        if cfg.preprocess:
            print("[4/8] Building and applying preprocessing pipeline ...")
            with pl.stage("Preprocess"):
                self._x_train, self._x_test = self._preprocess(profile)
                steps = self._pipeline.steps if self._pipeline else []
                pl.info(f"Pipeline steps: {[s for s, _ in steps]}")
        else:
            print("[4/8] Preprocessing skipped (preprocess=False).")
            pl.info("Preprocessing skipped")

        # Save the fitted pipeline for inference reuse
        if cfg.preprocess:
            self._save_pipeline()

        # Log preprocessing artefact
        pipeline_steps = self._pipeline.steps if self._pipeline else []
        self._logger.log_preprocessing(
            profile=profile,
            pipeline_steps=pipeline_steps,
            profiling_strategy=cfg.profiling_strategy,
            preprocessed_train=self._x_train,
            preprocessed_test=self._x_test,
        )

        # [5] Train
        # Count models so UI can show progress bar total
        _registry_preview = Model_Registry.get_model_registry(cfg.models_library)
        _n_total = len(_registry_preview.get_models(task_type=cfg.task_type))
        self._emit("n_models_total", {"total": _n_total})

        mode = "parallel" if cfg.parallel_training else "sequential"
        print(f"\n[5/8] Training initial models ({mode}) ...\n")
        with pl.stage(f"Train Models ({mode})"):
            trained_dict, all_eval_results = self._train_models()
            pl.info(f"Trained {len(trained_dict)} models successfully")
        # [6] Metrics
        print("\n[6/8] Saving model metrics ...")
        with pl.stage("Save Metrics"):
            self._logger.log_model_metrics(
                all_results=all_eval_results,
                task_type=cfg.task_type,
            )
            evaluation_df = ModelUtils.save_results(
                evaluation_results=all_eval_results,
                evaluation_results_path=cfg.result_csv_path,
            )
            pl.log_dataframe_shape("Evaluation DataFrame", evaluation_df)
        self._emit("metrics_ready", {"eval_df": evaluation_df})

        if not cfg.tune_models:
            print("[7/8] Hyperparameter tuning skipped (tune_models=False).")
            # Still select and save top models so the user gets useful output
            print("      Selecting and saving top models ...")
            with pl.stage("Select Top Models (no-tune)"):
                selector_cfg, top_models_df = self._select_top_models(evaluation_df)
                pl.info(f"Selected: {top_models_df['Model'].tolist()}")
                self._logger.log_selection(
                    all_results_df=evaluation_df,
                    selected_df=top_models_df,
                    selection_strategy=cfg.selection_strategy,
                    selector_config=selector_cfg,
                )
            self._emit("selection_done", {"top_models": top_models_df})

            # Save each top model to disk
            saved_paths = []
            for _, row in top_models_df.iterrows():
                model_name = row["Model"]
                if model_name in trained_dict:
                    _, model_obj = trained_dict[model_name]
                    self._save_model(model_name, model_obj)
                    saved_paths.append(
                        os.path.abspath(
                            os.path.join(
                                getattr(self, "_model_dir", cfg.model_dir_path),
                                f"{model_name}.joblib",
                            )
                        )
                    )
                    pl.info(f"Saved model: {model_name}")

            print("\n[MLTuneX] ---- Saved Models ----")
            for p in saved_paths:
                print(f"  {p}")
            print("[MLTuneX] -------------------------\n")
            self._emit("run_summary", {
                "saved_models": saved_paths,
                "pipeline_path": os.path.abspath(
                    os.path.join(getattr(self,"_model_dir",cfg.model_dir_path), "preprocessing_pipeline.joblib")
                ) if cfg.preprocess else None,
                "log_dir": self._logger.experiment_dir,
            })
            print("[8/8] Done.\n")
            pl.info("Pipeline completed (no tuning)")
            return

        # [7] Select
        print("[7/8] Selecting top candidates ...")
        with pl.stage("Select Top Models"):
            selector_cfg, top_models_df = self._select_top_models(evaluation_df)
            pl.info(f"Selected: {top_models_df['Model'].tolist()}")
            self._logger.log_selection(
                all_results_df=evaluation_df,
                selected_df=top_models_df,
                selection_strategy=cfg.selection_strategy,
                selector_config=selector_cfg,
            )
        self._emit("selection_done", {"top_models": top_models_df})

        # [8] Optimise
        print("[8/8] Running AI-guided hyperparameter optimisation ...")
        with pl.stage("Hyperparameter Optimisation"):
            optimizer, best_model_name, best_params = self._optimize(
                trained_dict, top_models_df, profile
            )
            pl.info(f"Best model: {best_model_name}  params: {best_params}")

        print(f"\n[MLTuneX] Best model : {best_model_name}")
        print(f"  Best params : {best_params}\n")

        self._logger.log_tuning(
            ai_suggestions=getattr(optimizer, "_last_search_spaces", []),
            trial_history=getattr(optimizer, "trial_history", []),
            best_model_name=best_model_name,
            best_params=best_params,
            best_score=getattr(optimizer, "best_score", 0.0),
            n_trials=cfg.n_trials,
            optimizer_method=cfg.optimizer_method,
            token_usage=getattr(optimizer, "_advisor_token_usage", None),
        )
        self._emit("tuning_done", {
            "best_model": best_model_name,
            "best_params": best_params,
            "best_score": getattr(optimizer, "best_score", 0.0),
            "trial_history": getattr(optimizer, "trial_history", []),
        })

        print("Retraining best model with optimal parameters ...")
        with pl.stage("Retrain Best Model"):
            final_model = self._retrain_best(best_model_name, best_params)

        print("Saving final model ...")
        self._save_model(best_model_name, final_model)
        final_model_path = os.path.abspath(
            os.path.join(getattr(self,"_model_dir",cfg.model_dir_path), f"{best_model_name}.joblib")
        )
        pipeline_path = os.path.abspath(
            os.path.join(
                getattr(self, "_model_dir", cfg.model_dir_path),
                "preprocessing_pipeline.joblib",
            )
        ) if cfg.preprocess else None

        print("\n[MLTuneX] ========== Run Summary ==========")
        print(f"  Best model   : {best_model_name}")
        print(f"  Best score   : {getattr(optimizer, 'best_score', 0.0):.5f}")
        print(f"  Model saved  : {final_model_path}")
        if pipeline_path and os.path.exists(pipeline_path):
            print(f"  Pipeline     : {pipeline_path}")
        print(f"  Logs & reports: {self._logger.experiment_dir}")
        print("[MLTuneX] ====================================\n")

        self._emit("run_summary", {
            "best_model":     best_model_name,
            "best_score":     getattr(optimizer, "best_score", 0.0),
            "saved_models":   [final_model_path],
            "pipeline_path":  pipeline_path,
            "log_dir":        self._logger.experiment_dir,
        })
        pl.info("Pipeline completed successfully",
                model=best_model_name, path=final_model_path,
                artefacts=self._logger.experiment_dir)
        print("Done.\n")

    # ── Steps ─────────────────────────────────────────────────────────

    def _load_data(self) -> pd.DataFrame:
        return DataSourceFactory.create(self._cfg.source).read()

    def _profile_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        profiler = DataProfilerFactory.create(self._cfg.profiling_strategy)
        return profiler.profile(df, self._cfg.target_column)

    def _preprocess(self, profile: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        director = AdaptivePipelineDirector(task_type=self._cfg.task_type)
        pipeline = director.build_from_profile(profile)
        self._pipeline = pipeline
        with _silent():
            x_train_pp = pipeline.fit_transform(self._x_train)
            x_test_pp  = pipeline.transform(self._x_test)
        return x_train_pp, x_test_pp

    # ── Training ──────────────────────────────────────────────────────

    def _train_models(self) -> Tuple[Dict[str, Any], List[Dict]]:
        cfg      = self._cfg
        registry = Model_Registry.get_model_registry(cfg.models_library)
        models   = registry.get_models(task_type=cfg.task_type)
        evaluator = EvaluatorFactory.create(cfg.task_type)

        metric_names = list(evaluator.metrics().keys())
        table = TrainingTable(metric_names=metric_names, total_models=len(models))
        table.start()

        if cfg.parallel_training:
            trained, eval_results = self._train_parallel(models, table)
        else:
            trained, eval_results = self._train_sequential(models, evaluator, table)

        table.close()
        return trained, eval_results

    def _train_sequential(
        self,
        models: List[Tuple[str, Any]],
        evaluator: Any,
        table: TrainingTable,
    ) -> Tuple[Dict[str, Any], List[Dict]]:
        cfg     = self._cfg
        trainer = LibraryTrainer.get_trainer(cfg.models_library)
        trained: Dict[str, Any] = {}
        eval_results: List[Dict] = []

        for model_name, estimator_class in models:
            t0 = time.perf_counter()
            try:
                with _silent():
                    model = trainer.train_model(
                        model=estimator_class,
                        X_train=self._x_train,
                        y_train=self._y_train,
                        task_type=cfg.task_type,
                    )
                elapsed = time.perf_counter() - t0
                if model is None:
                    table.add_result(model_name, None, status="failed", elapsed=elapsed)
                    if self._plog:
                        self._plog.warning(f"Model returned None: {model_name}")
                    continue

                with _silent():
                    result  = evaluator.evaluate(model_name, model,
                                                 self._x_test, self._y_test)
                metrics = result.get(model_name) or {}
                trained[model_name] = (model_name, model)
                eval_results.append(result)
                table.add_result(model_name, metrics, status="done", elapsed=elapsed)
                if self._plog:
                    self._plog.info(f"Trained {model_name}", elapsed=f"{elapsed:.1f}s",
                                    **{k: f"{v:.4f}" for k, v in metrics.items()
                                       if isinstance(v, float)})
                self._emit("model_done", {
                    "model": model_name, "metrics": metrics, "elapsed": elapsed
                })

            except Exception as exc:
                elapsed = time.perf_counter() - t0
                table.add_result(model_name, None, status="failed", elapsed=elapsed)
                if self._plog:
                    self._plog.error(f"Training failed: {model_name}", exc=exc)

        return trained, eval_results

    def _train_parallel(
        self,
        models: List[Tuple[str, Any]],
        table: TrainingTable,
    ) -> Tuple[Dict[str, Any], List[Dict]]:
        cfg    = self._cfg
        n_jobs = cfg.n_jobs if cfg.n_jobs > 0 else mp.cpu_count()
        n_jobs = min(n_jobs, len(models))

        # Bug fix: pass y_test into the worker so metrics are not NaN
        work_args = [
            (name, cls,
             self._x_train, self._y_train,
             self._x_test,  self._y_test,   # <── y_test now included
             cfg.task_type)
            for name, cls in models
        ]

        trained: Dict[str, Any] = {}
        eval_results: List[Dict] = []

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_jobs) as pool:
            for result_tuple in pool.imap_unordered(
                _train_single_model_worker, work_args
            ):
                model_name, model, metrics, elapsed, status = result_tuple
                if "done" in status and model is not None:
                    trained[model_name] = (model_name, model)
                    eval_results.append({model_name: metrics or None})
                    table.add_result(model_name, metrics, status="done", elapsed=elapsed)
                    if self._plog:
                        self._plog.info(f"[parallel] Trained {model_name}",
                                        elapsed=f"{elapsed:.1f}s")
                    self._emit("model_done", {
                        "model": model_name, "metrics": metrics, "elapsed": elapsed
                    })
                else:
                    table.add_result(model_name, None, status="failed", elapsed=elapsed)
                    if self._plog:
                        self._plog.warning(f"[parallel] Failed: {model_name}  status={status}")

        return trained, eval_results

    # ── Selection ─────────────────────────────────────────────────────

    def _select_top_models(
        self, evaluation_df: pd.DataFrame
    ) -> Tuple[SelectorConfig, pd.DataFrame]:
        cfg     = self._cfg
        primary = cfg.selector_primary_metric or (
            "Accuracy" if cfg.task_type == "classification" else "R2"
        )
        selector_config = SelectorConfig(
            primary_metric   = primary,
            stability_weight = cfg.selector_stability_weight,
            train_metric     = cfg.selector_train_metric,
            gap_penalty      = cfg.selector_gap_penalty,
        )
        selector = ModelSelectorFactory.create(cfg.selection_strategy,
                                               config=selector_config)
        return selector_config, selector.select(evaluation_df, k=cfg.top_k)

    # ── Optimisation ──────────────────────────────────────────────────

    def _optimize(
        self,
        trained_dict: Dict[str, Any],
        top_models_df: pd.DataFrame,
        profile: Dict[str, Any],
    ) -> Tuple[Optimizer, str, Dict[str, Any]]:
        cfg = self._cfg

        llm     = LLMManager.get_llm_instance(cfg.model_provider_model_name)
        advisor = LLMAdvisorAdapter(llm)

        registry  = Model_Registry.get_model_registry(cfg.models_library)
        top_names = top_models_df["Model"].tolist()
        schema    = str(registry.get_all_hyperparameters(
            top_models=top_names, models=trained_dict
        ))

        model_search_spaces = advisor.suggest_search_spaces(
            data_profile=self._profile_to_xml(profile),
            top_models=top_models_df.to_json(),
            model_hyperparameter_schema=schema,
        )
        # Extract token_usage from the underlying LLM handler.
        # LLMAdvisorAdapter wraps the handler in self._llm; BaseLLMHandler
        # stores token_usage directly. Also handles direct BaseLLMHandler.
        _advisor_token_usage = (
            getattr(advisor, "token_usage", None) or
            getattr(getattr(advisor, "_llm", None), "token_usage", None)
        )

        # Bug fix: guard against empty search spaces from LLM
        if not model_search_spaces:
            if self._plog:
                self._plog.error("AI advisor returned empty search spaces — "
                                 "check LLM output. Falling back to first top model.")
            raise ValueError(
                "AI advisor returned an empty search-space list. "
                "Check that the LLM response is a non-empty JSON array."
            )

        optimizer = OptimizerFactory.create(
            method=cfg.optimizer_method,
            task_type=cfg.task_type,
            n_trials=cfg.n_trials,
        )
        optimizer._last_search_spaces    = model_search_spaces
        optimizer._advisor_token_usage   = _advisor_token_usage

        if self._plog:
            self._plog.info(f"Running Optuna: {cfg.n_trials} trials")
        with _silent():
            best_model_name, best_params = optimizer.optimize(
                model_search_spaces=model_search_spaces,
                x_train=self._x_train,
                y_train=self._y_train,
                trained_models=trained_dict,
            )
        return optimizer, best_model_name, best_params

    def _retrain_best(self, model_name: str, best_params: Dict[str, Any]) -> Any:
        registry = Model_Registry.get_model_registry(self._cfg.models_library)
        models   = dict(registry.get_models(task_type=self._cfg.task_type))
        if model_name not in models:
            raise ValueError(f"Model '{model_name}' not found in registry.")
        trainer = LibraryTrainer.get_trainer(self._cfg.models_library)
        with _silent():
            return trainer.train_model(
                model=models[model_name],
                params=best_params,
                X_train=self._x_train,
                y_train=self._y_train,
                task_type=self._cfg.task_type,
            )

    def _save_model(self, model_name: str, model: Any) -> None:
        _dir = getattr(self, "_model_dir", self._cfg.model_dir_path)
        os.makedirs(_dir, exist_ok=True)
        save_path = os.path.abspath(
            os.path.join(_dir, f"{model_name}.joblib")
        )
        ModelUtils.save_model(model, save_path)
        # Always show the user where the model landed — CLI, import, Streamlit
        print(f"[MLTuneX] Model saved  : {save_path}")
        if self._plog:
            self._plog.info(f"Model saved", model=model_name, path=save_path)
        self._emit("model_saved", {"model": model_name, "path": save_path})

    def _save_pipeline(self) -> None:
        """Persist the fitted preprocessing pipeline so it can be reused at inference time."""
        if self._pipeline is None:
            return
        _dir = getattr(self, "_model_dir", self._cfg.model_dir_path)
        os.makedirs(_dir, exist_ok=True)
        save_path = os.path.abspath(
            os.path.join(_dir, "preprocessing_pipeline.joblib")
        )
        try:
            import joblib
            joblib.dump(self._pipeline, save_path)
            print(f"[MLTuneX] Pipeline saved: {save_path}")
            if self._plog:
                self._plog.info("Preprocessing pipeline saved", path=save_path)
            self._emit("pipeline_saved", {"path": save_path})
        except Exception as exc:
            if self._plog:
                self._plog.error("Failed to save preprocessing pipeline", exc=exc)

    @staticmethod
    def _profile_to_xml(profile: Dict[str, Any]) -> str:
        lines = ["<DataProfile>"]
        for key, value in profile.items():
            lines.append(f"  <{key}>{value}</{key}>")
        lines.append("</DataProfile>")
        return "\n".join(lines)