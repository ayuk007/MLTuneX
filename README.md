# MLTuneX — Enhanced AutoML Fine-Tuning System

MLTuneX is a modular, extensible AutoML framework that trains multiple models,
selects top candidates, and uses AI to guide Optuna-powered hyperparameter
optimisation. The enhanced version is fully SOLID-compliant and built around
well-defined abstract interfaces and design patterns.

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│              MLTuneXOrchestrator  (Facade Pattern)             │
│                                                                │
│  DataSourceFactory ──► DataSource ──► DataFrame                │
│  DataProfilerFactory ──► DataProfiler ──► profile dict         │
│  AdaptivePipelineDirector ──► PreprocessingPipeline            │
│  ModelRegistry + LibraryTrainer ──► trained models             │
│  EvaluatorFactory ──► Evaluator ──► results DataFrame          │
│  ModelSelectorFactory ──► ModelSelector ──► top candidates     │
│  LLMAdvisorAdapter ──► AIAdvisor ──► search spaces             │
│  OptimizerFactory ──► Optimizer ──► best model + params        │
└────────────────────────────────────────────────────────────────┘
```

---

## SOLID Compliance

| Principle | How it is applied |
|---|---|
| **SRP** | Each module has a single, well-scoped responsibility |
| **OCP** | New sources / models / strategies registered without modifying existing code |
| **LSP** | All interchangeable components implement a consistent abstract interface |
| **ISP** | Interfaces are small and focused (`DataSource`, `Evaluator`, `Optimizer`, …) |
| **DIP** | All dependencies are on abstractions; concrete classes never leak across module boundaries |

---

## What Was Added / Changed

### 1. Data Ingestion — `mltunex/data/sources.py`

New generic `DataSource` interface with concrete implementations for CSV, Excel,
Parquet, Feather, SQL, and in-memory DataFrames. `DataSourceFactory` selects the
right implementation automatically. Register custom sources with:
```python
DataSourceFactory.register(".ext", MySource)
```

### 2. Data Profiling — `mltunex/data/profiler.py`

New `DataProfiler` interface (Strategy Pattern). `BasicDataProfiler` covers
shape, dtypes, missing %, cardinality and target distribution. `ExtendedDataProfiler`
adds variance groups, skew, kurtosis and correlations.

### 3. Preprocessing Engine — `mltunex/preprocessing/__init__.py`

Strategy + Builder + Director patterns. Available strategies: NumericImputer,
CategoricalImputer, StandardScalerStrategy, MinMaxScalerStrategy,
OneHotEncoderStrategy, OrdinalEncoderStrategy, OutlierClipper.
`AdaptivePipelineDirector` auto-configures from a profile dict.

### 4. Model Abstraction — `mltunex/model_registry/model_interface.py`

New `Model` interface with `train`, `predict`, `evaluate`. `SklearnModel` wraps
any sklearn-compatible estimator. `ModelFactory` is the abstract factory.

### 5. Model Selection — `mltunex/model_registry/selector.py`

New `ModelSelector` interface. Strategies: `TopKByMetricSelector`,
`StabilityAwareSelector`, `GeneralizationSelector`. Swap via:
```python
selector = ModelSelectorFactory.create("stability", metric="Accuracy")
```

### 6. Evaluation Engine — `mltunex/evaluate/evaluator.py`

New `Evaluator` interface with `ClassificationEvaluator` and `RegressionEvaluator`.
Register custom evaluators via `EvaluatorFactory.register(...)`.

### 7. Optimization Pipeline — `mltunex/hyperparam_tuner/optimizer.py`

- `AIAdvisor` — abstract interface for search-space advisors
- `LLMAdvisorAdapter` — Adapter Pattern bridging LLMManager into AIAdvisor
- `Optimizer` — abstract interface for optimization backends
- `OptunaOptimizer` — Optuna implementation; accepts any `AIAdvisor`

### 8. Orchestration Layer — `mltunex/orchestrator.py`

`MLTuneXOrchestrator` (Facade) wires everything together. `OrchestratorConfig`
is a single dataclass controlling every aspect of a run.

```python
from mltunex.orchestrator import MLTuneXOrchestrator, OrchestratorConfig

config = OrchestratorConfig(
    source="data/titanic.csv",
    target_column="Survived",
    task_type="classification",
    model_provider_model_name="Groq:qwen/qwen3-32b",
    selection_strategy="stability",
    profiling_strategy="extended",
    top_k=3,
    n_trials=30,
)
MLTuneXOrchestrator(config).run()
```

---

## Extensibility Cheat-Sheet

| Goal | How |
|---|---|
| New data source format | `DataSourceFactory.register(".fmt", MySource)` |
| New profiling strategy | `DataProfilerFactory.register("deep", MyProfiler)` |
| New preprocessing step | `builder.add_custom_step("name", MyStrategy())` |
| New ML model backend | Implement `Model` + `ModelFactory`; register in `Model_Registry` |
| New selection strategy | `ModelSelectorFactory.register("my", MySelector)` |
| New evaluation task | `EvaluatorFactory.register("ranking", MyEvaluator)` |
| New optimizer | `OptimizerFactory.register("hyperopt", MyOptimizer)` |

No existing module requires modification for any of the above.

---

## Backward Compatibility

The original `MLTuneX` class in `mltunex.main` is **unchanged**. All existing
code continues to work. `MLTuneXOrchestrator` is an additive, parallel entry-point.

---

## Installation

```bash
pip install -e .
```
