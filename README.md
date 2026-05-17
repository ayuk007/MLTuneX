<div align="center">
  <h1>🤖 MLTuneX</h1>
  <p><strong>Automated Machine Learning Fine-Tuning System</strong></p>
  <p>
    <a href="https://pypi.org/project/MLTuneX/"><img src="https://img.shields.io/pypi/v/MLTuneX?color=6366f1" alt="PyPI" /></a>
    <a href="https://pypi.org/project/MLTuneX/"><img src="https://img.shields.io/pypi/pyversions/MLTuneX" alt="Python" /></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-22c55e" alt="License" /></a>
  </p>
  <p>
    <a href="#installation">Installation</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#usage">Usage</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#configuration-reference">Configuration</a> •
    <a href="#extending-mltunex">Extending</a> •
    <a href="#experiment-artefacts">Artefacts</a>
  </p>
</div>

---

MLTuneX is a production-grade AutoML library that takes your dataset from raw CSV to a fine-tuned model in a single command. It automatically profiles your data, builds a preprocessing pipeline, trains every applicable model, selects the top candidates, uses an LLM to generate smart hyperparameter search spaces, and runs Optuna to find the best configuration — all with zero boilerplate.

```
Raw dataset
    │
    ▼
Profile ──► Preprocess ──► Train all models ──► Evaluate ──► Select top-K
    │                                                         |
    ▼                                                         ▼
AI advisor generates search spaces  (Groq / OpenAI / custom LLM)
    │
    ▼
Optuna optimises ──► Best model + fitted pipeline saved inside experiment dir
    │
    ▼
Experiment artefacts: Markdown reports · Excel metrics · pipeline.log · token usage
```

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Python API](#python-api)
  - [Command-Line Interface](#command-line-interface)
  - [Streamlit UI](#streamlit-ui)
- [Architecture](#architecture)
  - [Pipeline Stages](#pipeline-stages)
  - [Design Patterns](#design-patterns)
  - [SOLID Compliance](#solid-compliance)
- [Configuration Reference](#configuration-reference)
- [Experiment Artefacts](#experiment-artefacts)
- [AI Schema Validation and Fallback](#ai-schema-validation-and-fallback)
- [Extending MLTuneX](#extending-mltunex)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## Features

| Feature | Detail |
|---|---|
| **Zero-boilerplate AutoML** | One function call covers the full pipeline end to end |
| **AI-guided tuning** | LLM-generated hyperparameter search spaces tailored to your dataset profile |
| **Pluggable LLM registry** | Groq and OpenAI built-in; attach any custom LLM in ~10 lines |
| **Schema-validated AI output** | Structural validation + 3-retry loop with exponential back-off; predefined fallback for 28 models |
| **Token usage tracking** | Prompt / completion / total tokens logged per run in the tuning report |
| **Adaptive preprocessing** | Profile-driven pipeline (imputation, outlier clipping, encoding, scaling) built automatically |
| **Parallel training** | Optional multiprocessing pool — all models trained simultaneously |
| **Jupyter / Colab support** | Training table auto-detects the runtime and uses plain output in notebooks; all C-level library noise suppressed |
| **3 interfaces** | Python API · CLI (`mltunex`) · Streamlit UI (`mltunex ui`) |
| **Experiment-scoped artefacts** | Every run writes to a single timestamped directory: models, pipeline, Markdown reports, Excel metrics, structured log |
| **Pluggable tuning schemas** | Response format driven by `ResponseSchemaRegistry`; add custom schemas for new optimisers |
| **Pluggable selectors** | TopK, StabilityAware, GeneralizationAware, or register your own |
| **No-tune mode** | Skip LLM entirely — select and save top-K models without an API key |

---

## Installation

### From PyPI

```bash
pip install MLTuneX
```

### From source

```bash
git clone https://github.com/ayuk007/MLTuneX.git
cd MLTuneX
pip install -e .
```

### Optional extras

```bash
pip install MLTuneX[catboost]   # CatBoost support
pip install MLTuneX[parquet]    # Parquet / Feather file support
```

### API keys

```bash
# Groq (free tier at console.groq.com)
export GROQ_API_KEY="gsk_..."

# OpenAI
export OPENAI_API_KEY="sk-..."
```

Keys can also be entered directly in the Streamlit UI — they are used only for the current session and never written to disk.

---

## Quick Start

```python
from mltunex.orchestrator import MLTuneXOrchestrator, OrchestratorConfig

config = OrchestratorConfig(
    source                    = "titanic.csv",
    target_column             = "Survived",
    task_type                 = "classification",
    model_provider_model_name = "Groq:qwen/qwen3-32b",
)

MLTuneXOrchestrator(config).run()
```

That single call: loads and profiles data, builds a preprocessing pipeline, trains every sklearn / XGBoost / LightGBM classifier, evaluates each one, asks the LLM for hyperparameter search spaces, runs 25 Optuna trials, retrains with the best params, and saves every artefact to a timestamped experiment directory.

---

## Usage

### Python API

#### Minimal classification

```python
from mltunex.orchestrator import MLTuneXOrchestrator, OrchestratorConfig

config = OrchestratorConfig(
    source                    = "data/titanic.csv",
    target_column             = "Survived",
    task_type                 = "classification",
    model_provider_model_name = "Groq:qwen/qwen3-32b",
)
MLTuneXOrchestrator(config).run()
```

#### Regression with full options

```python
config = OrchestratorConfig(
    source          = "data/housing.parquet",
    target_column   = "SalePrice",
    task_type       = "regression",
    test_size       = 0.15,

    model_provider_model_name = "OpenAI:gpt-4o",

    preprocess        = True,
    parallel_training = True,
    n_jobs            = 8,

    tune_models  = True,
    n_trials     = 50,

    selection_strategy        = "stability",
    top_k                     = 5,
    selector_primary_metric   = "R2",
    selector_stability_weight = 0.3,

    log_dir         = "logs/",
    experiment_name = "housing_v1",
)
MLTuneXOrchestrator(config).run()
```

#### Skip tuning — no API key required

```python
config = OrchestratorConfig(
    source        = "data/fraud.csv",
    target_column = "is_fraud",
    task_type     = "classification",
    # model_provider_model_name not required when tune_models=False

    tune_models        = False,   # top-K models saved directly, no LLM call
    selection_strategy = "generalization",
    selector_train_metric = "Train_Accuracy",
    top_k              = 3,
)
MLTuneXOrchestrator(config).run()
```

#### Pass an in-memory DataFrame

```python
import pandas as pd

df = pd.read_csv("data.csv")
config = OrchestratorConfig(
    source        = df,           # pass DataFrame directly
    target_column = "label",
    task_type     = "classification",
    model_provider_model_name = "Groq:qwen/qwen3-32b",
)
MLTuneXOrchestrator(config).run()
```

#### Jupyter / Colab usage

No extra configuration needed. MLTuneX auto-detects the Jupyter environment and switches the training table to plain text output:

```python
from mltunex.orchestrator import MLTuneXOrchestrator, OrchestratorConfig

config = OrchestratorConfig(
    source        = "titanic.csv",
    target_column = "Survived",
    task_type     = "classification",
    # tune_models=True requires model_provider_model_name
    model_provider_model_name = "Groq:qwen/qwen3-32b",
)
MLTuneXOrchestrator(config).run()

# Skip AI tuning — no API key needed
config_notune = OrchestratorConfig(
    source        = "titanic.csv",
    target_column = "Survived",
    task_type     = "classification",
    tune_models   = False,   # model_provider_model_name not required
)
MLTuneXOrchestrator(config_notune).run()
```

#### Load saved artefacts for inference

Both the model and the preprocessing pipeline are saved inside the experiment directory:

```python
import joblib, pandas as pd

pipeline = joblib.load("logs/exp_20250502_143022/models/preprocessing_pipeline.joblib")
model    = joblib.load("logs/exp_20250502_143022/models/RandomForestClassifier.joblib")

X_new  = pd.read_csv("new_data.csv").drop(columns=["target"])
preds  = model.predict(pipeline.transform(X_new))
```

---

### Command-Line Interface

```bash
mltunex --help
```

#### Minimal run

```bash
mltunex --data titanic.csv \
        --target Survived \
        --task classification \
        --llm "Groq:qwen/qwen3-32b"
```

#### Skip tuning — `--llm` not required

```bash
mltunex --data fraud.csv --target label --task classification \
        --no-tune --selector stability --top-k 5
```

#### Full options reference

```
required arguments:
  --data PATH           Dataset file (.csv, .xlsx, .parquet, .feather)
  --target COLUMN       Target column name
  --task                classification | regression
  --llm PROVIDER:MODEL  LLM for AI tuning. Required unless --no-tune is set.
                        Format: Provider:ModelName
                        Examples: "Groq:qwen/qwen3-32b"  "OpenAI:gpt-4o"

training options:
  --test-size FLOAT     Test split fraction (default: 0.2)
  --no-preprocess       Skip adaptive preprocessing
  --parallel            Train models in parallel (multiprocessing pool)
  --jobs N              Worker count for parallel training (-1 = all CPUs)
  --library LIB         Model library backend (default: sklearn)

tuning options:
  --no-tune             Skip AI hyperparameter optimisation entirely.
                        Top-K models are selected and saved directly.
                        --llm is NOT required when this flag is set.
  --trials N            Optuna trial count (default: 25)
  --optimizer NAME      Optimizer backend (default: optuna)

model selection:
  --selector            topk | stability | generalization (default: topk)
  --top-k N             Candidates forwarded to optimiser (default: 3)
  --primary-metric      Override primary ranking metric
                        (default: Accuracy for classification, R2 for regression)
  --stability-weight    [stability selector] Instability penalty weight (default: 0.2)
  --train-metric        [generalization selector] Train-set metric column name
  --gap-penalty         [generalization selector] Gap penalty weight (default: 0.5)
  --profiling           basic | extended (default: extended)

output:
  --results DIR         Results CSV directory (default: results/)
  --models DIR          Fallback model directory (default: models/).
                        Models are always saved inside <log-dir>/<exp-name>_<ts>/models/
  --log-dir DIR         Experiment log root (default: logs/)
  --exp-name NAME       Tag prepended to the experiment folder (default: exp)
```

#### Launch the Streamlit UI

```bash
mltunex ui
```

---

### Streamlit UI

```bash
mltunex ui
# or
python -m mltunex.ui
```

Opens at `http://localhost:8501`. The sidebar is replaced by a **Configure** tab so there are no sidebar collapse/visibility issues.

**Tab overview:**

| Tab | Contents |
|---|---|
| **Configure** | Upload data, select target column from a populated dropdown, configure all options, paste API key securely, launch the run |
| **Run** | Live event log, model training progress bar, incremental results table, final metrics, saved artefact paths |
| **Profile** | Dataset overview metrics, missing data table with severity indicators, skewness bar chart, target distribution chart |
| **Results** | Sortable model leaderboard + primary-metric bar chart |
| **Selection** | Strategy card with description, selected candidates table |
| **Tuning** | Best score metrics, optimal parameters table, score progression line chart, full ranked trial history |
| **Reports** | Inline rendered Markdown reports (light-themed container), Excel preview with download, pipeline log viewer |

**UI notes:**

- Target column is a **dropdown** auto-populated from the uploaded file's columns — no typing required
- AI Advisor section (provider, model, API key) is **hidden** when "AI hyperparameter tuning" toggle is off
- All orchestrator `print()` output and C-level library noise (LightGBM, XGBoost, Optuna) is fully suppressed from the terminal — nothing leaks to the Streamlit server console
- API keys are kept in session memory only — never written to disk or logs

---

## Architecture

### Pipeline Stages

```
Input  →  CSV / Excel / Parquet / Feather / DataFrame / SQL
           DataSourceFactory → DataSource.read() → DataFrame

[1] Profiling       DataProfilerFactory → DataProfiler.profile() → metadata dict
[2] Splitting       Data_Splitter (stratified)
[3] Preprocessing   AdaptivePipelineDirector builds PreprocessingPipeline from profile
                    Steps: NumericImputer · OutlierClipper · CategoricalImputer ·
                           OneHotEncoder · OrdinalEncoder · StandardScaler / MinMaxScaler
                    Fitted pipeline → <experiment_dir>/models/preprocessing_pipeline.joblib
[4] Training        Model_Registry → LibraryTrainer → trained models
                    Rich live terminal table updated after each model
                    Optional multiprocessing Pool (parallel_training=True)
[5] Evaluation      EvaluatorFactory → Evaluator.evaluate() → metrics dict
                    model_metrics.xlsx → All Models / Ranked / Failed sheets
[6] Selection       ModelSelectorFactory → ModelSelector.select() → top-K DataFrame
                    Strategies: TopKByMetric · StabilityAware · GeneralizationAware
                    When tune_models=False → top models saved here, pipeline ends
[7] AI Search Space LLMHandlerRegistry → BaseLLMHandler.suggest_search_spaces()
                    → ResponseSchemaRegistry validates format instructions
                    → HyperparamSchema validates parsed output (3-retry + fallback)
                    → Token usage accumulated (prompt / completion / total)
[8] Optimisation    OptunaOptimizer.optimize() → best model + params
                    Trial history recorded per trial
[9] Retrain + Save  Best model retrained with optimal params
                    → <experiment_dir>/models/<ModelName>.joblib
```

### Design Patterns

| Pattern | Where used |
|---|---|
| **Facade** | `MLTuneXOrchestrator` — single `run()` call hides all complexity |
| **Factory** | `DataSourceFactory`, `DataProfilerFactory`, `EvaluatorFactory`, `ModelSelectorFactory`, `OptimizerFactory`, `LLMHandlerRegistry` |
| **Strategy** | `DataProfiler`, `PreprocessingStrategy`, `ModelSelector`, `Evaluator`, `Optimizer`, `TuningResponseSchema` |
| **Builder** | `PreprocessingPipelineBuilder` — fluent pipeline assembly |
| **Director** | `AdaptivePipelineDirector` — profile-driven auto-configuration |
| **Adapter** | `LLMAdvisorAdapter` — bridges `BaseLLMHandler` into the `AIAdvisor` interface |
| **Registry** | `LLMHandlerRegistry`, `ResponseSchemaRegistry`, `ModelSelectorFactory`, `OptimizerFactory` — all open for extension |

### SOLID Compliance

| Principle | How applied |
|---|---|
| **SRP** | Each class/module has one reason to change |
| **OCP** | Every registry has `.register()` — extend without modifying existing code |
| **LSP** | All concrete strategies are valid substitutes for their abstract base |
| **ISP** | Interfaces are small: `DataSource.read()`, `Evaluator.evaluate()`, `BaseLLMHandler._call_llm()` |
| **DIP** | All cross-module dependencies are on abstract interfaces, never concrete classes |

---

## Configuration Reference

### OrchestratorConfig

#### Required

| Field | Type | Description |
|---|---|---|
| `source` | `str \| pd.DataFrame` | Dataset file path or in-memory DataFrame |
| `target_column` | `str` | Name of the target column |
| `task_type` | `str` | `"classification"` or `"regression"` |
| `model_provider_model_name` | `str` | `"Provider:ModelName"`. Defaults to `"Groq:none"` — only required when `tune_models=True` |

#### I/O and logging

| Field | Default | Description |
|---|---|---|
| `log_dir` | `"logs/"` | Root directory for experiment folders |
| `experiment_name` | `"exp"` | Tag prepended to the timestamped folder |
| `result_csv_path` | `"results/"` | Directory for the evaluation CSV |
| `model_dir_path` | `"models/"` | Fallback only. At runtime, models are saved to `<experiment_dir>/models/` |

#### Training

| Field | Default | Description |
|---|---|---|
| `test_size` | `0.2` | Held-out fraction for evaluation |
| `preprocess` | `True` | Build and apply adaptive preprocessing pipeline |
| `parallel_training` | `False` | Train models in a multiprocessing pool |
| `n_jobs` | `-1` | Worker count (`-1` = all available CPUs) |
| `models_library` | `"sklearn"` | Model library backend |

#### Tuning

| Field | Default | Description |
|---|---|---|
| `tune_models` | `True` | Run AI-guided Optuna tuning. Set `False` to skip LLM and save top-K models directly |
| `n_trials` | `25` | Optuna trial count |
| `optimizer_method` | `"optuna"` | Optimizer backend (extendable via `OptimizerFactory.register`) |
| `hyperparameter_framework` | `"Optuna"` | Framework name looked up in `ResponseSchemaRegistry` |

#### Model selection

| Field | Default | Description |
|---|---|---|
| `profiling_strategy` | `"extended"` | `"basic"` or `"extended"` data profiling depth |
| `selection_strategy` | `"topk"` | `"topk"`, `"stability"`, or `"generalization"` |
| `top_k` | `3` | Number of candidates forwarded to the optimizer |
| `selector_primary_metric` | `None` | Override primary metric (auto-derives: Accuracy / R2) |
| `selector_stability_weight` | `0.2` | Instability penalty for `stability` strategy |
| `selector_train_metric` | `None` | Train-set metric column for `generalization` strategy |
| `selector_gap_penalty` | `0.5` | Train/test gap penalty for `generalization` strategy |

### SelectorConfig

All three selector strategies share the same constructor signature. The factory always receives a `SelectorConfig` so the orchestrator never needs to know which strategy is active:

```python
from mltunex.model_registry.selector import SelectorConfig, ModelSelectorFactory

cfg = SelectorConfig(
    primary_metric   = "Accuracy",
    stability_weight = 0.3,             # used by StabilityAwareSelector
    train_metric     = "Train_Accuracy", # used by GeneralizationSelector
    gap_penalty      = 0.4,             # used by GeneralizationSelector
)
selector = ModelSelectorFactory.create("stability", config=cfg)
top_df   = selector.select(results_df, k=3)
```

### LLMHandlerConfig

Provider-agnostic configuration used by all LLM handlers:

```python
from mltunex.ai_handler.llm_handler_base import LLMHandlerConfig

config = LLMHandlerConfig(
    model_name    = "qwen/qwen3-32b",
    temperature   = 0.0,
    system_prompt = "...",         # must contain all 4 placeholders
    framework     = "Optuna",      # looked up in ResponseSchemaRegistry
)
```

---

## Experiment Artefacts

Every run creates a single self-contained timestamped directory:

```
logs/
└── exp_20250502_143022/
    ├── pipeline.log                   structured log of every pipeline event
    ├── preprocessing_report.md        dataset profile, missing data, skewness,
    │                                  pipeline steps used
    ├── preprocessed_train.csv         training data after preprocessing
    ├── preprocessed_test.csv          test data after preprocessing
    ├── model_metrics.xlsx             3 sheets: All Models · Ranked · Failed
    │                                  (header styling, best row highlighted)
    ├── selection_report.md            selector config, full leaderboard,
    │                                  selected candidates, summary
    ├── tuning_report.md               AI advisor usage, search spaces,
    │                                  trial history with medals, best config
    └── models/
        ├── preprocessing_pipeline.joblib
        └── RandomForestClassifier.joblib
```

The `tuning_report.md` always includes an **AI Advisor — Usage Summary** section:

| Property | Value |
|---|---|
| Provider | `Groq` |
| Model | `qwen/qwen3-32b` |
| LLM calls | `1` |
| Prompt tokens | `1,234` |
| Completion tokens | `567` |
| **Total tokens** | **`1,801`** |
| Fallback used | `False` |

If all 3 LLM retries fail, `Fallback used: True` is shown with a note explaining that predefined search spaces from `FallbackHyperparams` were used instead.

### Inference with saved artefacts

```python
import joblib, pandas as pd

pipeline = joblib.load("logs/exp_20250502_143022/models/preprocessing_pipeline.joblib")
model    = joblib.load("logs/exp_20250502_143022/models/RandomForestClassifier.joblib")

X_new  = pd.read_csv("new_data.csv").drop(columns=["target"])
preds  = model.predict(pipeline.transform(X_new))
```

---

## AI Schema Validation and Fallback

MLTuneX validates every LLM response through a three-layer system before passing it to Optuna:

**Layer 1 — Schema validation** (`HyperparamSchema.validate()`):
- Top-level must be a non-empty JSON array
- Every entry must have `model_name` (string) and `suggested_hyperparameters` (dict)
- Each parameter definition must have a valid `type` from `{int, float, categorical, bool, fixed}`
- `int`/`float` require `low < high` as numbers
- `categorical` requires a non-empty `values` list
- `fixed` requires a `value` key

**Layer 2 — Retry loop**: Up to 3 attempts with 1.5s / 3s back-off. Each failure is logged to `pipeline.log` at `WARNING` level with the exact error.

**Layer 3 — Predefined fallback** (`FallbackHyperparams`): If all 3 attempts fail, returns predefined search spaces covering 28 built-in models:

| Family | Models |
|---|---|
| Tree / Ensemble | RandomForest, ExtraTrees, DecisionTree, AdaBoost, BaggingClassifier |
| Boosting | XGBoost, LightGBM, GradientBoosting |
| Linear | LogisticRegression, Ridge, Lasso, ElasticNet, SGD |
| SVM | SVC, SVR, LinearSVC |
| Neighbours | KNeighbors |
| Naive Bayes | GaussianNB, BernoulliNB |
| Discriminant | LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis |
| Other | HuberRegressor, LinearRegression |

Unknown models receive an empty `suggested_hyperparameters` dict and are skipped in tuning without crashing.

---

## Extending MLTuneX

No core module needs modification for any of the following. Every extension point uses a `.register()` class method.

### Custom Data Source

```python
from mltunex.data.sources import DataSource, DataSourceFactory
import pandas as pd

class JSONDataSource(DataSource):
    def __init__(self, path: str, **kw):
        self._path, self._kw = path, kw

    def read(self) -> pd.DataFrame:
        return pd.read_json(self._path, **self._kw)

DataSourceFactory.register(".json", JSONDataSource)
# Now works: OrchestratorConfig(source="data.json", ...)
```

### Custom Preprocessing Step

```python
from mltunex.preprocessing import PreprocessingStrategy, PreprocessingPipelineBuilder
import numpy as np

class LogTransformer(PreprocessingStrategy):
    def __init__(self, columns: list): self._cols = columns
    def fit(self, df): return self
    def transform(self, df):
        out = df.copy()
        for col in self._cols:
            if col in out.columns:
                out[col] = np.log1p(out[col].clip(lower=0))
        return out

pipeline = (
    PreprocessingPipelineBuilder()
    .add_numeric_imputer()
    .add_custom_step("log_transform", LogTransformer(["income", "age"]))
    .add_standard_scaler()
    .build()
)
```

### Custom Model Selector

```python
from mltunex.model_registry.selector import (
    ModelSelector, SelectorConfig, ModelSelectorFactory
)
import pandas as pd

class F1FocusedSelector(ModelSelector):
    def __init__(self, config: SelectorConfig):
        self._metric = "f1"

    def select(self, results_df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
        return results_df.nlargest(k, self._metric).reset_index(drop=True)

ModelSelectorFactory.register("f1_focused", F1FocusedSelector)

# Use it:
config = OrchestratorConfig(..., selection_strategy="f1_focused")
```

### Custom LLM Provider

```python
from mltunex.ai_handler.llm_handler_base import (
    BaseLLMHandler, LLMHandlerConfig, LLMHandlerRegistry
)

class MyVendorHandler(BaseLLMHandler):
    provider_name = "MyVendor"

    def _call_llm(self, prompt_vars: dict) -> str:
        # Call your own API. Return raw text.
        # The base class handles: retry, JSON repair, schema validation, fallback.
        return my_api.complete(
            system = prompt_vars["HyperparameterResponsePrompt"],
            user   = prompt_vars["Data_Profile"],
        )

LLMHandlerRegistry.register(MyVendorHandler)

# Use it:
config = OrchestratorConfig(
    ...,
    model_provider_model_name = "MyVendor:my-model",
)
```

To also capture token usage, override `_call_llm_with_obj` instead, returning `(str, response_obj)`:

```python
def _call_llm_with_obj(self, prompt_vars: dict):
    resp = my_api.complete(...)
    return resp.text, resp   # base class calls token_usage.record(resp)
```

### Custom Tuning Framework Schema

```python
from mltunex.ai_handler.response_schema_registry import (
    TuningResponseSchema, ResponseSchemaRegistry
)

class RayTuneSchema(TuningResponseSchema):
    @property
    def framework_name(self) -> str:
        return "RayTune"

    def format_instructions(self) -> str:
        return """Return a JSON array. Each entry: {"model_name": "<str>",
"suggested_hyperparameters": {"<param>": {"type": "uniform", "lower": 0.0, "upper": 1.0}}}"""

    def validate(self, raw):
        if not isinstance(raw, list) or not raw:
            raise ValueError("Expected a non-empty JSON array.")
        return raw

ResponseSchemaRegistry.register(RayTuneSchema())

# Use it:
config = OrchestratorConfig(..., hyperparameter_framework="RayTune")
```

### Custom Optimizer

```python
from mltunex.hyperparam_tuner.optimizer import Optimizer, OptimizerFactory
from typing import Dict, Any, Tuple
import random

class RandomSearchOptimizer(Optimizer):
    def __init__(self, task_type: str, n_trials: int = 25):
        self._task_type = task_type
        self._n_trials  = n_trials

    def optimize(self, model_search_spaces, x_train, y_train,
                 trained_models) -> Tuple[str, Dict[str, Any]]:
        space = random.choice(model_search_spaces)
        return space["model_name"], {}

OptimizerFactory.register("random", RandomSearchOptimizer)

config = OrchestratorConfig(..., optimizer_method="random")
```

---

## Supported LLM Providers

| Provider | String format | Example |
|---|---|---|
| Groq | `Groq:model-name` | `Groq:qwen/qwen3-32b` |
| OpenAI | `OpenAI:model-name` | `OpenAI:gpt-4o` |
| Custom | `MyVendor:model-name` | Register via `LLMHandlerRegistry.register(MyHandler)` |

Model name validation is no longer enforced by MLTuneX — any model name is accepted and the provider API returns a clear error if the model is unknown. This allows you to use newly released models without updating the library.

---

## Troubleshooting

### `UnicodeEncodeError` on Windows terminal

```bash
chcp 65001              # set console code page to UTF-8
# or
set PYTHONIOENCODING=utf-8
```

### LightGBM `[LightGBM] [Warning]` or `[Info]` messages in terminal

MLTuneX sets `LIGHTGBM_VERBOSITY=-1` and `LIGHTGBM_SILENT=1` automatically at import time. If messages still appear (common in Jupyter where the kernel pre-imports LightGBM before MLTuneX sets the env vars), set them before importing MLTuneX:

```python
import os
os.environ["LIGHTGBM_VERBOSITY"] = "-1"
os.environ["LIGHTGBM_SILENT"]    = "1"

from mltunex.orchestrator import MLTuneXOrchestrator, OrchestratorConfig
```

Or from the terminal:

```bash
set LIGHTGBM_VERBOSITY=-1      # Windows
export LIGHTGBM_VERBOSITY=-1   # Linux / macOS
```

### `--llm is required when AI tuning is enabled`

Either provide `--llm "Groq:qwen/qwen3-32b"` or add `--no-tune` to skip AI tuning entirely. No API key is needed in no-tune mode.

### LLM returns invalid JSON — fallback fires

After 3 failed attempts, MLTuneX automatically falls back to predefined search spaces for known models and logs a warning. The run continues. Check `pipeline.log` in the experiment directory for the exact error from each attempt.

### `Exception ignored in sys.unraisablehook` from `rich.console` in Jupyter

This was caused by Rich's `Live` display context manager conflicting with Jupyter's output system. MLTuneX now auto-detects Jupyter / Colab (`ZMQInteractiveShell`) and uses plain text output instead of Rich Live. No action needed — update to the latest version if you still see this.

### `pyarrow.lib.ArrowTypeError` in Streamlit

This is handled internally — mixed-type columns (e.g. hyperparameter values containing int, float, str, None) are stringified before display. If you see this from your own data, cast the problematic column: `df["col"] = df["col"].astype(str)`.

### `pipeline.log` not created

On Windows, ensure the terminal is set to UTF-8 (`chcp 65001`) before launching. The pipeline logger avoids all non-ASCII characters in log messages but the terminal encoding must still support file creation.

### `ModuleNotFoundError: No module named 'lightgbm'`

```bash
pip install lightgbm
```

### Out-of-memory with parallel training

```python
config = OrchestratorConfig(..., parallel_training=True, n_jobs=2)
```

Or use the default sequential mode (`parallel_training=False`).

---

## Contributing

1. Fork the repository
2. Create a branch: `git checkout -b feature/my-feature`
3. Follow the SOLID / registry-based design conventions
4. Add tests for new functionality
5. Open a pull request with a description of what changed and why

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
  Built by <a href="https://github.com/ayuk007">Ayush Nashine</a>
</div>