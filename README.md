<div align="center">
  <h1>🤖 MLTuneX</h1>
  <p><strong>Automated Machine Learning Fine-Tuning System</strong></p>

  <p>
    <a href="https://pypi.org/project/MLTuneX/"><img src="https://img.shields.io/pypi/v/MLTuneX?color=blue" alt="PyPI version" /></a>
    <a href="https://pypi.org/project/MLTuneX/"><img src="https://img.shields.io/pypi/pyversions/MLTuneX" alt="Python versions" /></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License" /></a>
    <a href="https://github.com/ayuk007/MLTuneX/issues"><img src="https://img.shields.io/github/issues/ayuk007/MLTuneX" alt="Issues" /></a>
  </p>

  <p>
    <a href="#quick-start">Quick Start</a> •
    <a href="#installation">Installation</a> •
    <a href="#usage">Usage</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#configuration">Configuration</a> •
    <a href="#extending-mltunex">Extending</a>
  </p>
</div>

---

MLTuneX is a production-grade AutoML library that takes your dataset from raw CSV to
a fine-tuned model in a single command. It trains every applicable model in your
chosen library, selects the top candidates, asks an LLM for smart hyperparameter
search spaces, and runs Optuna to find the best configuration — all with zero
boilerplate.

```
Dataset  →  Profile  →  Preprocess  →  Train (all models)  →  Select top-K
        →  AI advisor generates search spaces
        →  Optuna optimises  →  Best model + fitted pipeline saved to disk
        →  Full experiment logged (Markdown reports + Excel metrics)
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
- [Extending MLTuneX](#extending-mltunex)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## Features

| Feature | Detail |
|---|---|
| **Zero-boilerplate AutoML** | One function call trains, evaluates, selects, and tunes every model |
| **AI-guided tuning** | Groq / OpenAI LLMs suggest hyperparameter search spaces tailored to your data |
| **Adaptive preprocessing** | Profile-driven pipeline (imputation, scaling, encoding, outlier clipping) built automatically |
| **Parallel training** | Optional multiprocessing pool — train all models simultaneously |
| **3 interfaces** | Python API, CLI (`mltunex`), and Streamlit UI (`mltunex ui`) |
| **Rich experiment logs** | Preprocessing report, model metrics Excel, selection report, tuning report — all Markdown / XLSX |
| **Artefact persistence** | Best model AND fitted preprocessing pipeline saved as `.joblib` files |
| **Pluggable everything** | Data sources, models, preprocessing steps, selectors, optimisers — all extensible without modifying core code |

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

### Streamlit UI (optional)

```bash
pip install streamlit            # already included in base install
```

### API keys

MLTuneX needs an API key for the LLM advisor. Set it as an environment variable
**before** running (or enter it directly in the Streamlit UI):

```bash
# Groq (free tier available at console.groq.com)
export GROQ_API_KEY="gsk_..."

# OpenAI
export OPENAI_API_KEY="sk-..."
```

---

## Quick Start

### 30-second example

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

That single call will:
1. Load and profile your data
2. Build and fit a preprocessing pipeline
3. Train every sklearn / XGBoost / LightGBM classifier
4. Evaluate each one and save a ranked Excel spreadsheet
5. Ask the LLM for hyperparameter search spaces for the top 3 models
6. Run 25 Optuna trials to find the best configuration
7. Retrain with the best params and save the model + pipeline to `models/`
8. Write 4 Markdown reports to `logs/exp_<timestamp>/`

---

## Usage

### Python API

#### Minimal classification example

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

#### Regression with full customisation

```python
from mltunex.orchestrator import MLTuneXOrchestrator, OrchestratorConfig

config = OrchestratorConfig(
    # Data
    source          = "data/housing.parquet",
    target_column   = "SalePrice",
    task_type       = "regression",
    test_size       = 0.15,

    # AI advisor
    model_provider_model_name = "OpenAI:gpt-4o",

    # Training
    preprocess        = True,
    parallel_training = True,
    n_jobs            = 8,

    # Tuning
    tune_models  = True,
    n_trials     = 50,

    # Selection — stability-aware: penalise high-variance models
    selection_strategy        = "stability",
    top_k                     = 5,
    selector_primary_metric   = "R2",
    selector_stability_weight = 0.3,

    # Output
    result_csv_path = "results/",
    model_dir_path  = "models/",
    log_dir         = "logs/",
    experiment_name = "housing_v1",
)

MLTuneXOrchestrator(config).run()
```

#### Skip tuning — just train and save top models

```python
config = OrchestratorConfig(
    source        = "data/fraud.csv",
    target_column = "is_fraud",
    task_type     = "classification",
    model_provider_model_name = "Groq:qwen/qwen3-32b",

    tune_models   = False,   # select top-K and save them; no LLM call needed
    top_k         = 3,
    selection_strategy = "generalization",
    selector_train_metric = "Train_Accuracy",
)
MLTuneXOrchestrator(config).run()
```

#### Load saved artefacts for inference

```python
import joblib
import pandas as pd

# Load fitted preprocessing pipeline
pipeline = joblib.load("models/preprocessing_pipeline.joblib")

# Load best model
model = joblib.load("models/RandomForestClassifier.joblib")

# Run inference on new data
new_data = pd.read_csv("new_customers.csv")
X_new    = new_data.drop(columns=["Survived"])
X_proc   = pipeline.transform(X_new)
preds    = model.predict(X_proc)
```

#### Using an in-memory DataFrame

```python
import pandas as pd
from mltunex.orchestrator import MLTuneXOrchestrator, OrchestratorConfig

df = pd.read_csv("data.csv")

config = OrchestratorConfig(
    source        = df,          # pass DataFrame directly
    target_column = "label",
    task_type     = "classification",
    model_provider_model_name = "Groq:qwen/qwen3-32b",
)
MLTuneXOrchestrator(config).run()
```

---

### Command-Line Interface

After installation, the `mltunex` command is available system-wide.

#### Basic usage

```bash
mltunex --data titanic.csv \
        --target Survived \
        --task classification \
        --llm "Groq:qwen/qwen3-32b"
```

#### All options

```bash
mltunex --help
```

```
usage: mltunex [-h] --data PATH --target COLUMN --task {classification,regression}
               --llm PROVIDER:MODEL [OPTIONS]

required arguments:
  --data PATH           Dataset file (.csv, .xlsx, .parquet, .feather)
  --target COLUMN       Target column name
  --task                classification | regression
  --llm PROVIDER:MODEL  LLM for AI tuning (e.g. Groq:qwen/qwen3-32b)

training options:
  --test-size FLOAT     Test split fraction (default: 0.2)
  --no-preprocess       Skip adaptive preprocessing
  --parallel            Train models in parallel
  --jobs N              Worker count (-1 = all CPUs)
  --library LIB         Model library (default: sklearn)

tuning options:
  --no-tune             Skip AI hyperparameter optimisation
  --trials N            Optuna trial count (default: 25)

model selection:
  --selector            topk | stability | generalization (default: topk)
  --top-k N             Candidates forwarded to optimiser (default: 3)
  --primary-metric      Override ranking metric (default: Accuracy / R2)

output:
  --results DIR         Results CSV directory (default: results/)
  --models DIR          Saved model directory (default: models/)
  --log-dir DIR         Experiment log root (default: logs/)
  --exp-name NAME       Experiment folder tag (default: exp)
```

#### CLI examples

```bash
# Regression, parallel training, 50 trials, custom output
mltunex --data housing.csv --target price --task regression \
        --llm "OpenAI:gpt-4o" --parallel --jobs 4 --trials 50 \
        --results out/results/ --models out/models/ --log-dir out/logs/

# Skip tuning, stability selector, top 5 models
mltunex --data fraud.csv --target label --task classification \
        --llm "Groq:qwen/qwen3-32b" --no-tune \
        --selector stability --top-k 5

# Extended profiling, generalization selector
mltunex --data churn.csv --target churned --task classification \
        --llm "Groq:qwen/qwen3-32b" --selector generalization \
        --train-metric Train_Accuracy
```

---

### Streamlit UI

Launch the interactive browser UI:

```bash
mltunex ui
```

Or equivalently:

```bash
python -m mltunex.ui
```

The UI opens at `http://localhost:8501` and provides:

- **Configuration sidebar** — upload data, set all options, enter API keys securely
- **Live Progress tab** — real-time event cards and training table as each model finishes
- **Data Profile tab** — dataset overview, missing data, skewness chart
- **Model Results tab** — sortable leaderboard + bar chart
- **Selection tab** — selected candidates with strategy explanation
- **Tuning tab** — score progression chart, full trial history
- **Reports tab** — inline Markdown reports with download buttons, Excel preview

> **Note:** API keys entered in the UI are stored only in memory for the duration
> of the browser session and are never written to disk.

---

## Architecture

### Pipeline Stages

```
Input (CSV / Excel / Parquet / Feather / DataFrame / SQL)
    |
    v
[1] Data Ingestion          DataSourceFactory  ->  DataSource.read()  ->  DataFrame
    |
    v
[2] Data Profiling          DataProfilerFactory  ->  DataProfiler.profile()  ->  metadata dict
    |
    v
[3] Train / Test Split      Data_Splitter
    |
    v
[4] Preprocessing           AdaptivePipelineDirector builds PreprocessingPipeline
    |                       from profile metadata  (impute, clip, encode, scale)
    |                       Fitted pipeline saved to  models/preprocessing_pipeline.joblib
    v
[5] Model Training          Model_Registry  ->  LibraryTrainer  ->  trained models
    |                       Optional: multiprocessing Pool (parallel_training=True)
    |                       Live terminal table (rich) updated after each model
    v
[6] Evaluation              EvaluatorFactory  ->  Evaluator.evaluate()  ->  metrics dict
    |                       Results saved to  results/<uuid>.csv
    |                       model_metrics.xlsx written to experiment dir
    v
[7] Model Selection         ModelSelectorFactory  ->  ModelSelector.select()  ->  top-K DataFrame
    |                       selection_report.md written to experiment dir
    v
[8] AI Search Spaces        LLMAdvisorAdapter  ->  AIAdvisor.suggest_search_spaces()
    |                       Adapter wraps GroqHyperparamGenerator / OpenAIHyperparamGenerator
    v
[9] Optimisation            OptunaOptimizer.optimize()  ->  best model + params
    |                       Trial history recorded; tuning_report.md written
    v
[10] Retrain + Save         Best model retrained with optimal params
     |                      Model saved to  models/<ModelName>.joblib
     v
     Run Summary            Absolute paths to every saved artefact printed / emitted
```

### Design Patterns

| Pattern | Where used |
|---|---|
| **Facade** | `MLTuneXOrchestrator` — single `run()` call hides all complexity |
| **Factory** | `DataSourceFactory`, `DataProfilerFactory`, `EvaluatorFactory`, `ModelSelectorFactory`, `OptimizerFactory` |
| **Strategy** | `DataProfiler`, `PreprocessingStrategy`, `ModelSelector`, `Evaluator`, `Optimizer` |
| **Builder** | `PreprocessingPipelineBuilder` — fluent pipeline assembly |
| **Director** | `AdaptivePipelineDirector` — profile-driven auto-configuration |
| **Adapter** | `LLMAdvisorAdapter` — decouples LLM backends from `OptunaOptimizer` |

### SOLID Compliance

| Principle | How it is applied |
|---|---|
| **SRP** | Each class/module has one reason to change |
| **OCP** | All factories have a `.register()` method — extend without modifying |
| **LSP** | Every concrete strategy is a valid drop-in for its interface |
| **ISP** | Interfaces are small (`DataSource.read()`, `Evaluator.evaluate()`, …) |
| **DIP** | All cross-module dependencies are on abstract interfaces |

---

## Configuration Reference

All options are exposed through `OrchestratorConfig`.

### Core (required)

| Field | Type | Description |
|---|---|---|
| `source` | `str \| pd.DataFrame` | Path to dataset file or in-memory DataFrame |
| `target_column` | `str` | Name of the target column |
| `task_type` | `str` | `"classification"` or `"regression"` |
| `model_provider_model_name` | `str` | LLM string: `"Provider:ModelName"` |

### I/O

| Field | Default | Description |
|---|---|---|
| `result_csv_path` | `"results/"` | Directory for evaluation CSV |
| `model_dir_path` | `"models/"` | Directory for saved model + pipeline |
| `log_dir` | `"logs/"` | Root directory for experiment folders |
| `experiment_name` | `"exp"` | Tag prepended to the experiment folder name |

### Training

| Field | Default | Description |
|---|---|---|
| `test_size` | `0.2` | Held-out fraction for evaluation |
| `preprocess` | `True` | Run adaptive preprocessing pipeline |
| `parallel_training` | `False` | Train models in a multiprocessing pool |
| `n_jobs` | `-1` | Worker count (`-1` = all CPUs) |
| `models_library` | `"sklearn"` | Model library backend |

### Tuning

| Field | Default | Description |
|---|---|---|
| `tune_models` | `True` | Run AI-guided Optuna tuning |
| `n_trials` | `25` | Number of Optuna trials |
| `optimizer_method` | `"optuna"` | Optimizer backend |

### Model Selection

| Field | Default | Description |
|---|---|---|
| `profiling_strategy` | `"extended"` | `"basic"` or `"extended"` |
| `selection_strategy` | `"topk"` | `"topk"`, `"stability"`, or `"generalization"` |
| `top_k` | `3` | Candidates forwarded to the optimizer |
| `selector_primary_metric` | `None` | Override ranking metric (auto-derives from task) |
| `selector_stability_weight` | `0.2` | Instability penalty for `stability` strategy |
| `selector_train_metric` | `None` | Train-set metric column for `generalization` strategy |
| `selector_gap_penalty` | `0.5` | Train/test gap penalty for `generalization` strategy |

---

## Experiment Artefacts

Every run creates a timestamped directory under `log_dir/`:

```
logs/
└── exp_20250502_143022/
    ├── pipeline.log             ← structured log of every pipeline event
    ├── preprocessing_report.md  ← dataset profile, missing data, pipeline steps
    ├── preprocessed_train.csv   ← transformed training data
    ├── preprocessed_test.csv    ← transformed test data
    ├── model_metrics.xlsx       ← All Models / Ranked / Failed sheets
    ├── selection_report.md      ← selector config, leaderboard, chosen candidates
    └── tuning_report.md         ← AI search spaces, trial history, best config

models/
├── preprocessing_pipeline.joblib   ← fitted pipeline (use for inference)
└── RandomForestClassifier.joblib   ← best retrained model
```

### Using saved artefacts for inference

```python
import joblib, pandas as pd

pipeline = joblib.load("models/preprocessing_pipeline.joblib")
model    = joblib.load("models/RandomForestClassifier.joblib")

raw   = pd.read_csv("new_data.csv")
X_new = raw.drop(columns=["Survived"])
X_pp  = pipeline.transform(X_new)
print(model.predict(X_pp))
```

---

## Extending MLTuneX

MLTuneX is designed so that new capabilities are added by **registering** a new
class, never by modifying existing code.

### Add a new data source format

```python
from mltunex.data.sources import DataSource, DataSourceFactory
import pandas as pd

class JSONDataSource(DataSource):
    def __init__(self, path: str, **kw):
        self._path = path
        self._kw = kw

    def read(self) -> pd.DataFrame:
        return pd.read_json(self._path, **self._kw)

DataSourceFactory.register(".json", JSONDataSource)
```

### Add a new model

```python
from mltunex.model_registry.model_interface import SklearnModel
from sklearn.neural_network import MLPClassifier

# Register directly in the sklearn registry
from mltunex.model_registry.sklearn_registry import SkLearn_Model_Registry
SkLearn_Model_Registry.ADDITIONAL_CLASSIFIERS.append(
    ("MLPClassifier", MLPClassifier)
)
```

### Add a custom preprocessing step

```python
from mltunex.preprocessing import PreprocessingStrategy, PreprocessingPipelineBuilder
import pandas as pd

class LogTransformer(PreprocessingStrategy):
    def __init__(self, columns: list):
        self._cols = columns

    def fit(self, df: pd.DataFrame) -> "LogTransformer":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        import numpy as np
        out = df.copy()
        for col in self._cols:
            if col in out.columns:
                out[col] = np.log1p(out[col].clip(lower=0))
        return out

# Use in a manual pipeline
pipeline = (
    PreprocessingPipelineBuilder()
    .add_numeric_imputer()
    .add_custom_step("log_transform", LogTransformer(["income", "age"]))
    .add_standard_scaler()
    .build()
)
```

### Add a new model selector strategy

```python
from mltunex.model_registry.selector import ModelSelector, SelectorConfig, ModelSelectorFactory
import pandas as pd

class F1FocusedSelector(ModelSelector):
    def __init__(self, config: SelectorConfig):
        self._metric = "f1"

    def select(self, results_df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
        return results_df.nlargest(k, self._metric).reset_index(drop=True)

ModelSelectorFactory.register("f1_focused", F1FocusedSelector)

# Then use it
config = OrchestratorConfig(
    ...,
    selection_strategy = "f1_focused",
)
```

### Add a new optimizer

```python
from mltunex.hyperparam_tuner.optimizer import Optimizer, OptimizerFactory
from typing import Dict, Any, Tuple
import pandas as pd

class RandomSearchOptimizer(Optimizer):
    def __init__(self, task_type: str, n_trials: int = 25):
        self._task_type = task_type
        self._n_trials  = n_trials

    def optimize(self, model_search_spaces, x_train, y_train,
                 trained_models) -> Tuple[str, Dict[str, Any]]:
        import random
        space = random.choice(model_search_spaces)
        return space["model_name"], {}

OptimizerFactory.register("random", RandomSearchOptimizer)
```

---

## Troubleshooting

### `UnicodeEncodeError` on Windows

Set the console code page to UTF-8 before running:

```bash
chcp 65001
mltunex --data data.csv ...
```

Or set the environment variable:

```bash
set PYTHONIOENCODING=utf-8
```

### LightGBM / XGBoost warnings in terminal

These come from C-level libraries. MLTuneX suppresses them automatically via fd-level
redirection and environment variables. If you still see them, set:

```bash
export LIGHTGBM_VERBOSITY=-1
```

### `ValueError: AI advisor returned empty search-space list`

The LLM returned no valid JSON. Check:
1. Your API key is set and valid
2. The model name is correct (e.g. `Groq:qwen/qwen3-32b`, not `Groq:qwen`)
3. The LLM is reachable (network / firewall)

### Out-of-memory with parallel training

Reduce the worker count:

```python
config = OrchestratorConfig(..., parallel_training=True, n_jobs=2)
```

Or disable parallel training and use sequential mode (the default).

### `ModuleNotFoundError: No module named 'lightgbm'`

```bash
pip install lightgbm
```

---

## Supported LLM Providers

| Provider | Format | Example |
|---|---|---|
| Groq | `Groq:model-name` | `Groq:qwen/qwen3-32b` |
| OpenAI | `OpenAI:model-name` | `OpenAI:gpt-4o` |

New providers can be added by implementing the `AIAdvisor` interface and registering
a matching `LLMConfig` entry.

---

## Contributing

Contributions are welcome. Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Follow the existing SOLID / design-pattern conventions
4. Add tests for new functionality
5. Open a pull request with a clear description

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
  Built with care by <a href="https://github.com/ayuk007">Ayush Nashine</a>
</div>
