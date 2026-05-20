<div align="center">
  <h1>🤖 MLTuneX</h1>
  <p><strong>Automated Machine Learning Fine-Tuning System</strong></p>
  <p>
    <a href="https://pypi.org/project/MLTuneX/"><img src="https://img.shields.io/pypi/v/MLTuneX?color=6366f1" alt="PyPI" /></a>
    <a href="https://pypi.org/project/MLTuneX/"><img src="https://img.shields.io/pypi/pyversions/MLTuneX" alt="Python" /></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-22c55e" alt="License" /></a>
  </p>
  <p>
    <a href="https://ayuk007.github.io/mltunex_docs/">📖 Full Documentation</a> •
    <a href="https://ayuk007.github.io/mltunex_docs/installation.html">Installation</a> •
    <a href="https://ayuk007.github.io/mltunex_docs/quickstart.html">Quick Start</a> •
    <a href="https://ayuk007.github.io/mltunex_docs/python-api.html">API Reference</a> •
    <a href="https://ayuk007.github.io/mltunex_docs/architecture.html">Architecture</a>
  </p>
</div>

---

MLTuneX is a production-grade AutoML library that takes your dataset from a raw CSV to a fine-tuned model in a single command, no boilerplate, no guesswork.

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

## Installation

```bash
pip install MLTuneX
```

**Optional extras**

```bash
pip install MLTuneX[catboost]   # CatBoost support
pip install MLTuneX[parquet]    # Parquet / Feather file support
```

**API keys** (only needed when `tune_models=True`)

```bash
export GROQ_API_KEY="gsk_..."      # Groq free tier at console.groq.com
export OPENAI_API_KEY="sk-..."     # OpenAI
```

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

One call handles everything: data profiling → adaptive preprocessing → training all models → evaluation → AI-guided hyperparameter search spaces → Optuna optimisation → saving the best model and all experiment artefacts.

**No API key? Use no-tune mode:**

```python
config = OrchestratorConfig(
    source        = "data/fraud.csv",
    target_column = "is_fraud",
    task_type     = "classification",
    tune_models   = False,   # skips LLM entirely no API key needed
)
MLTuneXOrchestrator(config).run()
```

**CLI:**

```bash
mltunex --data titanic.csv --target Survived --task classification \
        --llm "Groq:qwen/qwen3-32b"
```

**Streamlit UI:**

```bash
mltunex ui    # opens at http://localhost:8501
```

---

## Features

| Feature | Detail |
|---|---|
| **Zero-boilerplate AutoML** | One function call covers the full pipeline end to end |
| **AI-guided tuning** | LLM-generated hyperparameter search spaces tailored to your dataset profile |
| **Pluggable LLM registry** | Groq and OpenAI built-in; attach any custom LLM in ~10 lines |
| **Schema-validated AI output** | 3-retry loop with exponential back-off; predefined fallback for 28 models |
| **Token usage tracking** | Prompt / completion / total tokens logged per run |
| **Adaptive preprocessing** | Profile-driven pipeline built automatically imputation, encoding, scaling |
| **Parallel training** | Optional multiprocessing pool all models trained simultaneously |
| **Jupyter / Colab support** | Auto-detects runtime, uses plain output in notebooks |
| **3 interfaces** | Python API · CLI (`mltunex`) · Streamlit UI (`mltunex ui`) |
| **Experiment artefacts** | Timestamped directory per run: models, reports, Excel metrics, structured log |
| **Pluggable selectors** | TopK, StabilityAware, GeneralizationAware or register your own |
| **No-tune mode** | Select and save top-K models without any API key |

---

## Experiment Artefacts

Every run produces a self-contained timestamped directory:

```
logs/
└── exp_20250502_143022/
    ├── pipeline.log
    ├── preprocessing_report.md
    ├── preprocessed_train.csv
    ├── preprocessed_test.csv
    ├── model_metrics.xlsx          ← 3 sheets: All Models · Ranked · Failed
    ├── selection_report.md
    ├── tuning_report.md            ← AI usage, search spaces, trial history
    └── models/
        ├── preprocessing_pipeline.joblib
        └── RandomForestClassifier.joblib
```

**Inference:**

```python
import joblib, pandas as pd

pipeline = joblib.load("logs/exp_20250502_143022/models/preprocessing_pipeline.joblib")
model    = joblib.load("logs/exp_20250502_143022/models/RandomForestClassifier.joblib")

X_new = pd.read_csv("new_data.csv").drop(columns=["target"])
preds = model.predict(pipeline.transform(X_new))
```

---

## Documentation

The full documentation is hosted at **[ayuk007.github.io/mltunex-docs](https://ayuk007.github.io/mltunex-docs/)** and covers:

- [Installation & API key setup](https://ayuk007.github.io/mltunex_docs/installation.html)
- [Quick Start walkthrough](https://ayuk007.github.io/mltunex_docs/quickstart.html)
- [Python API reference](https://ayuk007.github.io/mltunex_docs/python-api.html)
- [CLI reference](https://ayuk007.github.io/mltunex_docs/cli.html)
- [Streamlit UI guide](https://ayuk007.github.io/mltunex_docs/streamlit-ui.html)
- [Architecture & design patterns](https://ayuk007.github.io/mltunex_docs/architecture.html)
- [Full configuration reference](https://ayuk007.github.io/mltunex_docs/configuration.html)
- [Experiment artefacts](https://ayuk007.github.io/mltunex_docs/artefacts.html)
- [AI schema validation & fallback](https://ayuk007.github.io/mltunex_docs/ai-validation.html)
- [Custom extensions](https://ayuk007.github.io/mltunex_docs/extending.html)
- [Custom LLM providers](https://ayuk007.github.io/mltunex_docs/llm-providers.html)
- [Troubleshooting](https://ayuk007.github.io/mltunex_docs/troubleshooting.html)

---

## Contributing

1. Fork the repository
2. Create a branch: `git checkout -b feature/my-feature`
3. Follow the SOLID / registry-based design conventions
4. Add tests for new functionality
5. Open a pull request

---

## License

MIT License: see [LICENSE](LICENSE) for details.

---

<div align="center">
  Built by <a href="https://github.com/ayuk007">Ayush Nashine</a>
</div>