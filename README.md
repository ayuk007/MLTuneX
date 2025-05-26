# 🤖 MLTuneX - AutoML Framework for Model Training and Hyperparameter Tuning

**MLTuneX** is a powerful and extensible AutoML library designed to make machine learning model training and hyperparameter tuning easy, customizable, and scalable.

🚀 With support for preprocessed data (currently), the library can:
- Train multiple models
- Evaluate their performance
- Tune top models using **Optuna** and **OpenAI GPT-based guidance**
- Save the best-performing model

⚙️ Currently supports:
- **Model Library**: `scikit-learn`
- **Tuning Framework**: `Optuna`

🧪 Upcoming support:
- Grid Search
- Random Search
- Ray Tune
- OpenAI-based advanced tuning agents

---


> ⚠️ **NOTE:** As of now, only preprocessed data is supported. You must provide a dataset that is already cleaned and encoded. Automated raw data handling is planned in upcoming versions.

---

## 📦 Installation

Install the package directly using pip:

```bash
pip install --no-cache-dir MLTuneX
```

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

```python
from mltunex.main import MLTuneX

mltunex = MLTuneX(
    data="/path/to/your/preprocessed_data.csv",  # Must be a cleaned CSV or pandas DataFrame
    target_column="your_target_column",          # Specify the target column
    task_type="regression"                       # Choose between "regression" or "classification"
)

mltunex.run(
    result_csv_path="/path/to/save/csv",         # Directory to store evaluation results
    model_dir_path="/path/to/save/models",       # Directory to save the best model
    tune_models="yes"                            # "yes" to enable hyperparameter tuning
)
```