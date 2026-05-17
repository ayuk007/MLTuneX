from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="MLTuneX",
    version="0.2.0",
    author="Ayush Nashine",
    author_email="ayush.nashine4807@gmail.com",
    description="Automated Machine Learning Fine-Tuning System.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ayuk007/MLTuneX",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "scikit-learn",
        "pandas",
        "numpy",
        "langchain",
        "openai",
        "langchain-openai",
        "langchain-groq",
        "langchain-community",
        "langchain-core",
        "optuna",
        "python-dotenv",
        "openpyxl",
        "rich",
        "json-repair",
        "streamlit>=1.30",
        "xgboost",
        "lightgbm",
    ],
    extras_require={
        "catboost": ["catboost"],
        "parquet":  ["pyarrow"],
        "feather":  ["pyarrow"],
    },
    entry_points={
        "console_scripts": [
            "mltunex=mltunex.cli:main",
        ],
    },
    python_requires=">=3.9",
)
