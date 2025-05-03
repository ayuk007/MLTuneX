from setuptools import setup, find_packages

setup(
    name = "MLTuneX",
    version = "0.1.0",
    author = "Ayush Nashine",
    author_email = "ayush.nashine@gmail.com",
    description = "A package for machine learning tuning and optimization.",
    long_description = open("README.md").read(),
    long_description_content_type = "text/markdown",
    url = "https://github.com/yourusername/MLTuneX",
    packages = find_packages(),
    install_requires = [
        "scikit-learn",
        "pandas",
        "numpy",
        "langchain",
        "openai",
        "langchain-openai",
        "langchain-community",
        "langchain-core",
        "optuna",
        "python-dotenv"
    ],
    python_requires='>=3.8',
)