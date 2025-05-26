from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name = "MLTuneX",
    version = "0.1.3",
    author = "Ayush Nashine",
    author_email = "ayush.nashine@gmail.com",
    description = "A package for machine learning tuning and optimization.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/ayuk007/MLTuneX",
    packages = find_packages(where='src'),
    package_dir = {'': 'src'},
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