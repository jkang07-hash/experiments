from setuptools import setup, find_packages

setup(
    name="groundwork",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "anthropic",
        "scikit-learn",
        "datasets",
    ],
)
