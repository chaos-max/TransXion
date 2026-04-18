"""Setup configuration for aml_terraformer package."""

from setuptools import setup, find_packages

setup(
    name="aml_terraformer",
    version="0.1.0",
    description="LLM-based perturbation for AML transaction graphs",
    author="AML Research Team",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "networkx>=2.8.0",
        "matplotlib>=3.5.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "openai": ["openai>=1.0.0"],
        "anthropic": ["anthropic>=0.18.0"],
        "deepseek": ["openai>=1.0.0"],
        "local": ["transformers>=4.30.0", "torch>=2.0.0", "peft>=0.4.0"],
        "gbt": ["lightgbm>=3.0.0"],
        "all": [
            "openai>=1.0.0",
            "anthropic>=0.18.0",
            "transformers>=4.30.0",
            "torch>=2.0.0",
            "peft>=0.4.0",
            "lightgbm>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "aml-terraformer=aml_terraformer.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
