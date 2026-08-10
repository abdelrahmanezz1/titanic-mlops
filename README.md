# Titanic MLOps

A simple machine learning project to demonstrate MLOps practices on the Titanic dataset.

## Features

- **Dependency Management:** Managed using `uv`.
- **Pipeline & Version Control:** Uses [DVC](https://dvc.org) for data and pipeline versioning (`dvc repro`, `dvc dag`).
- **Configuration Management:** Uses [Hydra](https://hydra.cc) for flexible and modular configuration management (e.g. toggling models and hyperparameters).
- **Experiment Tracking:** Automatically logs evaluation metrics (accuracy) and hyperparameters to `experiments.md`.

## Project Structure

- `src/titanic_mlops/train.py`: Script for training the ML model. Can train Logistic Regression, Decision Tree, or Random Forest models.
- `src/titanic_mlops/evaluate.py`: Script for evaluating the trained model and automatically appending results to the experiment log.
- `experiments.md`: Markdown file tracking the experiment history.
- `dvc.yaml`: DVC pipeline definition.
- `conf/`: Directory containing Hydra YAML configurations.

## Quickstart

1. **Sync Dependencies:** (Ensure `uv` is installed)
   ```bash
   uv sync
   ```

2. **Run DVC Pipeline:**
   ```bash
   uv run dvc repro
   ```

3. **Train a Specific Model Manually (e.g. Random Forest):**
   ```bash
   uv run python src/titanic_mlops/train.py model.type=random_forest
   uv run python src/titanic_mlops/evaluate.py
   ```
**This line will be reverted.**