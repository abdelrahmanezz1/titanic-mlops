from pathlib import Path

import hydra
import joblib
import pandas as pd
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import mlflow

from titanic_mlops.mlflow_setup import *
import mlflow
import titanic_mlops.mlflow_setup

DATA_PATH = Path("data/processed/train.csv")
MODEL_PATH = Path("models/model.pkl")


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    print("Tracking URI in train.py:", mlflow.get_tracking_uri())
    with mlflow.start_run() as run:
        # Load processed data
        df = pd.read_csv(DATA_PATH)

        # Features and target
        X = df.drop(columns=["Survived"])
        y = df["Survived"]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=cfg.train.test_size,
            random_state=cfg.train.random_state,
        )

        # Select model based on Hydra configuration
        if cfg.model.type == "logistic":
            model = LogisticRegression(max_iter=cfg.model.max_iter)

        elif cfg.model.type == "decision_tree":
            model = DecisionTreeClassifier(
                random_state=cfg.train.random_state
            )

        elif cfg.model.type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=cfg.model.n_estimators,
                random_state=cfg.train.random_state,
            )

        else:
            raise ValueError(f"Unknown model type: {cfg.model.type}")

        mlflow.log_param("model", cfg.model.type)
        mlflow.log_param("test_size", cfg.train.test_size)
        mlflow.log_param("random_state", cfg.train.random_state)
        
        if cfg.model.type == "random_forest":
            mlflow.log_param("n_estimators", cfg.model.n_estimators)

        # Train model
        model.fit(X_train, y_train)

        # Save model and test data
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "model": model,
                "X_test": X_test,
                "y_test": y_test,
                "model_type": cfg.model.type,
                "test_size": cfg.train.test_size,
                "run_id": run.info.run_id,
            },
            MODEL_PATH,
        )

        print("=" * 50)
        print(f"Model Type : {cfg.model.type}")
        print(f"Test Size  : {cfg.train.test_size}")
        print(f"Model saved to {MODEL_PATH}")
        print("=" * 50)


if __name__ == "__main__":
    main()