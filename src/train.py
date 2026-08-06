from pathlib import Path

import hydra
import joblib
import pandas as pd
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/processed/train.csv")
MODEL_PATH = Path("models/model.pkl")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):

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

    # Train model
    model = LogisticRegression(max_iter=cfg.model.max_iter)
    model.fit(X_train, y_train)

    # Save everything needed for evaluation
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "X_test": X_test,
            "y_test": y_test,
        },
        MODEL_PATH,
    )

    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()