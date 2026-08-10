from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report
import mlflow

from titanic_mlops.mlflow_setup import *

MODEL_PATH = Path("models/model.pkl")
EXPERIMENTS_FILE = Path("experiments.md")


def log_experiment(model_type, test_size, accuracy):
    if not EXPERIMENTS_FILE.exists():
        return
    
    with open(EXPERIMENTS_FILE, "r") as f:
        lines = f.readlines()
        
    insert_idx = -1
    last_exp = 0
    for i, line in enumerate(lines):
        if line.startswith("| ") and not line.startswith("| Experiment |") and not line.startswith("| :---"):
            try:
                last_exp = int(line.split("|")[1].strip())
                insert_idx = i + 1
            except ValueError:
                pass
                
    if insert_idx != -1:
        new_row = f"| {last_exp + 1} | {model_type} | {test_size} | {accuracy:.4f} | Auto-logged run |\n"
        lines.insert(insert_idx, new_row)
        
        with open(EXPERIMENTS_FILE, "w") as f:
            f.writelines(lines)


def main():
    # Load saved model and test data
    saved = joblib.load(MODEL_PATH)

    model = saved["model"]
    X_test = saved["X_test"]
    y_test = saved["y_test"]
    
    # Metadata for logging
    test_size = saved.get("test_size", "unknown")
    model_type = saved.get("model_type", "unknown")
    run_id = saved.get("run_id")

    # Predictions
    predictions = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.4f}\n")
    print(classification_report(y_test, predictions))
    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
    )

    if run_id:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", report["weighted avg"]["precision"])
            mlflow.log_metric("recall", report["weighted avg"]["recall"])
            mlflow.log_metric("f1_score", report["weighted avg"]["f1-score"])
    else:
        # Fallback if run_id wasn't saved (e.g. from an old model)
        with mlflow.start_run():
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", report["weighted avg"]["precision"])
            mlflow.log_metric("recall", report["weighted avg"]["recall"])
            mlflow.log_metric("f1_score", report["weighted avg"]["f1-score"])
    
    # Automatically log to experiments.md
    log_experiment(model_type, test_size, accuracy)
    print("Logged experiment results to experiments.md")


if __name__ == "__main__":
    main()