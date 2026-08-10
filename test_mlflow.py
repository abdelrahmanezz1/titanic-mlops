import dagshub
import mlflow

dagshub.init(
    repo_owner="abdelrahmanezz1",
    repo_name="titanic-mlops",
    mlflow=True,
)

print("Tracking URI:", mlflow.get_tracking_uri())

with mlflow.start_run():
    mlflow.log_param("model", "test")
    mlflow.log_metric("accuracy", 0.95)

print("Experiment logged successfully!")