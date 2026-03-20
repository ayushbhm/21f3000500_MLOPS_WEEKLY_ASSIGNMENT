import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:8100")
MODEL_REGISTRY_NAME = "IRIS_DecisionTree"

def load_data():
    iris = load_iris(as_frame=True)
    X    = iris.data
    y    = iris.target  # numeric: 0, 1, 2
    X.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_test, y_test

def fetch_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    try:
        model_uri = f"models:/{MODEL_REGISTRY_NAME}@champion"
        model     = mlflow.sklearn.load_model(model_uri)
        print(f"Loaded model with alias: champion")
    except Exception:
        versions  = client.get_latest_versions(MODEL_REGISTRY_NAME)
        latest    = max(versions, key=lambda v: int(v.version))
        model_uri = f"models:/{MODEL_REGISTRY_NAME}/{latest.version}"
        model     = mlflow.sklearn.load_model(model_uri)
        print(f"Loaded latest version: {latest.version}")
    return model

def evaluate():
    model          = fetch_model()
    X_test, y_test = load_data()
    y_pred         = model.predict(X_test)

    # Convert both to same type (string) to avoid mix error
    iris         = load_iris()
    y_test_names = [iris.target_names[i] for i in y_test]
    y_pred_names = y_pred if isinstance(y_pred[0], str) else [iris.target_names[i] for i in y_pred]

    metrics = {
        "accuracy":  accuracy_score(y_test_names, y_pred_names),
        "precision": precision_score(y_test_names, y_pred_names, average="weighted"),
        "recall":    recall_score(y_test_names, y_pred_names, average="weighted"),
        "f1_score":  f1_score(y_test_names, y_pred_names, average="weighted"),
    }

    print("\n── Evaluation Results ──────────────")
    for name, val in metrics.items():
        print(f"  {name:12s}: {val:.4f}")

    print("\n── Classification Report ───────────")
    print(classification_report(y_test_names, y_pred_names, target_names=iris.target_names))
    return metrics

if __name__ == "__main__":
    evaluate()
