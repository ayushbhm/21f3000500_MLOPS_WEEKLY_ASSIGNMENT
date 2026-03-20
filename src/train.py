import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import itertools

MLFLOW_TRACKING_URI = "http://34.59.123.187:8100"
EXPERIMENT_NAME     = "IRIS_Hyperparameter_Tuning"
MODEL_REGISTRY_NAME = "IRIS_DecisionTree"

# Task 1: Hyperparameter grid with 2 hyperparameters
PARAM_GRID = {
    "max_depth":         [3, 4, 5],
    "min_samples_split": [2, 5],
}

def load_data():
    df = pd.read_csv("iris.csv")
    X  = df.drop("species", axis=1)
    y  = df["species"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train, max_depth, min_samples_split):
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall":    recall_score(y_test, y_pred, average="weighted"),
        "f1_score":  f1_score(y_test, y_pred, average="weighted"),
    }

def run_training():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_data()

    best_accuracy = 0.0
    best_run_id   = None
    best_params   = {}

    keys   = list(PARAM_GRID.keys())
    combos = list(itertools.product(*PARAM_GRID.values()))

    print(f"Running {len(combos)} hyperparameter combinations...")

    for combo in combos:
        params            = dict(zip(keys, combo))
        max_depth         = params["max_depth"]
        min_samples_split = params["min_samples_split"]
        run_name          = f"dt_depth{max_depth}_split{min_samples_split}"

        with mlflow.start_run(run_name=run_name):

            # Task 2a: Log hyperparameters
            mlflow.log_param("max_depth",         max_depth)
            mlflow.log_param("min_samples_split", min_samples_split)
            mlflow.log_param("random_state",      42)
            mlflow.log_param("test_size",         0.2)
            mlflow.set_tag("model_type", "DecisionTreeClassifier")
            mlflow.set_tag("dataset",    "IRIS")

            # Train
            model   = train_model(X_train, y_train, max_depth, min_samples_split)

            # Task 2b: Log metrics
            metrics = evaluate_model(model, X_test, y_test)
            for metric_name, metric_val in metrics.items():
                mlflow.log_metric(metric_name, metric_val)

            print(f"  {run_name} => accuracy={metrics['accuracy']:.4f} f1={metrics['f1_score']:.4f}")

            # Task 2c: Log model to registry
            sample_X  = X_train.sample(min(100, len(X_train)), random_state=42)
            signature = infer_signature(sample_X, model.predict(sample_X))

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="iris_model",
                signature=signature,
                input_example=X_train.head(5),
                registered_model_name=MODEL_REGISTRY_NAME
            )

            current_run_id = mlflow.active_run().info.run_id
            if metrics["accuracy"] > best_accuracy:
                best_accuracy = metrics["accuracy"]
                best_run_id   = current_run_id
                best_params   = params

    print(f"\nBest run ID  : {best_run_id}")
    print(f"Best params  : {best_params}")
    print(f"Best accuracy: {best_accuracy:.4f}")
    print(f"View UI at   : {MLFLOW_TRACKING_URI}")
    return best_run_id

if __name__ == "__main__":
    run_training()
