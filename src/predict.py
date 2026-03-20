import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://34.59.123.187:8100")
MODEL_REGISTRY_NAME = "IRIS_DecisionTree"

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

def predict(input_data):
    model       = fetch_model()
    predictions = model.predict(input_data)
    return predictions

if __name__ == "__main__":
    df     = pd.read_csv("iris.csv")
    X      = df.drop("species", axis=1)
    sample = X.head(5)
    results = predict(sample)
    print("\nSample predictions:")
    for i, label in enumerate(results):
        print(f"  Sample {i+1}: {label}")
