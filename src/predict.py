import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:8100")
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

if __name__ == "__main__":
    iris   = load_iris(as_frame=True)
    X      = iris.data
    X.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    sample = X.head(5)
    model  = fetch_model()
    preds  = model.predict(sample)
    iris2  = load_iris()
    print("\nSample predictions:")
    for i, p in enumerate(preds):
        label = iris2.target_names[p] if isinstance(p, (int, np.integer)) else p
        print(f"  Sample {i+1}: {label}")
