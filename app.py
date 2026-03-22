import os
import mlflow
import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://34.59.123.187:8100")
MODEL_REGISTRY_NAME = "IRIS_DecisionTree"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load model at startup
client = mlflow.MlflowClient()
try:
    model = mlflow.sklearn.load_model(f"models:/{MODEL_REGISTRY_NAME}@champion")
    print("Loaded model with alias: champion")
except Exception:
    versions = client.get_latest_versions(MODEL_REGISTRY_NAME)
    latest = max(versions, key=lambda v: int(v.version))
    model = mlflow.sklearn.load_model(f"models:/{MODEL_REGISTRY_NAME}/{latest.version}")
    print(f"Loaded latest version: {latest.version}")

app = FastAPI()

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def root():
    return {"status": "IRIS API is running"}

@app.post("/predict")
def predict(data: IrisInput):
    features = [[data.sepal_length, data.sepal_width,
                 data.petal_length, data.petal_width]]
    prediction = model.predict(features)[0]
    species = ["setosa", "versicolor", "virginica"]
    label = prediction if isinstance(prediction, str) else species[int(prediction)]
    return {"prediction": label}
