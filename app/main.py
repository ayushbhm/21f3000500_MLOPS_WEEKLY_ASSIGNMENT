from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

model = joblib.load("model.joblib")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(data: IrisInput):
    features = np.array([[ 
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])

    pred = model.predict(features)[0]
    prob = model.predict_proba(features).max()

    # 🔥 CRITICAL FIX
    pred = int(pred)
    prob = float(prob)

    return {
        "prediction": pred,
        "confidence": prob
    }
