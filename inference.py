import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

bucket = "gs://week1-12-02-mlops-course-dulcet-bastion-452612-v4-unique"
folder = "20260212_184730"

model_path = f"{bucket}/my-models/iris-classifier-week-1/{folder}/model.joblib"
data_path = f"{bucket}/iris.csv"


os.system(f"gsutil cp {model_path} model.joblib")
os.system(f"gsutil cp {data_path} iris.csv")

model = joblib.load("model.joblib")

df = pd.read_csv("iris.csv")

X = df.drop("species", axis=1)
y = df["species"]


_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pred = model.predict(X_test)

print("Inference Accuracy:", accuracy_score(y_test, pred))
