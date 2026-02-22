from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib
import os

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

df.to_csv("data/iris.csv", index=False)

X = df.drop("target", axis=1)
y = df["target"]

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "models/model.pkl")
