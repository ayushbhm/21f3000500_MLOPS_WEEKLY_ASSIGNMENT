import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("iris_mlsecops_poisoning")

iris = load_iris()
df = pd.DataFrame(iris.data, columns=["sepal_length","sepal_width","petal_length","petal_width"])
df["label"] = iris.target
df.to_csv("iris_clean.csv", index=False)

def poison(df, pct):
    df = df.copy()
    n = int(len(df)*pct)
    idx = np.random.choice(len(df), n, replace=False)
    for i in idx:
        df.loc[i] = [
            np.random.uniform(4,8),
            np.random.uniform(2,5),
            np.random.uniform(1,7),
            np.random.uniform(0.1,2.5),
            np.random.choice([0,1,2])
        ]
    return df

poison(df,0.05).to_csv("iris_poison_5.csv", index=False)
poison(df,0.10).to_csv("iris_poison_10.csv", index=False)
poison(df,0.50).to_csv("iris_poison_50.csv", index=False)

X = df.drop("label", axis=1)
y = df["label"]
_, X_test, _, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

def run(file, lvl):
    d = pd.read_csv(file)
    X = d.drop("label", axis=1)
    y = d["label"]
    X_train, _, y_train, _ = train_test_split(X,y,test_size=0.2,random_state=42)

    m = RandomForestClassifier()
    m.fit(X_train,y_train)
    p = m.predict(X_test)

    with mlflow.start_run():
        mlflow.log_param("poisoning_level", lvl)
        mlflow.log_metric("accuracy", accuracy_score(y_test,p))
        mlflow.log_metric("precision", precision_score(y_test,p,average="weighted"))
        mlflow.log_metric("recall", recall_score(y_test,p,average="weighted"))
        mlflow.log_metric("f1", f1_score(y_test,p,average="weighted"))
        mlflow.sklearn.log_model(m,"model")

run("iris_clean.csv",0)
run("iris_poison_5.csv",5)
run("iris_poison_10.csv",10)
run("iris_poison_50.csv",50)
