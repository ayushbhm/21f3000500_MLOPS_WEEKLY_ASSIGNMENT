import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import json
import os

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

os.makedirs('data', exist_ok=True)
df.to_csv('data/iris.csv', index=False)

X = df[iris.feature_names]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
metrics = {
    "accuracy": round(accuracy_score(y_test, y_pred), 4),
    "precision": round(precision_score(y_test, y_pred, average='macro'), 4),
    "recall": round(recall_score(y_test, y_pred, average='macro'), 4)
}

os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/iris_model.pkl')

with open('model/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("Training complete. Metrics:", metrics)
