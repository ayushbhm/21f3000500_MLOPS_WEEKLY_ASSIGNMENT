import pytest
import joblib
import json
import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

@pytest.fixture(scope="module")
def model():
    assert os.path.exists('model/iris_model.pkl'), "Model file missing"
    return joblib.load('model/iris_model.pkl')

@pytest.fixture(scope="module")
def test_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    X = df[iris.feature_names]
    y = df['target']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_test, y_test

def test_model_file_exists():
    assert os.path.exists('model/iris_model.pkl')

def test_metrics_file_exists():
    assert os.path.exists('model/metrics.json')

def test_accuracy_threshold(model, test_data):
    X_test, y_test = test_data
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    assert acc >= 0.90, f"Accuracy {acc} below threshold"

def test_prediction_classes(model, test_data):
    X_test, _ = test_data
    preds = model.predict(X_test)
    assert set(preds).issubset({0, 1, 2}), "Unexpected prediction classes"

def test_metrics_values():
    with open('model/metrics.json') as f:
        metrics = json.load(f)
    assert metrics['accuracy'] >= 0.90, "Accuracy below threshold"
    assert metrics['precision'] >= 0.90, "Precision below threshold"
