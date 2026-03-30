import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
iris = load_iris()
clf = RandomForestClassifier()
clf.fit(iris.data, iris.target)
joblib.dump(clf, 'model.joblib')
print("Model saved!")
