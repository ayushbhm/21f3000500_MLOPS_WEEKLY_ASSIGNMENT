# MLflow IRIS Project

Simple ML pipeline using MLflow + DVC + CI on IRIS dataset.

## Files

- `src/train.py` → trains model + logs experiments to MLflow  
- `src/evaluate.py` → loads best model from MLflow and evaluates  
- `src/predict.py` → makes sample predictions  
- `.github/workflows/ci.yml` → runs evaluation & prediction in CI  
- `iris.csv.dvc` → tracks dataset using DVC  
- `requirements.txt` → dependencies  

