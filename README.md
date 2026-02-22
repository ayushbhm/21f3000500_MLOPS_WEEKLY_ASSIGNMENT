

## Overview
This assignment integrates DVC (Data Version Control) into the IRIS machine learning pipeline developed in Week 1. The goal is to version control the dataset and model artifacts, configure Google Cloud Storage (GCS) as remote storage, and demonstrate reproducibility across multiple training iterations.

## Files Included

train.py  
- Loads the IRIS dataset  
- Saves dataset as data/iris.csv  
- Trains RandomForest model  
- Saves model as models/model.pkl  
- In second iteration, simulates additional data before retraining  

inference.py  
- Loads trained model  
- Performs predictions  

data/iris.csv.dvc  
- DVC metadata file tracking dataset version  
- Actual dataset stored in GCS  

models/model.pkl.dvc  
- DVC metadata file tracking model version  
- Actual model stored in GCS  

.dvc/config  
- Contains configuration for GCS remote bucket (iris-dvc-21f3000500)

## Versioning

v1.0 – Original dataset and trained model  
v2.0 – Augmented dataset and retrained model  

Version switching is demonstrated using:
git checkout <tag>
dvc checkout

## Key Points

- Git tracks metadata (.dvc files)  
- DVC tracks large artifacts  
- Data and model stored in GCS  
- Pipeline is fully reproducible   – Inference script


