# IRIS ML Pipeline – Week 01

## Objective
Set up a cloud-based ML pipeline for IRIS classification using Vertex AI and Google Cloud Storage.

## Implementation
- Uploaded `iris.csv` to GCS.
- Trained model in Vertex AI using data from GCS.
- Stored model artifacts in GCS under timestamp-based folders.
- Ran training twice to create two model versions.
- Created `inference.py` to:
  - Download model from GCS
  - Download dataset from GCS
  - Run predictions and print accuracy
- Pushed all code to private GitHub repo under `week_01` branch.

## Repository Files
- `SDK_Custom_Container_Prediction.ipynb` – Training workflow
- `inference.py` – Inference script


