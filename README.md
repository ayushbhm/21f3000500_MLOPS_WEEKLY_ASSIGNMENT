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



week_07 
# MLOps Weekly Assignment - Iris API Deployment on GKE

## 📌 Overview

This project demonstrates the end-to-end deployment of a Machine Learning model using **FastAPI**, **Docker**, and **Google Kubernetes Engine (GKE)**. It includes stress testing, autoscaling, monitoring, and CI/CD integration.

---

## 🚀 Features

- FastAPI-based ML inference API
- Docker containerization
- Deployment on GKE
- Load testing using `wrk`
- Horizontal Pod Autoscaling (HPA)
- Monitoring using GCP Cloud Monitoring & Logging
- CI/CD workflow using GitHub Actions

---

## 📁 Project Structure & File Explanation

### 🔹 Core Application Files

- **`app/main.py`**  
  Contains the FastAPI application.  
  - Defines `/predict` endpoint  
  - Loads trained model  
  - Handles inference logic  

- **`train_model.py`**  
  Script used to train the Iris classification model.  
  - Uses scikit-learn  
  - Saves model as `model.joblib`  

---

### 🔹 Docker & Containerization

- **`Dockerfile`**  
  Defines how the application is containerized.  
  - Installs dependencies  
  - Copies application code  
  - Runs FastAPI server using Uvicorn  

---

### 🔹 Kubernetes Configuration

- **`deployment.yaml` / `k8s/deployment.yaml`**  
  Defines the Kubernetes Deployment.  
  - Specifies container image  
  - Sets CPU & memory requests/limits  
  - Manages pod replicas  

- **`service.yaml` / `k8s/service.yaml`**  
  Exposes the application using a LoadBalancer.  
  - Maps external traffic to container port  

- **`k8s/hpa.yaml`**  
  Defines Horizontal Pod Autoscaler (HPA).  
  - Scales pods based on CPU utilization  
  - Min replicas: 1  
  - Max replicas: 3  

---

### 🔹 Load Testing

- **`post.lua`**  
  Lua script used by `wrk` for sending POST requests.  
  - Defines request body and headers  
  - Simulates API traffic  

---

### 🔹 CI/CD

- **`.github/workflows/ci.yml`**  
  GitHub Actions workflow for CI/CD.  
  - Installs `wrk`  
  - Runs stress test automatically after push  

---

### 🔹 Other Files

- **`main.py` (root)**  
  May contain alternate or earlier version of API logic.

- **`model.joblib`** *(generated)*  
  Trained ML model used for predictions.

