# Week 6 — Continuous Deployment: IRIS ML API on GKE

## What This Does
Every push to `week_6` branch automatically builds a Docker image, pushes it to Google Artifact Registry, and deploys it to Google Kubernetes Engine as a live prediction API.

## File Structure

**`app.py`** — FastAPI inference server. Loads the IRIS DecisionTree model from MLflow Model Registry and exposes `GET /` (health check) and `POST /predict` (returns predicted species).

**`Dockerfile`** — Packages app.py and dependencies into a Docker image. Runs uvicorn on port 8000.

**`requirements.txt`** — Python dependencies: FastAPI, uvicorn, scikit-learn, MLflow, pandas, joblib.

**`k8s/deployment.yaml`** — Kubernetes Deployment. Runs 1 replica. Image URL injected at deploy time by GitHub Actions.

**`k8s/service.yaml`** — Kubernetes LoadBalancer Service. Exposes the API publicly on port 80.

**`.github/workflows/cd.yml`** — CD pipeline. On push to week_6: authenticates to GCP via Workload Identity, builds and pushes Docker image to Artifact Registry, deploys to GKE.

**`src/train.py`** — Hyperparameter tuning over max_depth and min_samples_split, logs all runs to MLflow, registers best model.

**`src/evaluate.py`** — Loads champion model from MLflow, evaluates on IRIS test set.

**`src/predict.py`** — Sanity check, runs predictions on 5 sample inputs.

## Test the Live API
curl -X POST http://34.134.116.187/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
