# 21f3000500 MLOps Weekly Assignment

## Week 4 — Continuous Integration for IRIS Pipeline

### What's included:
- IRIS classification model using Random Forest
- Data validation tests (pytest)
- Model evaluation tests (pytest)
- GitHub Actions CI pipeline
- DVC for data and model versioning (GCS remote)
- CML for automated test reports on Pull Requests

### How CI works:
1. Every push triggers GitHub Actions
2. DVC pulls versioned data and model from GCS
3. pytest runs data validation and model evaluation tests
4. CML posts a report as a PR comment
