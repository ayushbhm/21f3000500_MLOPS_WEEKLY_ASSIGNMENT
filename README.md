# MLSecOps Week 8 Assignment

## Overview
This project demonstrates data poisoning attacks on the IRIS dataset and analyzes their impact on model performance using MLflow.

---

## Folder Structure

### week8_mlsecops/

#### 1. run_mlsecops.py
- Main script for the entire pipeline
- Performs:
  - Loading IRIS dataset
  - Creating poisoned datasets (5%, 10%, 50%)
  - Training RandomForest models
  - Logging experiments in MLflow

---

#### 2. iris_clean.csv
- Original IRIS dataset
- Used as baseline (0% poisoning)

---

#### 3. iris_poison_5.csv
- Dataset with 5% samples corrupted
- Minor noise introduced

---

#### 4. iris_poison_10.csv
- Dataset with 10% samples corrupted
- Moderate degradation starts

---

#### 5. iris_poison_50.csv
- Dataset with 50% samples corrupted
- Severe corruption, behaves like random data

---

#### 6. mlruns/
- MLflow tracking directory
- Stores:
  - Experiment runs
  - Metrics (accuracy, precision, recall, F1)
  - Model artifacts

---

## Tasks Summary

### Task 2: Data Poisoning
- Randomly replaced feature values
- Assigned random class labels
- Simulates adversarial data injection

---

### Task 3: MLflow Experiments
- Trained 4 models (0%, 5%, 10%, 50%)
- Logged:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- Compared performance using MLflow UI

---

### Task 4: Analysis
- 5% → negligible impact
- 10% → noticeable degradation
- 50% → accuracy ~33% (random guessing)

---

### Task 5: Mitigation Strategies
- Statistical validation
- Anomaly detection
- Data provenance tracking
- Schema enforcement

---

## Conclusion
Model performance degrades as poisoning increases.
Data quality is more important than data quantity.
