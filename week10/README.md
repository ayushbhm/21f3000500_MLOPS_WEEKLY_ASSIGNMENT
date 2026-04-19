# Week 10 — LLM Fine-Tuning: IRIS Dataset

## Task 1 — v1 Raw Feature Format
Each sample serialized as raw numeric features with `input_text`/`output_text` fields.
File: `iris_v1_train_submission.jsonl`
Example: `{"input_text": "sepal_length: 5.1, sepal_width: 3.5, petal_length: 1.4, petal_width: 0.2", "output_text": "setosa"}`

## Task 2 — v2 Natural Language Format
Each sample expressed as a natural language sentence with `input_text`/`output_text` fields.
File: `iris_v2_train_submission.jsonl`
Example: `{"input_text": "A flower specimen has a sepal length of 5.1 cm, sepal width of 3.5 cm, petal length of 1.4 cm, and petal width of 0.2 cm. Identify the iris species.", "output_text": "This is Iris setosa."}`

## Task 3 — Fine-Tuning on Vertex AI
- Base model: gemini-2.5-pro
- Both jobs: 40 epochs, learning rate multiplier 5, adapter size 4
- v1 tuning job: projects/262041725178/locations/europe-west4/tuningJobs/1083257409943633920
- v2 tuning job: projects/262041725178/locations/europe-west4/tuningJobs/32792796859465728

## Task 4 — Evaluation Results

| Metric | v1 Raw Features | v2 Natural Language |
|--------|----------------|---------------------|
| Accuracy | 93.33% | 86.67% |
| Format Compliance | 100.00% | 100.00% |
| Setosa Precision | 1.00 | 0.71 |
| Setosa Recall | 1.00 | 1.00 |
| Versicolor Precision | 0.90 | 1.00 |
| Versicolor Recall | 0.90 | 0.80 |
| Virginica Precision | 0.90 | 1.00 |
| Virginica Recall | 0.90 | 0.80 |

## Finding
v1 (raw numeric) outperformed v2 (natural language) with 93.33% vs 86.67% accuracy.
gemini-2.5-pro is powerful enough to learn numeric boundaries directly from raw features.
v2 occasionally generated extra explanatory text ("Here's a breakdown...") after the
species name — while our parser still extracted the correct species, this represents
the format compliance failure mode unique to LLMs that traditional classifiers never exhibit.
Both models achieved 100% format compliance meaning all responses contained a valid species name.
