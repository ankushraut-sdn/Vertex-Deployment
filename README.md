# Google Vertex AI — Benefits Eligibility Model

## 1. Project Overview

**Project Name:** Vertex-Deployment (Benefits Eligibility Model)

### Description

This project packages and deploys a Benefits Eligibility Classifier to Google Cloud Vertex AI using a Custom Prediction Routine (CPR). The model predicts whether an employee qualifies for company benefits (`eligible` / `not_eligible`) based on 7 employment-profile features, using a scikit-learn `GradientBoostingClassifier`.

## 2. Core Features

- Gradient Boosting classifier trained on 7 employment features (age, employment type, years of service, weekly hours, department, salary band, dependents) to predict benefits eligibility.
- Vertex AI CPR-compliant `predictor.py` implementing the `Predictor` interface: `load()`, `preprocess()`, `predict()`, `postprocess()`.
- Accepts the Vertex AI request envelope `{"instances": [...]}` and returns `{"predictions": [...]}` with per-record label, top probability, and full class-probability breakdown.
- Governance metadata tracking owner, risk level, protected attributes (`age`, `has_dependents`), training date, and hyperparameters, intended for monitoring via Vertex AI Model Monitoring.

## 3. Tech Stack

| Component | Technology |
|---|---|
| Model | scikit-learn `GradientBoostingClassifier`, XGBoost |
| Data handling | pandas, numpy, joblib |
| Platform | Google Cloud Vertex AI (Custom Prediction Routine, Model Registry, Cloud Storage) |

## 4. Project Structure

```
Vertex-Deployment/
├── predictor.py             # Vertex AI Custom Prediction Routine (load/preprocess/predict/postprocess)
├── doc/                     # architecture diagram
├── *_guide.md                # deployment/runbook guides (ML, NN, SDOH variants)
└── LICENSE
```

## 5. Prerequisites

- A Google Cloud project with Vertex AI and Cloud Storage enabled
- A trained model bundle (`model.pkl`, `preprocessor.joblib`, `label_map.json`) — this repository contains the CPR predictor only, not a training script; see the guide documents below for the training walkthrough
- A `requirements.txt` for the CPR container image (not included in this repository — add one listing scikit-learn, xgboost, pandas, numpy, joblib before building the image)

## 6. How to Run

### Train

This repository does not include a training script. Produce `model.pkl`, `preprocessor.joblib`, and `label_map.json` using your own pipeline (see `benefits_ml_vertex_deployment_guide.md` / `benefits_nn_vertex_guide.md` for the full training walkthrough).

### Deploy

1. Upload the `model/` artifacts to a Cloud Storage bucket accessible by your Vertex AI service account.
2. Build and push a CPR container image using `predictor.py` and a `requirements.txt` you supply, via the Vertex AI SDK's `local_model` helper or Cloud Build.
3. Register the model in the Vertex AI Model Registry, pointing `artifact_uri` at the GCS path.
4. Deploy to a Vertex AI Endpoint; requests must use the `{"instances": [...]}` envelope.
5. Monitor `age` and `has_dependents` (protected attributes) for bias drift using Vertex AI Model Monitoring.

### Feature schema

| Feature | Type | Allowed values |
| :--- | :--- | :--- |
| age | int | 22-65 |
| employment_type | string | `full_time`, `part_time`, `contractor` |
| years_of_service | int | 0-40 |
| weekly_hours | int | 10-55 |
| department | string | `engineering`, `sales`, `hr`, `finance`, `operations` |
| salary_band | string | `band_1`...`band_5` |
| has_dependents | int | 0 or 1 |

## 7. Known Limitations

- No training script or `requirements.txt` is included in this repository — it covers the CPR predictor only.
- Bias/drift monitoring on protected attributes is a manual Vertex AI Model Monitoring setup step, not automated in this codebase.

## 8. License

MIT License (see `LICENSE`).
