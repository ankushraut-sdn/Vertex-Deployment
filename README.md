# GOOGLE VERTEX AI
## Benefits Eligibility Model
### Technical Specification & Model Documentation

**Version 1.0.0**  •  **March 28, 2026**  •  **HR Analytics Team**

---

## 1. Project Overview

This document describes all files required to deploy the Benefits Eligibility Classifier to Google Cloud Vertex AI using a Custom Prediction Routine (CPR). The model predicts whether an employee qualifies for company benefits based on their employment profile.

| Attribute | Value |
|---|---|
| Use Case | Employee benefits eligibility classification |
| Algorithm | Gradient Boosting Classifier (scikit-learn) |
| Owner | HR Analytics Team |
| Version | 1.0.0 |
| Risk Level | Medium — affects employee compensation |
| Training Date | 2026-03-28 |
| Output Classes | `not_eligible` \| `eligible` |

## 2. Repository & Artifact Structure

All files must be arranged exactly as shown. The `model/` directory is uploaded to Cloud Storage and registered as a Vertex AI Model resource.

```text
benefits-model-vertex/
├── model/
│   ├── model.pkl               ← trained GradientBoostingClassifier
│   ├── preprocessor.joblib     ← fitted ColumnTransformer
│   └── label_map.json          ← {"0": "not_eligible", "1": "eligible"}
├── code/
│   ├── predictor.py            ← Vertex AI Custom Prediction Routine
│   ├── requirements.txt        ← runtime Python dependencies
│   └── config.yaml             ← (optional) deployment config
├── metadata/
│   └── model_metadata.json     ← governance & audit fields
├── test/
│   ├── sample_request.json     ← validation payload
│   └── expected_response.json  ← expected inference output
└── README.md
```

## 3. Artifact 1 — Model Bundle (MANDATORY)

### 3.1 Cloud Storage artifact contents

1. `model.pkl` — serialised `GradientBoostingClassifier`
2. `preprocessor.joblib` — fitted `ColumnTransformer` (`StandardScaler` + `OneHotEncoder`)
3. `label_map.json` — integer-to-class-name mapping

### 3.2 Feature Schema

| Feature | Type | Description / Allowed Values |
|---|---|---|
| age | int | Employee age in years (22–65) |
| employment_type | string | `full_time` \| `part_time` \| `contractor` |
| years_of_service | int | Completed years at the company (0–40) |
| weekly_hours | int | Contracted weekly hours (10–55) |
| department | string | `engineering` \| `sales` \| `hr` \| `finance` \| `operations` |
| salary_band | string | `band_1` \| `band_2` \| `band_3` \| `band_4` \| `band_5` |
| has_dependents | int | Binary flag — `0 = no dependents`, `1 = has dependents` |

### 3.3 label_map.json

```json
{
 "0": "not_eligible",
 "1": "eligible"
}
```

## 4. Artifact 2 — Custom Predictor (MANDATORY)

**File:** `code/predictor.py` — defines the `Predictor` class implementing Vertex AI's Custom Prediction Routine (CPR) interface.

### 4.1 load(artifacts_uri)

Called once after container startup. Vertex AI downloads Cloud Storage artifacts to a local temp directory and passes its path. Loads `model.pkl`, `preprocessor.joblib`, and `label_map.json`.

```python
def load(self, artifacts_uri: str):
    self.model        = joblib.load(os.path.join(artifacts_uri, 'model.pkl'))
    self.preprocessor = joblib.load(os.path.join(artifacts_uri, 'preprocessor.joblib'))
    with open(os.path.join(artifacts_uri, 'label_map.json')) as f:
        self.label_map = json.load(f)
```

### 4.2 preprocess(prediction_input)

Receives the raw Vertex AI request dict `{"instances": [...]}`. Validates all 7 required features are present and returns a pandas DataFrame.

```python
def preprocess(self, prediction_input: dict) -> pd.DataFrame:
    instances = prediction_input.get('instances', prediction_input)
    if isinstance(instances, dict): instances = [instances]
    df = pd.DataFrame(instances)
    missing = set(EXPECTED_FEATURES) - set(df.columns)
    if missing: raise ValueError(f'Missing features: {missing}')
    return df[EXPECTED_FEATURES]
```

### 4.3 predict(instances)

Applies the preprocessor transform then runs `clf.predict` and `clf.predict_proba`. Returns a dict of raw integer predictions and probability arrays.

```python
def predict(self, instances: pd.DataFrame) -> dict:
    X           = self.preprocessor.transform(instances)
    raw_preds   = self.model.predict(X).tolist()
    class_proba = self.model.predict_proba(X).tolist()
    return {'raw_predictions': raw_preds, 'probabilities': class_proba}
```

### 4.4 postprocess(prediction_results)

Converts raw model output into the Vertex AI response envelope `{"predictions": [...]}`. Each prediction record includes a label, max confidence score, and full probability dict.

```python
def postprocess(self, prediction_results: dict) -> dict:
    labeled = [self.label_map[str(p)] for p in prediction_results['raw_predictions']]
    label_order = [self.label_map[k] for k in sorted(self.label_map)]
    return {'predictions': [
        {'label': lbl, 'probability': max(prob),
         'all_proba': dict(zip(label_order, prob))}
        for lbl, prob in zip(labeled, prediction_results['probabilities'])
    ]}
```

## 5. Artifact 3 — Requirements File (MANDATORY)

**File:** `code/requirements.txt` — installed into the CPR container image at build time.

```text
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
xgboost>=2.0
joblib>=1.3
```

## 6. Artifact 4 — Model Metadata

**File:** `metadata/model_metadata.json` — used for the Vertex AI Model Registry, governance, and auditing.

| Field | Value for This Model |
|---|---|
| model_version | 1.0.0 |
| owner | HR Analytics Team |
| use_case | Employee benefits eligibility classification |
| risk_level | medium |
| protected_attributes | age, has_dependents |
| training_date | 2026-03-28 |
| algorithm | GradientBoostingClassifier |
| training_samples | 1,600 |
| test_samples | 400 |

```json
{
  "model_version": "1.0.0",
  "owner": "HR Analytics Team",
  "use_case": "Employee benefits eligibility classification",
  "risk_level": "medium",
  "protected_attributes": ["age", "has_dependents"],
  "training_date": "2026-03-28",
  "algorithm": "GradientBoostingClassifier",
  "hyperparameters": {
    "n_estimators": 200, "learning_rate": 0.05,
    "max_depth": 4, "subsample": 0.8
  }
}
```

## 7. Artifact 5 — Sample Payloads (Validation)

### 7.1 test/sample_request.json

```json
{
  "instances": [
    {
      "age": 34, "employment_type": "full_time",
      "years_of_service": 5, "weekly_hours": 40,
      "department": "engineering", "salary_band": "band_3",
      "has_dependents": 1
    },
    {
      "age": 27, "employment_type": "part_time",
      "years_of_service": 0, "weekly_hours": 20,
      "department": "sales", "salary_band": "band_1",
      "has_dependents": 0
    }
  ]
}
```

### 7.2 test/expected_response.json

```json
{
  "predictions": [
    {"label": "eligible",     "probability": 0.92, "all_proba": {"not_eligible": 0.08, "eligible": 0.92}},
    {"label": "not_eligible", "probability": 0.87, "all_proba": {"not_eligible": 0.87, "eligible": 0.13}}
  ]
}
```

## 8. Deployment Notes

1. Upload the `model/` directory to a Cloud Storage bucket accessible by your Vertex AI service account.
2. Build and push a CPR container image with `predictor.py` and `requirements.txt` using the Vertex AI SDK `local_model` helper or Cloud Build.
3. Register the model in Vertex AI Model Registry, pointing `artifact_uri` at the GCS path.
4. Deploy to a Vertex AI Endpoint; the request envelope must use the `{"instances": [...]}` format.
5. Monitor `protected_attributes` (`age`, `has_dependents`) for bias drift using Vertex AI Model Monitoring.

---

End of Document
