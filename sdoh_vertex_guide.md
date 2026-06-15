# Deploying the SDoH Health Risk Classifier to Google Vertex AI

This guide covers the full path from your local codebase to a live Vertex AI
prediction endpoint. The model is a PyTorch MLP (`SDoHMLP`) trained on Social
Determinants of Health features. Vertex AI's Custom Prediction Routine (CPR)
requires a Docker image — and because this model uses PyTorch rather than
scikit-learn, you must base that image on a PyTorch-aware container and ensure
`model_config.json` is available alongside `model.pt` so the `Predictor` class
can reconstruct the network architecture before loading weights.

---

## Prerequisites

**Google Cloud resources**
- An active GCP project with billing enabled.
- The following APIs enabled:

  ```bash
  gcloud services enable \
      aiplatform.googleapis.com \
      storage.googleapis.com \
      artifactregistry.googleapis.com \
      cloudbuild.googleapis.com
  ```

- A **GCS bucket** in the same region as your endpoint:

  ```bash
  gcloud storage buckets create gs://my-sdoh-model-bucket \
      --location=us-central1
  ```

- A Service Account (or your own identity) with `Vertex AI User` and
  `Storage Object Admin` roles.

**Local tooling**

```bash
pip install google-cloud-aiplatform google-cloud-storage torch
gcloud auth application-default login
gcloud config set project my-gcp-project-id
gcloud config set ai/region us-central1
```

---

## Step 1 — Generate Labeled Training Data (optional)

```bash
python generate_sdoh_examples.py \
    --n 2000 \
    --seed 42 \
    --out-dir ./data
```

Writes `./data/sdoh_examples.csv` and `./data/sdoh_examples.json`.

---

## Step 2 — Train the Model and Build the Vertex Package Locally

```bash
# With your own labeled data
python train_dl_sdoh.py \
    --platform vertex \
    --data-file ./data/sdoh_examples.csv \
    --output-dir ./dist

# With synthetic data (2 000 samples)
python train_dl_sdoh.py \
    --platform vertex \
    --samples 2000 \
    --output-dir ./dist
```

After completion, `./dist/vertex/` contains:

```
dist/vertex/
├── model/
│   ├── model.pt                  ← PyTorch state dict (weights only)
│   ├── model_config.json         ← architecture spec: input_dim, hidden_dims, dropout
│   ├── preprocessor.joblib       ← sklearn ColumnTransformer
│   └── label_map.json
├── code/
│   ├── predictor.py              ← Predictor class: load/preprocess/predict/postprocess
│   └── requirements.txt
├── metadata/
│   └── model_metadata.json
└── test/
    ├── sample_request.json       ← {"instances": [...]}
    └── expected_response.json
```

> **Why model_config.json matters:** PyTorch saves only the weight tensor values
> in `model.pt`, not the class definition. `predictor.py` reads `model_config.json`
> first to reconstruct the `SDoHMLP` architecture, then calls `load_state_dict()`
> to populate the weights. Both files must be present in the GCS artifact path
> that Vertex stages locally at `AIP_STORAGE_URI`.

---

## Step 3 — Upload Model Artifacts to GCS

```bash
BUCKET="my-sdoh-model-bucket"
GCS_PREFIX="sdoh-classifier/v1/model"

gcloud storage cp ./dist/vertex/model/* \
    gs://${BUCKET}/${GCS_PREFIX}/

# Verify all four files are present
gcloud storage ls gs://${BUCKET}/${GCS_PREFIX}/
# Expected:
#   model.pt
#   model_config.json
#   preprocessor.joblib
#   label_map.json
```

---

## Step 4 — Create an Artifact Registry Repository

```bash
gcloud artifacts repositories create sdoh-models \
    --repository-format=docker \
    --location=us-central1 \
    --description="SDoH health risk model images"

gcloud auth configure-docker us-central1-docker.pkg.dev
```

---

## Step 5 — Build and Push a Custom Prediction Container

Vertex AI's CPR requires a Docker image that includes `predictor.py` and all
runtime dependencies. Because this model uses PyTorch, the base image must
include `torch`. The official Vertex AI PyTorch prediction base images are the
correct starting point.

### 5a — Write the Dockerfile

Create `Dockerfile` at the project root:

```dockerfile
# Vertex AI PyTorch prediction base (CPU); swap tag for GPU variant if needed
FROM us-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.2-2:latest

WORKDIR /opt/app

# Copy inference code
COPY dist/vertex/code/predictor.py   ./predictor.py
COPY dist/vertex/code/requirements.txt ./requirements.txt

# Install additional dependencies not in the base image
# (scikit-learn and joblib are needed for the preprocessor)
RUN pip install --no-cache-dir \
    scikit-learn>=1.3 \
    joblib>=1.3 \
    pandas>=2.0

# Vertex CPR convention
ENV HANDLER_MODULE=predictor
ENV HANDLER_CLASS=Predictor
```

> **GPU variant:** Replace the base image with
> `us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.2-2:latest` and deploy to
> a GPU machine type (e.g. `n1-standard-4` + `nvidia-tesla-t4`). For the
> inference volumes typical of this use case, CPU is sufficient.

### 5b — Build and push

```bash
PROJECT_ID="my-gcp-project-id"
REGION="us-central1"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/sdoh-models/sdoh-predictor:v1"

docker build -t ${IMAGE_URI} .
docker push ${IMAGE_URI}
```

Using Cloud Build (no local Docker required):

```bash
gcloud builds submit \
    --tag ${IMAGE_URI} \
    --region=${REGION} \
    .
```

---

## Step 6 — (Optional) Run a Training Job on Vertex AI

### 6a — Create a training container

```dockerfile
# Dockerfile.train
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    scikit-learn>=1.3 joblib>=1.3 pandas>=2.0

COPY train_dl_sdoh.py .
COPY generate_sdoh_examples.py .

ENTRYPOINT ["python", "train_dl_sdoh.py"]
```

```bash
TRAIN_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/sdoh-models/sdoh-trainer:v1"
docker build -f Dockerfile.train -t ${TRAIN_IMAGE} .
docker push ${TRAIN_IMAGE}
```

### 6b — Submit the training job

```python
from google.cloud import aiplatform

aiplatform.init(project="my-gcp-project-id", location="us-central1")

BUCKET      = "my-sdoh-model-bucket"
TRAIN_IMAGE = "us-central1-docker.pkg.dev/my-gcp-project-id/sdoh-models/sdoh-trainer:v1"
IMAGE_URI   = "us-central1-docker.pkg.dev/my-gcp-project-id/sdoh-models/sdoh-predictor:v1"

job = aiplatform.CustomContainerTrainingJob(
    display_name     = "sdoh-training-job",
    container_uri    = TRAIN_IMAGE,
    model_serving_container_image_uri = IMAGE_URI,
)

model = job.run(
    args = [
        "--platform", "vertex",
        "--output-dir", "/tmp/dist",
        "--samples", "2000",
    ],
    replica_count   = 1,
    machine_type    = "n1-standard-8",
    accelerator_type  = "NVIDIA_TESLA_T4",   # remove for CPU-only
    accelerator_count = 1,
    base_output_dir = f"gs://{BUCKET}/sdoh-classifier/training-output",
)
```

---

## Step 7 — Register the Model in Vertex AI Model Registry

```python
from google.cloud import aiplatform

aiplatform.init(project="my-gcp-project-id", location="us-central1")

BUCKET    = "my-sdoh-model-bucket"
IMAGE_URI = "us-central1-docker.pkg.dev/my-gcp-project-id/sdoh-models/sdoh-predictor:v1"

model = aiplatform.Model.upload(
    display_name            = "sdoh-health-risk-classifier",
    artifact_uri            = f"gs://{BUCKET}/sdoh-classifier/v1/model",
    serving_container_image_uri     = IMAGE_URI,
    serving_container_predict_route = "/predict",
    serving_container_health_route  = "/health",
    serving_container_environment_variables = {
        "HANDLER_MODULE": "predictor",
        "HANDLER_CLASS":  "Predictor",
    },
    description = "PyTorch MLP SDoH health risk classifier v1",
    labels      = {"team": "population-health", "stage": "production"},
)

print("Registered model:", model.resource_name)
```

```bash
gcloud ai models upload \
    --region=us-central1 \
    --display-name="sdoh-health-risk-classifier" \
    --artifact-uri="gs://${BUCKET}/sdoh-classifier/v1/model" \
    --container-image-uri="${IMAGE_URI}" \
    --container-health-route="/health" \
    --container-predict-route="/predict"
```

---

## Step 8 — Create the Endpoint and Deploy the Model

### Create the endpoint

```python
endpoint = aiplatform.Endpoint.create(
    display_name = "sdoh-health-risk-endpoint",
    labels       = {"team": "population-health"},
)
print("Endpoint:", endpoint.resource_name)
```

```bash
gcloud ai endpoints create \
    --region=us-central1 \
    --display-name="sdoh-health-risk-endpoint"
```

### Deploy the model to the endpoint

```python
deployed_model = endpoint.deploy(
    model          = model,
    deployed_model_display_name = "sdoh-classifier-v1",
    machine_type   = "n1-standard-4",
    min_replica_count = 1,
    max_replica_count = 3,
    traffic_split  = {"0": 100},
)

print("Deployment complete. Endpoint:", endpoint.resource_name)
```

```bash
MODEL_ID=$(gcloud ai models list \
    --region=us-central1 \
    --filter="displayName=sdoh-health-risk-classifier" \
    --format="value(name)" | head -1)

ENDPOINT_ID=$(gcloud ai endpoints list \
    --region=us-central1 \
    --filter="displayName=sdoh-health-risk-endpoint" \
    --format="value(name)" | head -1)

gcloud ai endpoints deploy-model ${ENDPOINT_ID} \
    --region=us-central1 \
    --model=${MODEL_ID} \
    --display-name="sdoh-classifier-v1" \
    --machine-type=n1-standard-4 \
    --min-replica-count=1 \
    --max-replica-count=3 \
    --traffic-split=0=100
```

Deployment takes 5–10 minutes. Monitor status:

```bash
gcloud ai endpoints describe ${ENDPOINT_ID} \
    --region=us-central1 \
    --format="value(deployedModels[0].createTime)"
```

---

## Step 9 — Run Prediction Against the Live Endpoint

The `predictor.py` entry-point expects `{"instances": [...]}`.

### Python — Vertex AI SDK

```python
from google.cloud import aiplatform

aiplatform.init(project="my-gcp-project-id", location="us-central1")

endpoint = aiplatform.Endpoint(
    endpoint_name = "projects/my-gcp-project-id/locations/us-central1/endpoints/ENDPOINT_ID"
)

instances = [
    {
        "age": 58, "prior_er_visits": 4, "household_size": 5,
        "pct_income_on_housing": 62,
        "income_level": "below_poverty", "insurance_status": "uninsured",
        "food_security": "very_low", "education_level": "less_than_hs",
        "health_literacy": "low", "housing_stability": "unstable",
        "neighborhood_safety": "unsafe", "rural_urban": "rural",
        "social_support": "none", "language_barrier": "yes",
        "has_primary_care": "no"
    },
    {
        "age": 34, "prior_er_visits": 0, "household_size": 3,
        "pct_income_on_housing": 22,
        "income_level": "middle", "insurance_status": "employer",
        "food_security": "secure", "education_level": "bachelors",
        "health_literacy": "high", "housing_stability": "homeowner",
        "neighborhood_safety": "high", "rural_urban": "urban",
        "social_support": "strong", "language_barrier": "no",
        "has_primary_care": "yes"
    }
]

response = endpoint.predict(instances=instances)
print(response.predictions)
# Expected (Predictor.postprocess enriches each prediction):
# [
#   {"label": "high_risk", "probability": 0.91,
#    "all_proba": {"low_risk": 0.09, "high_risk": 0.91}},
#   {"label": "low_risk",  "probability": 0.88,
#    "all_proba": {"low_risk": 0.88, "high_risk": 0.12}}
# ]
```

### Python — raw REST with google-auth

```python
import json
import google.auth
import google.auth.transport.requests
import requests

PROJECT_ID  = "my-gcp-project-id"
REGION      = "us-central1"
ENDPOINT_ID = "1234567890123456789"   # numeric ID from gcloud output

credentials, _ = google.auth.default(
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
credentials.refresh(google.auth.transport.requests.Request())

url = (
    f"https://{REGION}-aiplatform.googleapis.com/v1/"
    f"projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}:predict"
)

payload = {
    "instances": [
        {
            "age": 58, "prior_er_visits": 4, "household_size": 5,
            "pct_income_on_housing": 62,
            "income_level": "below_poverty", "insurance_status": "uninsured",
            "food_security": "very_low", "education_level": "less_than_hs",
            "health_literacy": "low", "housing_stability": "unstable",
            "neighborhood_safety": "unsafe", "rural_urban": "rural",
            "social_support": "none", "language_barrier": "yes",
            "has_primary_care": "no"
        }
    ]
}

response = requests.post(
    url,
    headers = {
        "Authorization": f"Bearer {credentials.token}",
        "Content-Type":  "application/json",
    },
    json = payload,
)
response.raise_for_status()
print(response.json())
```

### gcloud CLI

```bash
cat > /tmp/sdoh_request.json << 'EOF'
{
  "instances": [
    {
      "age": 58, "prior_er_visits": 4, "household_size": 5,
      "pct_income_on_housing": 62,
      "income_level": "below_poverty", "insurance_status": "uninsured",
      "food_security": "very_low", "education_level": "less_than_hs",
      "health_literacy": "low", "housing_stability": "unstable",
      "neighborhood_safety": "unsafe", "rural_urban": "rural",
      "social_support": "none", "language_barrier": "yes",
      "has_primary_care": "no"
    }
  ]
}
EOF

gcloud ai endpoints predict ${ENDPOINT_ID} \
    --region=us-central1 \
    --json-request=/tmp/sdoh_request.json
```

---

## Step 10 — (Optional) Batch Prediction

Prepare a newline-delimited JSON file — one instance dict per line, no envelope:

```bash
python - << 'EOF'
import pandas as pd, json

FEATURES = [
    "age", "prior_er_visits", "household_size", "pct_income_on_housing",
    "income_level", "insurance_status", "food_security", "education_level",
    "health_literacy", "housing_stability", "neighborhood_safety",
    "rural_urban", "social_support", "language_barrier", "has_primary_care",
]

df = pd.read_csv("./data/sdoh_examples.csv")
with open("/tmp/sdoh_batch_input.jsonl", "w") as f:
    for row in df[FEATURES].to_dict(orient="records"):
        f.write(json.dumps(row) + "\n")
EOF

# Upload to GCS
gcloud storage cp /tmp/sdoh_batch_input.jsonl \
    gs://my-sdoh-model-bucket/sdoh-classifier/batch-input/sdoh_batch_input.jsonl
```

Submit the batch job:

```python
batch_job = model.batch_predict(
    job_display_name       = "sdoh-batch-predict",
    gcs_source             = "gs://my-sdoh-model-bucket/sdoh-classifier/batch-input/sdoh_batch_input.jsonl",
    gcs_destination_prefix = "gs://my-sdoh-model-bucket/sdoh-classifier/batch-output",
    instances_format       = "jsonl",
    predictions_format     = "jsonl",
    machine_type           = "n1-standard-4",
    starting_replica_count = 1,
    max_replica_count      = 5,
    sync                   = True,
)

print("Batch output at:", batch_job.output_info.gcs_output_directory)
```

---

## Step 11 — Tear Down the Endpoint

```python
endpoint.undeploy_all()
endpoint.delete()
```

```bash
gcloud ai endpoints undeploy-model ${ENDPOINT_ID} \
    --region=us-central1 \
    --deployed-model-id=DEPLOYED_MODEL_ID

gcloud ai endpoints delete ${ENDPOINT_ID} \
    --region=us-central1
```

---

## Quick-Reference Checklist

| Step | Action | Done? |
|------|--------|-------|
| 1 | Enable required GCP APIs | ☐ |
| 2 | Generate labeled data (`generate_sdoh_examples.py`) | ☐ |
| 3 | Run `train_dl_sdoh.py --platform vertex` | ☐ |
| 4 | Upload model artifacts to GCS (all 4 files including `model_config.json`) | ☐ |
| 5 | Create Artifact Registry repository | ☐ |
| 6 | Build and push PyTorch prediction container | ☐ |
| 7 | Register model in Vertex AI Model Registry | ☐ |
| 8 | Create endpoint and deploy model | ☐ |
| 9 | Send a test request and validate output | ☐ |
| 10 | Undeploy and delete endpoint when finished | ☐ |

---

## Key PyTorch-Specific Notes

| Concern | Detail |
|---------|--------|
| Model artifact | `model.pt` contains **weights only** (state dict), not the class definition |
| Architecture recovery | `predictor.py` reads `model_config.json` to rebuild `SDoHMLP`, then calls `load_state_dict()` |
| `model_config.json` in GCS | This file **must** be uploaded to GCS alongside `model.pt` — it is not optional |
| Eval mode | `load_artifacts()` calls `model.eval()` after loading; omitting this leaves BatchNorm and Dropout in training mode |
| No-grad context | `run_inference()` wraps the forward pass in `torch.no_grad()` to suppress gradient computation |
| CPU loading | `torch.load(..., map_location="cpu")` ensures the state dict loads on CPU containers even when the model was trained on GPU |
| Base image | Must use a PyTorch Vertex prediction image — the default sklearn CPR base image does not include `torch` |
| Sparse matrix | The sklearn preprocessor returns a sparse matrix; `run_inference()` calls `.toarray()` before converting to a tensor |

---

## Three-Platform Comparison (SDoH Classifier)

| Concern | SageMaker | Azure ML | Vertex AI |
|---------|-----------|----------|-----------|
| Entry-point style | Four functions | `init() / run()` | `Predictor` class |
| Request envelope key | `instances` | `data` | `instances` |
| Model archive | `model.tar.gz` required | Plain folder | Plain GCS folder |
| Custom container required? | No (PyTorch managed image) | No (curated env + `torch` dep) | Yes (push image to Artifact Registry) |
| PyTorch container image | `pytorch-inference:2.2.0-cpu-py310` | Base image + conda `torch>=2.2` | `pytorch-cpu.2-2` CPR base |
| Auth mechanism | IAM / SigV4 | Bearer key | OAuth2 Bearer (gcloud token) |
| `model_config.json` placement | Inside `model.tar.gz` | In registered model folder | In GCS artifact path |
| GPU training support | `ml.p3.2xlarge` | `Standard_NC6` | `n1-standard-8` + T4 |
| Batch scoring | Batch Transform | Batch Endpoints | Batch Prediction jobs |
