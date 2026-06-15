# Porting the Benefits Eligibility Classifier to Google Vertex AI

This guide walks from your local codebase to a live Vertex AI endpoint —
training, packaging, uploading, deploying, and calling the prediction API.
Your existing `train_gb_benefits.py` already contains the Vertex packager, so
the artifacts are generated for you automatically.

---

## Prerequisites

**Google Cloud resources**
- An active GCP project with billing enabled.
- The following APIs enabled on your project:

  ```bash
  gcloud services enable \
      aiplatform.googleapis.com \
      storage.googleapis.com \
      artifactregistry.googleapis.com \
      cloudbuild.googleapis.com
  ```

- A **GCS bucket** in the same region you will deploy to. Example:
  `gs://my-benefits-model-bucket`

  ```bash
  gcloud storage buckets create gs://my-benefits-model-bucket \
      --location=us-central1
  ```

- A **Service Account** with the `Vertex AI User` and `Storage Object Admin`
  roles, or your own user identity with equivalent permissions.

**Local tooling**

```bash
pip install google-cloud-aiplatform google-cloud-storage
gcloud auth application-default login
gcloud config set project my-gcp-project-id
gcloud config set ai/region us-central1
```

Confirm access:

```bash
gcloud ai models list --region=us-central1
```

---

## Step 1 — Generate Labeled Training Data (optional)

If you do not already have a labeled CSV, use the bundled generator.

```bash
python generate_benefits_examples.py \
    --n 2000 \
    --seed 42 \
    --out-dir ./data
```

This writes `./data/benefits_examples.csv` and `./data/benefits_examples.json`.

---

## Step 2 — Train the Model and Build the Vertex Package Locally

Run the training script targeting the Vertex platform. Pass `--data-file` if you
have your own labeled data; omit it to use the built-in synthetic generator.

```bash
# With your own labeled data
python train_gb_benefits.py \
    --platform vertex \
    --data-file ./data/benefits_examples.csv \
    --output-dir ./dist

# With synthetic data (2 000 samples)
python train_gb_benefits.py \
    --platform vertex \
    --samples 2000 \
    --output-dir ./dist
```

After the script completes, `./dist/vertex/` will contain:

```
dist/vertex/
├── model/
│   ├── model.pkl
│   ├── preprocessor.joblib
│   └── label_map.json
├── code/
│   ├── predictor.py          ← Predictor class with load/preprocess/predict/postprocess
│   └── requirements.txt
├── metadata/
│   └── model_metadata.json
└── test/
    ├── sample_request.json   ← {"instances": [...]}
    └── expected_response.json
```

---

## Step 3 — Upload Model Artifacts to GCS

Vertex AI loads model artifacts directly from a GCS path. Upload the contents of
`dist/vertex/model/` there.

```bash
BUCKET="my-benefits-model-bucket"
GCS_PREFIX="benefits-classifier/v1/model"

gcloud storage cp ./dist/vertex/model/* \
    gs://${BUCKET}/${GCS_PREFIX}/

# Verify
gcloud storage ls gs://${BUCKET}/${GCS_PREFIX}/
```

The GCS artifact URI you will use in subsequent steps is:

```
gs://my-benefits-model-bucket/benefits-classifier/v1/model
```

---

## Step 4 — Build and Push a Custom Prediction Container

Vertex AI's Custom Prediction Routine (CPR) requires a Docker image that includes
your `predictor.py` and its dependencies. Build and push that image to Artifact
Registry.

### 4a — Create an Artifact Registry repository

```bash
gcloud artifacts repositories create benefits-models \
    --repository-format=docker \
    --location=us-central1 \
    --description="Benefits eligibility model images"

gcloud auth configure-docker us-central1-docker.pkg.dev
```

### 4b — Write a Dockerfile

Create `Dockerfile` at the project root:

```dockerfile
FROM us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest

# Copy inference code
WORKDIR /opt/app
COPY dist/vertex/code/predictor.py ./predictor.py
COPY dist/vertex/code/requirements.txt ./requirements.txt

# Install additional dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Vertex CPR convention: predictor module must be importable as 'predictor'
ENV HANDLER_MODULE=predictor
ENV HANDLER_CLASS=Predictor
```

> **Note:** The base image `sklearn-cpu.1-3` already bundles scikit-learn 1.3,
> numpy, pandas, and joblib. If your `requirements.txt` lists versions already
> satisfied by the base image, pip will skip them safely.

### 4c — Build and push the image

```bash
PROJECT_ID="my-gcp-project-id"
REGION="us-central1"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/benefits-models/benefits-predictor:v1"

docker build -t ${IMAGE_URI} .
docker push ${IMAGE_URI}
```

If you prefer not to run Docker locally, use Cloud Build:

```bash
gcloud builds submit \
    --tag ${IMAGE_URI} \
    --region=${REGION} \
    .
```

---

## Step 5 — (Optional) Run a Training Job on Vertex AI

If you want Vertex AI to manage training rather than doing it locally:

### 5a — Containerize the training script

Add a training stage to the Dockerfile (or use a separate `Dockerfile.train`):

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train_gb_benefits.py .
COPY generate_benefits_examples.py .

ENTRYPOINT ["python", "train_gb_benefits.py"]
```

Build and push:

```bash
TRAIN_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/benefits-models/benefits-trainer:v1"
docker build -f Dockerfile.train -t ${TRAIN_IMAGE} .
docker push ${TRAIN_IMAGE}
```

### 5b — Submit a Custom Training Job

```python
from google.cloud import aiplatform

aiplatform.init(project="my-gcp-project-id", location="us-central1")

BUCKET      = "my-benefits-model-bucket"
TRAIN_IMAGE = "us-central1-docker.pkg.dev/my-gcp-project-id/benefits-models/benefits-trainer:v1"

job = aiplatform.CustomContainerTrainingJob(
    display_name     = "benefits-training-job",
    container_uri    = TRAIN_IMAGE,
    model_serving_container_image_uri = IMAGE_URI,  # from Step 4c
)

model = job.run(
    args = [
        "--platform", "vertex",
        "--output-dir", "/tmp/dist",
        "--samples", "2000",
    ],
    replica_count  = 1,
    machine_type   = "n1-standard-4",
    base_output_dir = f"gs://{BUCKET}/benefits-classifier/training-output",
)
```

Vertex AI automatically uploads artifacts from `base_output_dir` to GCS when the
job completes.

---

## Step 6 — Register the Model in Vertex AI Model Registry

```python
from google.cloud import aiplatform

aiplatform.init(project="my-gcp-project-id", location="us-central1")

BUCKET     = "my-benefits-model-bucket"
PROJECT_ID = "my-gcp-project-id"
REGION     = "us-central1"
IMAGE_URI  = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/benefits-models/benefits-predictor:v1"

model = aiplatform.Model.upload(
    display_name            = "benefits-eligibility-classifier",
    artifact_uri            = f"gs://{BUCKET}/benefits-classifier/v1/model",
    serving_container_image_uri = IMAGE_URI,
    serving_container_predict_route  = "/predict",
    serving_container_health_route   = "/health",
    serving_container_environment_variables = {
        "HANDLER_MODULE": "predictor",
        "HANDLER_CLASS":  "Predictor",
    },
    description = "GradientBoosting benefits eligibility classifier v1",
    labels      = {"team": "hr-analytics", "stage": "production"},
)

print("Registered model resource name:", model.resource_name)
```

### Via gcloud CLI

```bash
gcloud ai models upload \
    --region=us-central1 \
    --display-name="benefits-eligibility-classifier" \
    --artifact-uri="gs://${BUCKET}/benefits-classifier/v1/model" \
    --container-image-uri="${IMAGE_URI}" \
    --container-health-route="/health" \
    --container-predict-route="/predict"
```

---

## Step 7 — Deploy an Endpoint

### Create the endpoint resource

```python
endpoint = aiplatform.Endpoint.create(
    display_name = "benefits-eligibility-endpoint",
    labels       = {"team": "hr-analytics"},
)

print("Endpoint resource name:", endpoint.resource_name)
```

```bash
gcloud ai endpoints create \
    --region=us-central1 \
    --display-name="benefits-eligibility-endpoint"
```

### Deploy the model to the endpoint

```python
deployed_model = endpoint.deploy(
    model          = model,
    deployed_model_display_name = "benefits-classifier-v1",
    machine_type   = "n1-standard-2",
    min_replica_count = 1,
    max_replica_count = 3,          # auto-scales up to 3 replicas
    traffic_split  = {"0": 100},    # 100 % to this deployment
)

print("Deployment complete. Endpoint:", endpoint.resource_name)
```

```bash
MODEL_ID=$(gcloud ai models list \
    --region=us-central1 \
    --filter="displayName=benefits-eligibility-classifier" \
    --format="value(name)" | head -1)

ENDPOINT_ID=$(gcloud ai endpoints list \
    --region=us-central1 \
    --filter="displayName=benefits-eligibility-endpoint" \
    --format="value(name)" | head -1)

gcloud ai endpoints deploy-model ${ENDPOINT_ID} \
    --region=us-central1 \
    --model=${MODEL_ID} \
    --display-name="benefits-classifier-v1" \
    --machine-type=n1-standard-2 \
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

## Step 8 — Run Prediction Against the Live Endpoint

The `predictor.py` generated by your packager expects `{"instances": [...]}` —
matching the `sample_request.json` in `dist/vertex/test/`.

### Python — Vertex AI SDK

```python
from google.cloud import aiplatform

aiplatform.init(project="my-gcp-project-id", location="us-central1")

endpoint = aiplatform.Endpoint(
    endpoint_name = "projects/my-gcp-project-id/locations/us-central1/endpoints/ENDPOINT_ID"
)

instances = [
    {
        "age": 34, "employment_type": "full_time", "years_of_service": 5,
        "weekly_hours": 40, "department": "engineering",
        "salary_band": "band_3", "has_dependents": 1
    },
    {
        "age": 27, "employment_type": "part_time", "years_of_service": 0,
        "weekly_hours": 20, "department": "sales",
        "salary_band": "band_1", "has_dependents": 0
    }
]

response = endpoint.predict(instances=instances)
print(response.predictions)
# Expected:
# [
#   {"label": "eligible",     "probability": 0.92, "all_proba": {"not_eligible": 0.08, "eligible": 0.92}},
#   {"label": "not_eligible", "probability": 0.87, "all_proba": {"not_eligible": 0.87, "eligible": 0.13}}
# ]
```

### Python — raw REST with google-auth

```python
import json
import google.auth
import google.auth.transport.requests
import requests

PROJECT_ID   = "my-gcp-project-id"
REGION       = "us-central1"
ENDPOINT_ID  = "1234567890123456789"    # ← numeric ID from gcloud output

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
            "age": 34, "employment_type": "full_time", "years_of_service": 5,
            "weekly_hours": 40, "department": "engineering",
            "salary_band": "band_3", "has_dependents": 1
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
cat > /tmp/request.json << 'EOF'
{
  "instances": [
    {
      "age": 34, "employment_type": "full_time", "years_of_service": 5,
      "weekly_hours": 40, "department": "engineering",
      "salary_band": "band_3", "has_dependents": 1
    }
  ]
}
EOF

gcloud ai endpoints predict ${ENDPOINT_ID} \
    --region=us-central1 \
    --json-request=/tmp/request.json
```

---

## Step 9 — (Optional) Batch Prediction

For scoring large files offline without a persistent endpoint:

```python
from google.cloud import aiplatform

aiplatform.init(project="my-gcp-project-id", location="us-central1")

BUCKET = "my-benefits-model-bucket"

# Upload scoring data
# File must be newline-delimited JSON — one instance dict per line (no envelope)
# e.g.  {"age":34,"employment_type":"full_time",...}
# Convert your CSV first:
#   python -c "
#   import pandas as pd, json
#   df = pd.read_csv('./data/benefits_examples.csv')
#   feat = ['age','employment_type','years_of_service','weekly_hours','department','salary_band','has_dependents']
#   df[feat].apply(lambda r: print(r.to_json()), axis=1)"  > /tmp/batch_input.jsonl

batch_job = model.batch_predict(
    job_display_name         = "benefits-batch-predict",
    gcs_source               = f"gs://{BUCKET}/benefits-classifier/batch-input/batch_input.jsonl",
    gcs_destination_prefix   = f"gs://{BUCKET}/benefits-classifier/batch-output",
    instances_format         = "jsonl",
    predictions_format       = "jsonl",
    machine_type             = "n1-standard-4",
    starting_replica_count   = 1,
    max_replica_count        = 5,
    sync                     = True,    # set False to poll manually
)

print("Batch output at:", batch_job.output_info.gcs_output_directory)
```

---

## Step 10 — Tear Down the Endpoint

Vertex AI endpoints incur charges while they have deployed models. Undeploy and
delete when done.

```python
# Undeploy all models then delete the endpoint
endpoint.undeploy_all()
endpoint.delete()
```

```bash
# Undeploy the model first (use the deployed model ID from the deploy output)
gcloud ai endpoints undeploy-model ${ENDPOINT_ID} \
    --region=us-central1 \
    --deployed-model-id=DEPLOYED_MODEL_ID

# Then delete the endpoint
gcloud ai endpoints delete ${ENDPOINT_ID} \
    --region=us-central1
```

---

## Quick-Reference Checklist

| Step | Action | Done? |
|------|--------|-------|
| 1 | Enable required GCP APIs | ☐ |
| 2 | Generate labeled data (`generate_benefits_examples.py`) | ☐ |
| 3 | Run `train_gb_benefits.py --platform vertex` | ☐ |
| 4 | Upload model artifacts to GCS | ☐ |
| 5 | Build and push custom prediction container to Artifact Registry | ☐ |
| 6 | Upload model to Vertex AI Model Registry | ☐ |
| 7 | Create endpoint and deploy model | ☐ |
| 8 | Send a test request and validate output | ☐ |
| 9 | Undeploy and delete endpoint when finished | ☐ |

---

## Key Environment Variables Inside the Container

| Variable | Set by | Purpose |
|----------|--------|---------|
| `AIP_STORAGE_URI` | Vertex AI | GCS path where model artifacts are staged locally |
| `AIP_PREDICT_ROUTE` | Vertex AI | HTTP path the container must serve predictions on |
| `AIP_HEALTH_ROUTE` | Vertex AI | HTTP path for liveness/readiness probes |
| `HANDLER_MODULE` | Your Dockerfile | Python module name containing the Predictor class |
| `HANDLER_CLASS` | Your Dockerfile | Class name inside that module |

Your `predictor.py` uses `load(artifacts_uri)` where Vertex AI passes
`AIP_STORAGE_URI` automatically — the model files are copied from GCS to that
local path before `load()` is called.

---

## Response Format Differences from SageMaker and Azure

The `predictor.py` postprocess method returns richer per-instance objects compared
to the flat lists returned by the SageMaker and Azure inference scripts.

```json
{
  "predictions": [
    {
      "label":      "eligible",
      "probability": 0.92,
      "all_proba":  { "not_eligible": 0.08, "eligible": 0.92 }
    },
    {
      "label":      "not_eligible",
      "probability": 0.87,
      "all_proba":  { "not_eligible": 0.87, "eligible": 0.13 }
    }
  ]
}
```

If you need a uniform response shape across all three platforms, edit
`postprocess()` in `predictor.py` to return `predictions` and `probabilities` as
flat lists, matching the format used by `inference.py` (SageMaker) and `score.py`
(Azure).

---

## Three-Platform Comparison

| Concern | SageMaker | Azure ML | Vertex AI |
|---------|-----------|----------|-----------|
| Entry-point style | Four functions | `init() / run()` | `Predictor` class |
| Request envelope key | `instances` | `data` | `instances` |
| Model packaging | `model.tar.gz` (required) | Plain folder | Plain GCS folder |
| Custom container required? | No (built-in SKLearn image) | No (curated env) | Yes (CPR needs a pushed image) |
| Auth mechanism | IAM / SigV4 | Bearer key | OAuth2 Bearer (gcloud token) |
| Auto-scaling | Endpoint config | Deployment instance count | `min/max_replica_count` |
| Batch scoring service | Batch Transform | Batch Endpoints | Batch Prediction jobs |
