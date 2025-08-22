## Customer Intelligence Platform API

Production-ready REST API for customer analytics: churn prediction, segmentation, forecasting, sentiment, and lightweight insights. Runs locally with Flask or in production on Google Cloud Run.

 Live API: https://churn-capstone-1010496745012.us-central1.run.app

## What’s inside
- Flask API with health/status, ML predictions, and utilities
- Churn prediction models: standalone RandomForest and unified Logistic Regression
- KMeans segmentation via unified endpoint
- Sales forecasting (Ridge)
- Sentiment analysis with NLTK VADER (no external service)
- Docker image (gunicorn) and Cloud Run deployment
- Optional CI/CD via GitHub Actions

## Project structure
```
customer-intelligence-platform/
├─ app.py                       # Flask app with all endpoints
├─ requirements.txt             # Python dependencies
├─ Dockerfile                   # Production container (gunicorn + Flask)
├─ .dockerignore                # Trim Docker context
├─ models/
│   ├─ rf_churn_model.joblib
│   ├─ rf_churn_threshold.json
│   ├─ best_forecast_model_Ridge_prod.joblib
│   └─ (auto-trained) logreg_model.joblib, kmeans_model.joblib
├─ data/
│   ├─ processed/cleaned_transactions.csv  # used to auto-train unified models
│   └─ ...
├─ dashboard/, docs/, notebooks/, src/
└─ README.md
```

## Run locally (Python)
Prereqs: Python 3.11+

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py   # serves http://127.0.0.1:5001
```

- Health: http://127.0.0.1:5001/health
- Status: http://127.0.0.1:5001/status
- Routes: http://127.0.0.1:5001/__routes__

## Run locally (Docker)
```powershell
docker build -t churn-capstone:local .
docker run --rm -p 8080:8080 churn-capstone:local
```

- Health: http://127.0.0.1:8080/health
- Unified schema: http://127.0.0.1:8080/predict/schema
- Churn (RF) schema: http://127.0.0.1:8080/predict/churn/schema

## Configuration (env vars)
Defaults are sensible; override if needed.

- MODELS_DIR: default ./models
- CHURN_MODEL_PATH: models/rf_churn_model.joblib
- CHURN_THRESH_PATH: models/rf_churn_threshold.json (fallback models/rf_threshold.json)
- FORECAST_MODEL_PATH: models/best_forecast_model_Ridge_prod.joblib
- INSIGHTS_FILE: ./customer_insights_mistral.txt (optional)
- AUDIO_FILE: ./audio_output/insights_from_file.mp3 (optional)
- PORT: container port (Cloud Run injects 8080)

## Endpoints
Base URLs
- Local (python): http://127.0.0.1:5001
- Local (docker): http://127.0.0.1:8080
- Cloud Run: https://churn-capstone-1010496745012.us-central1.run.app

### Health and utilities
- GET `/health` → { status: "ok" }
- GET `/status` → model/asset load status and metadata
- GET `/__routes__` → list all registered routes (debug)

### Unified prediction
- GET `/predict/schema` → expected feature order for unified models
  - Default order: ["age","tenure_months","num_purchases","avg_spent"]

- POST `/predict`
  - Body: { "model_type": "logreg" | "kmeans", "features": [..] }
  - LogisticRegression returns label (and prob if available)
  - KMeans returns cluster label

Examples
```json
{ "model_type": "logreg", "features": [32, 24, 5, 70.5] }
```
```json
{ "model_type": "kmeans", "features": [32, 24, 5, 70.5] }
```

Notes
- If models are missing, the app can auto-train:
  - KMeans(n_clusters=4)
  - LogisticRegression if a numeric "churn" column exists
  - Source: data/processed/cleaned_transactions.csv

### Standalone churn (RandomForest)
- GET `/predict/churn/schema` → n_features and feature_names if available
- POST `/predict/churn`
  - Body: { "features": [ .. ] } in the exact order your RF expects
  - Threshold comes from models/rf_churn_threshold.json (key best_threshold_f2); default 0.50

Examples
```json
{ "features": [45, 0, 1, 60, 20, 300.0] }
```
```json
{ "features": [22, 1, 0, 2, 1, 15.0] }
```

Tip: Gender/segment must match your training encoding (e.g., LabelEncoder indices).

### Forecasting
- POST `/predict/forecast` → Body: { "features": [..] } per your Ridge model

### Sentiment & insights
- POST `/sentiment` → { text: "..." } → VADER polarity scores
- POST `/llm_insights` → returns contents of INSIGHTS_FILE if present
- GET/POST `/tts_insights` → returns audio file if present

## Deploy to Google Cloud Run

### One-off build and deploy (Artifact Registry + Cloud Run)
```powershell
gcloud auth login
gcloud config set project digital-hall-469808-a1
gcloud config set run/region us-central1
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com

gcloud artifacts repositories create churn-capstone `
  --repository-format=docker `
  --location=us-central1 `
  --description "Docker repo for churn-capstone"  # run once

gcloud auth configure-docker us-central1-docker.pkg.dev

gcloud builds submit --tag us-central1-docker.pkg.dev/digital-hall-469808-a1/churn-capstone/churn-capstone:latest .

gcloud run deploy churn-capstone `
  --image us-central1-docker.pkg.dev/digital-hall-469808-a1/churn-capstone/churn-capstone:latest `
  --region us-central1 `
  --platform managed `
  --allow-unauthenticated
```

### GitHub Actions (optional)
Create repo secrets (Settings → Secrets and variables → Actions):
- GCP_PROJECT_ID=digital-hall-469808-a1
- GCP_REGION=us-central1
- SERVICE_NAME=churn-capstone
- AR_REPO=churn-capstone
- GCP_SA_KEY (JSON content; never commit the file)

Push to main to trigger the workflow in `.github/workflows/deploy-cloud-run.yml`.

## Troubleshooting
- 404 Not Found → check path and method; use `GET /__routes__` to list routes
- 405 Method Not Allowed → use POST for `/predict` and `/predict/churn`
- Secret scanning (GH013) → remove secrets from history and rotate keys
- Windows file locks (PBIX, __pycache__) → close apps/OneDrive, clear read-only, retry
- VADER offline → Docker image preloads VADER; local Python needs first-run download
- Port mismatch → Python uses 5001; Docker/Cloud Run use 8080

## License
Proprietary / Educational use
