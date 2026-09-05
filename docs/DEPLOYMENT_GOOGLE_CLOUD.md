# Google Cloud Run Deployment Guide

Deploy AbuseRing Sentinel as a single Docker container on Google Cloud Run. The container serves
both the React frontend and the FastAPI backend from the same URL.

---

## Architecture

```
Browser
  ↓
Google Cloud Run (single container)
  ├── React frontend (static assets served by FastAPI)
  └── FastAPI backend (API + ML models + graph data)
```

- **Frontend**: Pre-built React app served as static files.
- **Backend**: FastAPI with XGBoost model, NetworkX graph, SHAP explainer, all loaded at startup.
- **API routes** (`/health`, `/users/{id}/risk`, etc.) return JSON.
- **All other GET routes** return the React SPA `index.html` for client-side routing.

---

## Prerequisites

| Tool | Install |
|------|---------|
| Docker Desktop | https://www.docker.com/products/docker-desktop |
| Google Cloud CLI (`gcloud`) | https://cloud.google.com/sdk/docs/install |
| A Google Cloud account with billing enabled | https://console.cloud.google.com |

---

## 1. Enable Required Google Cloud APIs

```bash
gcloud auth login
gcloud config set project PROJECT_ID

gcloud services enable \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  artifactregistry.googleapis.com
```

---

## 2. Create an Artifact Registry Repository

```bash
gcloud artifacts repositories create REPOSITORY_NAME \
  --repository-format=docker \
  --location=REGION \
  --description="AbuseRing Sentinel Docker images"
```

Authenticate Docker with Artifact Registry:

```bash
gcloud auth configure-docker REGION-docker.pkg.dev
```

---

## 3. Build the Docker Image Locally

From the project root (the directory containing `Dockerfile`):

```bash
docker build -t abusering-sentinel .
```

> **Note:** The build uses a multi-stage Dockerfile. Stage 1 builds the React frontend with Node 20.
> Stage 2 sets up Python 3.12 with all ML dependencies and copies the compiled frontend into the
> final image. The `data/processed/` directory (CSV files, parquet features) must exist locally
> since these files are gitignored and generated during the ML pipeline.

---

## 4. Test Locally with Docker

```bash
docker run --rm -p 8080:8080 -e PORT=8080 abusering-sentinel
```

Then verify:

| Check | URL |
|-------|-----|
| Frontend homepage | http://localhost:8080/ |
| Health API | http://localhost:8080/health |
| User risk | http://localhost:8080/users/USR_00000085/risk |
| User investigation | http://localhost:8080/users/USR_00000085/investigation |
| Review queue | http://localhost:8080/review-queue |
| Direct SPA route refresh | http://localhost:8080/users/USR_00000085 |

---

## 5. Tag and Push to Artifact Registry

```bash
docker tag abusering-sentinel \
  REGION-docker.pkg.dev/PROJECT_ID/REPOSITORY_NAME/abusering-sentinel:latest

docker push \
  REGION-docker.pkg.dev/PROJECT_ID/REPOSITORY_NAME/abusering-sentinel:latest
```

---

## 6. Deploy to Cloud Run

```bash
gcloud run deploy SERVICE_NAME \
  --image=REGION-docker.pkg.dev/PROJECT_ID/REPOSITORY_NAME/abusering-sentinel:latest \
  --region=REGION \
  --platform=managed \
  --port=8080 \
  --memory=2Gi \
  --cpu=2 \
  --min-instances=0 \
  --max-instances=1 \
  --timeout=300 \
  --allow-unauthenticated
```

### Resource Recommendations

| Setting | Value | Reason |
|---------|-------|--------|
| CPU | 2 | XGBoost + SHAP computation is CPU-intensive |
| Memory | 2 GiB | Model + graph + feature data loaded into memory at startup |
| Min instances | 0 | Cost saving — container starts on first request |
| Max instances | 1 | Keeps in-memory ReviewQueue consistent for the demo |
| Timeout | 300s | Allows time for cold-start model loading |

### Why max-instances = 1?

The ReviewQueue is **in-memory only**. Setting `max-instances=1` ensures all users hit the same
queue state during the demo. With multiple instances, each would have its own independent queue.

> **⚠️ Limitation:** Even with `max-instances=1`, the queue state resets whenever Cloud Run
> restarts the container (e.g., after a period of inactivity, a new deployment, or a crash).
> This is acceptable for a buildathon demo. For production use, the queue would need a
> persistent backing store (e.g., Redis, Cloud SQL).

---

## 7. Get the Public URL

After deployment, `gcloud run deploy` prints the service URL:

```
Service URL: https://SERVICE_NAME-XXXXXXXXXX-REGION.a.run.app
```

Verify the deployment:

```bash
curl https://SERVICE_NAME-XXXXXXXXXX-REGION.a.run.app/health
```

Expected response:

```json
{"status":"ok","model_loaded":true,"graph_loaded":true,"investigator_loaded":true,"risk_engine_loaded":true}
```

---

## 8. Redeploying After Code Changes

```bash
# Rebuild the image
docker build -t abusering-sentinel .

# Tag and push
docker tag abusering-sentinel \
  REGION-docker.pkg.dev/PROJECT_ID/REPOSITORY_NAME/abusering-sentinel:latest
docker push \
  REGION-docker.pkg.dev/PROJECT_ID/REPOSITORY_NAME/abusering-sentinel:latest

# Redeploy (Cloud Run picks up the new image)
gcloud run deploy SERVICE_NAME \
  --image=REGION-docker.pkg.dev/PROJECT_ID/REPOSITORY_NAME/abusering-sentinel:latest \
  --region=REGION
```

---

## 9. Alternative: Build with Cloud Build

Instead of building locally, you can use Google Cloud Build:

```bash
gcloud builds submit \
  --tag=REGION-docker.pkg.dev/PROJECT_ID/REPOSITORY_NAME/abusering-sentinel:latest \
  .
```

> **Note:** Cloud Build runs in the cloud. Since `data/processed/` files are gitignored, you
> must either:
> - Commit the data files temporarily, or
> - Use a custom `cloudbuild.yaml` that generates the data first, or
> - Build locally and push the image (recommended for this project).

---

## 10. View Logs

```bash
gcloud run services logs read SERVICE_NAME --region=REGION --limit=50
```

Or use the Cloud Console: **Cloud Run → SERVICE_NAME → Logs**.

---

## 11. Troubleshooting

### Container fails to start

- **Out of memory**: Increase `--memory` to `4Gi`.
- **Startup timeout**: Increase `--timeout` to `600`.
- **Missing data files**: Ensure `data/processed/` contains the CSV and parquet files before building.

### Cold start is slow

The first request after scaling from zero triggers model loading (XGBoost, SHAP, NetworkX graph).
This can take 15–30 seconds. Set `--min-instances=1` to keep one instance always warm
(increases cost).

### CORS errors

In the Docker container, the frontend and backend share the same origin, so CORS is not needed
for normal operation. If you access the API from a different origin, set the `FRONTEND_URL`
environment variable:

```bash
gcloud run services update SERVICE_NAME \
  --region=REGION \
  --set-env-vars="FRONTEND_URL=https://your-other-frontend.example.com"
```

---

## 12. Stop or Delete the Service

```bash
# Delete the service
gcloud run services delete SERVICE_NAME --region=REGION

# Delete the Artifact Registry repository (and all images)
gcloud artifacts repositories delete REPOSITORY_NAME --location=REGION
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | Set by Cloud Run | `8080` | Port the container listens on |
| `FRONTEND_URL` | No | — | Additional CORS origin (not needed when frontend is in the same container) |
| `ENVIRONMENT` | No | `development` | Set to `production` for production logging |
| `DATA_DIRECTORY` | No | `data` | Path to the data directory |
| `RANDOM_SEED` | No | `42` | Seed for reproducible ML results |

---

## Known Limitations

1. **ReviewQueue is in-memory**: Resets on container restart or scale events. Not durable.
2. **Cold start**: First request after scale-from-zero takes 15–30s for model loading.
3. **Data files not in git**: `data/processed/` CSV/parquet files are gitignored. The Docker image
   must be built locally where these files exist.
4. **Single container**: This is a monolithic deployment suitable for a demo. For production,
   consider separating frontend (CDN/Netlify) and backend (Cloud Run) for independent scaling.
