# Voice Command API

The API is implemented with FastAPI in `api/main.py`.

## Start locally

From the repository root, with the project virtual environment active:

```bash
pip install -r requirements.txt
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Open the interactive API documentation at `http://127.0.0.1:8000/docs`.

The service needs the trained model files in `models/` and the processed MFCC archive at `notebooks/data/processed/mfcc_features.npz`. These generated artifacts are intentionally not committed to GitHub; provision them in the runtime environment before starting the service.

## Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Readiness and artifact status |
| GET | `/metrics` | Uptime, request count and active model |
| POST | `/predict` | Predict one WAV recording |
| POST | `/predict/batch` | Predict several WAV recordings |
| POST | `/retrain` | Upload labelled WAV files and trigger retraining |

### Single prediction

Use multipart form data with one field named `file`, containing a WAV file.

### Batch prediction

Use multipart form data with one or more files under the field name `files`.

### Retraining

Upload:
- `metadata`: a CSV with exactly the required columns `filename,label` (additional columns are ignored).
- `audio_files`: the WAV recordings referenced by the CSV.

Example metadata:

```csv
filename,label
new_yes_01.wav,yes
new_stop_01.wav,stop
new_up_01.wav,up
```

Accepted labels are the existing 12 model classes: `down`, `go`, `left`, `no`, `off`, `on`, `right`, `silence`, `stop`, `unknown`, `up`, and `yes`.

The retraining workflow adds the new labelled recordings to the original training features, trains a candidate Linear SVM, and evaluates it on the unchanged validation split. The candidate is promoted only if its validation macro-F1 is higher than the currently selected model's recorded validation macro-F1. The held-out test split is not used for retraining or promotion. A promotion creates `models/linear_svm_retrained.joblib` and updates `models/deployment_model.json`.

The current inference service reads that deployment configuration on each prediction request, so a promoted model is used without restarting the API. Retraining is a synchronous, resource-intensive operation; protect this endpoint with authentication and request limits before exposing it publicly.
