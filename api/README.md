# Voice Command API

The API is implemented with FastAPI in `api/main.py`.

## Start locally

From the repository root, with the project virtual environment active:

```bash
pip install -r requirements.txt
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Open the interactive API documentation at `http://127.0.0.1:8000/docs`.

## Model checkpoint provisioning

At startup, the API downloads `small_audio_cnn.pt` from the public GitHub Release selected by `MODEL_RELEASE_TAG` (defaults to `v1.0.0`). The download is checked against the SHA-256 digest published in the GitHub release metadata before replacing the local checkpoint. The release is the source of truth for the deployed CNN weights.

To pin a different approved release in Render, set the environment variable `MODEL_RELEASE_TAG` to its tag (for example, `v1.1.0`) and redeploy. Do not put a GitHub token in the repository; the current public release download does not require one.

The API also requires `models/final_test_metrics.json` for model-selection metadata. That small evaluation file remains part of the repository.

## Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Readiness and artifact status |
| GET | `/metrics` | Uptime, request count and active model |
| POST | `/predict` | Predict one WAV recording |
| POST | `/predict/batch` | Predict several WAV recordings |
| POST | `/retrain` | Returns HTTP 501; legacy direct API training is disabled |
| POST | `/training/jobs` | Authenticated dashboard job submission |
| GET | `/training/jobs` | List recent jobs |
| GET | `/training/workers` | List registered workers |
| POST | `/training/workers/register` | Register a worker with the enrollment secret |
| POST | `/training/workers/{worker_id}/claim` | Claim the next queued job |
| POST | `/training/workers/{worker_id}/heartbeat` | Worker liveness heartbeat |
| GET | `/training/jobs/{job_id}/candidate` | Create a temporary candidate download URL |

### Single prediction

Use multipart form data with one field named `file`, containing a WAV file.

### Batch prediction

Use multipart form data with one or more files under the field name `files`.

### CNN retraining

CNN training is deliberately not executed by the hosted FastAPI service. Render's free instance is sized for inference, not repeated audio training, and its filesystem is ephemeral. The hosted `POST /retrain` route therefore returns HTTP 501 instead of attempting the old, incorrect Linear SVM workflow.

Run retraining on the computer that contains the original Google Speech Commands dataset and the generated split index. The required index is `notebooks/data/processed/audio_index_with_silence.csv`; it must point to the original audio files used in notebooks 01–02. Do not upload the original dataset or private recordings to the public repository.

Prepare a CSV with columns `filename,label` and put the matching WAV files in one folder. Example:

```csv
filename,label
new_yes_01.wav,yes
new_stop_01.wav,stop
new_up_01.wav,up
```

Accepted labels: `down`, `go`, `left`, `no`, `off`, `on`, `right`, `silence`, `stop`, `unknown`, `up`, and `yes`.

From the repository root, with the project virtual environment active, run:

```bash
python -m src.retraining --metadata data/new_labels.csv --audio-dir data/new_audio --project-root . --epochs 3
```

The script fine-tunes the existing `small_audio_cnn.pt`, using the new recordings only as additional training examples. It evaluates both the current checkpoint and candidate on the same unchanged validation split. A candidate is saved as `models/small_audio_cnn_candidate.pt` only when its validation macro-F1 is strictly higher. It writes the comparison to `models/latest_retraining_result.json`. The held-out test split is never loaded or used.

A passing candidate is not deployed automatically. Review its validation result, then publish `small_audio_cnn_candidate.pt` as `small_audio_cnn.pt` in a new GitHub Release (for example `v1.1.0`). Set Render's `MODEL_RELEASE_TAG` to that new tag and redeploy. The current release remains the rollback point if the candidate does not improve.


## Remote dashboard-controlled retraining

The Streamlit Retrain tab can submit labelled audio to the FastAPI coordinator. The coordinator stores job records and files in a private Supabase database/storage bucket. Registered worker computers poll for jobs and execute the existing validation-gated CNN fine-tuning locally. Render does not perform training.

Complete the setup in [REMOTE_TRAINING_SETUP.md](REMOTE_TRAINING_SETUP.md) before using these routes. Configure `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `DASHBOARD_API_TOKEN`, and `WORKER_ENROLLMENT_TOKEN` on the API host. Apply `training_jobs_schema.sql` in Supabase.

Workers run from a clone that contains the original dataset, audio index and model metadata:

```bash
source venv/Scripts/activate
export TRAINING_WORKER_ID="your-worker-id"
export TRAINING_WORKER_TOKEN="your-worker-token"
python -m src.worker --project-root .
```

A worker synchronizes the checkpoint from the API's configured `MODEL_RELEASE_TAG` before polling. A successful candidate is uploaded to private storage and can be downloaded from the dashboard. Candidate deployment remains a manual approval and GitHub Release operation.
