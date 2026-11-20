# Remote training setup

This feature keeps Streamlit and FastAPI on Render while registered computers perform CNN fine-tuning locally. Supabase stores the job queue and uploaded files because Render's free filesystem is ephemeral.

## 1. Configure Supabase

1. Create a Supabase project.
2. Open **SQL Editor** and run the complete contents of `api/training_jobs_schema.sql`.
3. The SQL creates the private `training-jobs` Storage bucket and the two job tables.
4. Copy the project URL and the **service_role** key from Project Settings → API. Never put the service-role key in Streamlit, a worker, or GitHub.

## 2. Configure the FastAPI Render service

Add these environment variables to the `voice-command-api` service:

- `SUPABASE_URL`: Supabase project URL.
- `SUPABASE_SERVICE_ROLE_KEY`: Supabase service-role key.
- `SUPABASE_TRAINING_BUCKET`: `training-jobs` (optional; this is the default).
- `DASHBOARD_API_TOKEN`: a long random secret used by the dashboard to submit jobs and view status.
- `WORKER_ENROLLMENT_TOKEN`: a separate long random secret used only when registering a new worker.
- `MODEL_RELEASE_TAG`: the currently approved model release, for example `v1.2.0`.

Generate random values locally in Git Bash with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(40))"
```

Generate a different value for each secret. Do not commit them. Redeploy the API after adding the variables.

## 3. Configure the Streamlit Render service

Set `API_BASE_URL` to `https://voice-command-api.onrender.com` on the dashboard service. The dashboard asks for the `DASHBOARD_API_TOKEN` when you open its Retrain tab. Do not set the service-role key on the dashboard.

## 4. Register a training computer

Every worker computer must have:

- This repository cloned.
- The project's Python dependencies installed.
- The original Google Speech Commands WAV dataset.
- `notebooks/data/processed/audio_index_with_silence.csv`.
- The current model metadata file `models/final_test_metrics.json`.

The worker synchronizes the checkpoint from the API's configured GitHub release when it starts.

In the dashboard → Retrain → Register this computer, enter a worker name and the `WORKER_ENROLLMENT_TOKEN`. Copy the returned worker ID and token immediately. On the worker computer, from the repository root in Git Bash:

```bash
source venv/Scripts/activate
export TRAINING_WORKER_ID="paste-worker-id"
export TRAINING_WORKER_TOKEN="paste-worker-token"
python -m src.worker --project-root .
```

Keep the terminal running. A worker polls for jobs and sends heartbeats while training. Other contributors can register their own computers the same way. Each worker must independently have the original dataset and processed index; only the newly labelled recordings are transferred with a job.

## 5. Submit a job

Prepare a CSV with exactly the required columns:

```csv
filename,label
new_yes_01.wav,yes
new_stop_01.wav,stop
```

Upload the CSV and matching WAV files in the dashboard. The current limits are 100 recordings and 40 MB combined per job, with 1–5 epochs. A job is assigned to one available worker. The worker fine-tunes locally and uploads validation results and, when validation macro-F1 improves, a candidate checkpoint.

A `candidate_ready` job is not deployed automatically. Review its validation metrics before manually promoting the candidate into a new GitHub Release and changing `MODEL_RELEASE_TAG` on Render. This preserves the current release as a rollback point.

## Security and operational notes

- The dashboard token, worker enrollment token, worker bearer tokens, and Supabase service-role key are secrets. Never commit them.
- The Storage bucket is private; workers receive short-lived signed download URLs.
- Disable a worker by setting its `enabled` field to false in `training_workers`.
- The initial implementation stores job status and uploads durably in Supabase. Render remains a lightweight coordinator and inference host.
- Do not upload private or sensitive recordings unless you have permission and a lawful basis to process them.
- The current dashboard displays queued/running/completed states and validation results. Detailed per-epoch progress and automatic model-release publishing are not enabled.
