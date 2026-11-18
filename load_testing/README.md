# Locust load testing

This folder contains Locust tests for the FastAPI voice-command service.

## 1. Prepare WAV samples

Put a few representative WAV files in the project's `data/inference_samples/`
folder. These should be short, valid WAV recordings, ideally covering different
commands. Do not commit private recordings or the original dataset.

Alternatively, set `LOCUST_AUDIO_FILES` to a comma-separated list of WAV paths.
Paths are resolved from the directory where Locust is started.

## 2. Start the interactive test

From the repository root in Git Bash, with the virtual environment active:

```bash
locust -f load_testing/locustfile.py --host https://voice-command-api.onrender.com
```

Open http://localhost:8089. For the first test against the Render free instance,
use only 1 user and a spawn rate of 1 user/second. Run for 2-3 minutes and watch
the Render service logs and Locust failures.

The test mixes:
- `GET /health`
- `POST /predict` with one WAV file
- `POST /predict/batch` with up to three WAV files

The health task runs even if no WAV samples are found. Prediction tasks are
skipped until WAV files are configured.

## 3. Save a headless report

Example after the interactive smoke test:

```bash
locust -f load_testing/locustfile.py \
  --host https://voice-command-api.onrender.com \
  --headless -u 1 -r 1 -t 2m \
  --csv load_testing/results/render_smoke
```

This writes request statistics, failures, and percentile data as CSV files.
Keep generated result files out of Git unless the assignment requires them.

## 4. Controlled load progression

Do not jump straight to high concurrency on the free Render instance. Increase
gradually (for example 1, then 2, then 5 users), and stop if errors, latency, or
service restarts rise sharply. Record the user count, duration, date, deployment
release, and Render instance configuration with each run.

Locust user count is not the same thing as server container count. These local
tests measure concurrent clients against the deployed service. The assignment's
separate Docker/container-count experiment must be run in a Docker-capable cloud
environment; changing Locust's user count does not simulate additional API
containers.

Useful settings:

- `LOCUST_AUDIO_DIR`: directory containing WAV samples (default:
  `data/inference_samples`).
- `LOCUST_AUDIO_FILES`: optional comma-separated explicit WAV file paths. When
  set, it takes precedence over `LOCUST_AUDIO_DIR`.
- `LOCUST_BATCH_SIZE`: maximum WAV files per batch request, from 1 to 10
  (default: 3).
