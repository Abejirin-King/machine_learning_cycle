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


## 5. Docker container-count experiment (Killercoda)

The repository now includes a CPU-only API Dockerfile and Nginx configurations
for 1, 2, and 5 API replicas. This experiment uses Docker directly, so Docker
Compose is not required. Run it in a Docker-capable Linux playground such as
Killercoda, not against the Render service.

### Build the API image

From the repository root in the Killercoda terminal:

```bash
git clone https://github.com/Abejirin-King/machine_learning_cycle.git
cd machine_learning_cycle
docker build -t voice-command-api:v1.2.0 .
```

The image downloads the pinned v1.2.0 CNN checkpoint from the public GitHub
Release when each API container starts. No Supabase credentials are needed for
inference or this load test.

### Create the Docker network

```bash
docker network create voice-test
```

### Run the 1-container configuration

```bash
docker run -d --name api1 --network voice-test voice-command-api:v1.2.0
docker run -d --name voice-proxy --network voice-test -p 8080:80 \
  -v "$PWD/load_testing/nginx-1.conf:/etc/nginx/nginx.conf:ro" nginx:alpine
```

Open the Killercoda forwarded URL for port 8080 and check that
`/health` returns JSON with `"status":"ok"`.

For 2 and 5 containers, first remove the proxy and all API containers:

```bash
docker rm -f voice-proxy api1
```

Then start the required number of API containers, using these commands for
two replicas:

```bash
docker run -d --name api1 --network voice-test voice-command-api:v1.2.0
docker run -d --name api2 --network voice-test voice-command-api:v1.2.0
docker run -d --name voice-proxy --network voice-test -p 8080:80 \
  -v "$PWD/load_testing/nginx-2.conf:/etc/nginx/nginx.conf:ro" nginx:alpine
```

For five replicas, start `api1` through `api5` with the same `docker run`
command, changing the container name each time, then start the proxy with
`load_testing/nginx-5.conf`:

```bash
docker run -d --name voice-proxy --network voice-test -p 8080:80 \
  -v "$PWD/load_testing/nginx-5.conf:/etc/nginx/nginx.conf:ro" nginx:alpine
```

Do not run the five-container test if the playground runs out of memory or
restarts containers. Record the actual number of healthy API containers.

### Run Locust against the forwarded URL

On your Windows machine, use the Killercoda public forwarded URL as Locust's
host (include `https://` or `http://` exactly as shown by Killercoda):

```bash
locust -f load_testing/locustfile.py --host https://YOUR-FORWARDED-HOST
```

Keep the WAV sample paths configured on the Windows machine as before. Use
the same Locust user count, spawn rate, test duration, and WAV files for each
container-count run so the comparisons are meaningful. Export each run's
`*_stats.csv` and `*_stats_history.csv` files separately.

The Docker experiment measures API replicas behind one Nginx load balancer.
It is distinct from the earlier Render experiment, which varied Locust users
but did not vary the number of server containers.

### Clean up

```bash
docker rm -f voice-proxy api1 api2 api3 api4 api5 2>/dev/null || true
docker network rm voice-test
```
