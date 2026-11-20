# Audio-Based Voice Command Recognition System

An end-to-end audio classification system that recognises short spoken commands and supports unknown-word and background-noise detection. It is designed as the voice-recognition component of a voice-controlled educational application.

## Demo and deployed application

- **Video demonstration (YouTube):** [Add the YouTube demo link here]
- **Live Streamlit dashboard:** [Add the deployed dashboard URL here]
- **Live FastAPI service:** https://voice-command-api.onrender.com
- **Interactive API documentation (Swagger):** https://voice-command-api.onrender.com/docs
- **API health check:** https://voice-command-api.onrender.com/health

The video demonstration should cover the dashboard, single-audio prediction, batch prediction, model status, the remote retraining workflow, and the load-testing results. Replace the two placeholders above with the published video and dashboard URLs before submission.

## Project description

The project implements a complete machine-learning workflow for recognising short spoken voice commands. A user can submit a WAV recording and receive a predicted command and confidence score. The application also supports batch predictions, audio-feature visualisations, and controlled model updates using labelled recordings contributed by registered worker computers.

The system uses the Google Speech Commands v0.02 dataset. The dataset contains 105,829 original one-second audio recordings across 35 spoken-word categories and background-noise recordings. It is distributed under the Creative Commons Attribution 4.0 licence.

- Dataset card: https://huggingface.co/datasets/google/speech_commands
- Original dataset archive: https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
- Dataset paper: https://arxiv.org/abs/1804.03209

The project groups the recordings into 12 target classes: ten core commands (`yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`), `unknown`, and `silence`. Auxiliary spoken words are grouped into `unknown`; one-second clips sampled from background-noise recordings form the `silence` class.

Three model approaches are compared: a Linear Support Vector Machine, a Random Forest, and a convolutional neural network (CNN) trained on log-Mel spectrograms. Model selection is based on validation performance, and the selected model is evaluated on a separate held-out test split.

### Main features

- Single WAV-file command prediction.
- Batch WAV prediction with downloadable CSV results.
- Top class scores for each prediction.
- Waveform, log-Mel spectrogram, and MFCC visualisations.
- API health, active-model status, and request metrics.
- Dashboard-controlled submission of labelled audio for retraining.
- Worker registration, job assignment, heartbeats, and candidate checkpoint submission.
- Validation-based candidate promotion and manually controlled model releases.

A prediction confidence score is the model's output score. It should not be interpreted as a calibrated probability.

## Project results

The validation comparison selected the small audio CNN. Its recorded validation metrics were:

| Metric | Small Audio CNN |
|---|---:|
| Accuracy | 0.8368 |
| Balanced accuracy | 0.9107 |
| Macro precision | 0.7348 |
| Macro recall | 0.9107 |
| Macro F1-score | 0.8045 |

The project also includes held-out test evaluation in `notebooks/05_final_test_evaluation.ipynb`. The test split is kept separate from model selection and retraining decisions.

### Flood-request simulation (Locust)

Locust was used to simulate concurrent requests against the deployed Render API. Each run lasted two minutes. The user count was varied while the API deployment remained the same.

| Concurrent users | Requests | Failed requests | Average response time | Median | 95th percentile | Throughput |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 39 | 0 | 893 ms | 700 ms | 1,900 ms | 0.335 req/s |
| 2 | 81 | 0 | 948 ms | 670 ms | 2,300 ms | 0.678 req/s |
| 5 | 205 | 0 | 836 ms | 730 ms | 1,900 ms | 1.708 req/s |

A separate Docker experiment used five concurrent Locust users and varied the number of API containers behind Nginx.

| API containers | Requests | Failed requests | Average response time | Median | 95th percentile | Throughput |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 176 | 0 | 1,368.7 ms | 700 ms | 4,600 ms | 1.476 req/s |
| 2 | 179 | 0 | 1,313.9 ms | 1,200 ms | 4,000 ms | 1.495 req/s |
| 5 | 176 | 0 | 1,320.9 ms | 1,200 ms | 4,000 ms | 1.475 req/s |

All recorded runs completed without failed requests. In the Docker experiment, two containers produced the best measured throughput and average response time, but five containers did not improve throughput further. Batch prediction was the slowest request type.

These are small, controlled experiments, not a measurement of production capacity. Each run was short, concurrency was limited, and the Docker tests ran in a shared Killercoda environment. The Render and Docker experiments also tested different conditions and should not be compared as if they were one continuous benchmark.

Detailed reports and raw Locust exports are available in [load_testing/results](load_testing/results/):
- [Render concurrency comparison](load_testing/results/render_load_test_comparison.md)
- [Docker container comparison](load_testing/results/docker_container_comparison.md)

## Repository structure

```text
notebooks/       Dataset exploration, preprocessing, training, evaluation, prediction
src/             Preprocessing, classical/deep models, prediction, retraining
api/             FastAPI inference and retraining endpoints
app/             Streamlit dashboard
data/            Local source and processed data (not committed)
models/          Locally generated model artifacts (not committed)
tests/           Automated tests
load_testing/    Locust scripts, Nginx configs, reports, and result exports
requirements.txt
Dockerfile
```

## Setup and installation

### 1. Clone the repository

The following commands are for Windows using Git Bash:

```bash
git clone https://github.com/Abejirin-King/machine_learning_cycle.git
cd machine_learning_cycle
```

### 2. Create and activate a virtual environment

Python 3.11 is recommended.

```bash
py -3.11 -m venv .venv
source .venv/Scripts/activate
```

If the `py` launcher is unavailable, use `python -m venv .venv`.

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare the dataset and model artifacts

Download and extract the Google Speech Commands v0.02 archive. Follow the dataset and preprocessing instructions in notebooks 01 and 02. Run notebooks 01-05 in order to explore and prepare the data, train the traditional ML and CNN models, and evaluate the models. Run notebook 06 to test single-file and batch prediction.

The deployed API uses the approved CNN checkpoint from the configured GitHub Release. On startup, the API downloads the checkpoint for the configured model release if it is not already available. The local notebook workflow generates experiment artifacts that are not committed to the repository.

For local retraining, the required files include the original audio dataset, `notebooks/data/processed/audio_index_with_silence.csv`, the current CNN checkpoint, and `models/final_test_metrics.json`. The MFCC feature archive is not required for CNN retraining.

### 5. Start the FastAPI service

From the repository root, activate the virtual environment and run:

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

The local API documentation is available at http://127.0.0.1:8000/docs and the health endpoint is http://127.0.0.1:8000/health.

### 6. Start the Streamlit dashboard

Open a second Git Bash terminal, navigate to the repository, and activate the same virtual environment:

```bash
cd machine_learning_cycle
source .venv/Scripts/activate
streamlit run app/dashboard.py
```

The dashboard normally opens at http://localhost:8501. By default, it connects to the API at http://127.0.0.1:8000. If the API is hosted elsewhere, update the FastAPI base URL in the dashboard sidebar or set the `API_BASE_URL` environment variable before launching Streamlit.

### 7. Run the automated tests

```bash
python -m pytest -v
```

### 8. Optional: run the load tests

Install Locust if it is not already installed:

```bash
pip install locust
```

From the repository root, run:

```bash
locust -f load_testing/locustfile.py
```

Open the Locust web interface shown in the terminal, set the host to the API URL being tested, and configure the desired number of users and spawn rate. Set `LOCUST_AUDIO_FILES` or `LOCUST_AUDIO_DIR` to point to WAV samples if prediction and batch-prediction tasks are to be included. The detailed Docker/Nginx container comparison procedure is documented in [load_testing/README.md](load_testing/README.md).

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | API and model readiness |
| GET | `/metrics` | Basic request metrics |
| POST | `/predict` | Predict a command from one WAV file |
| POST | `/predict/batch` | Predict commands from multiple WAV files |
| POST | `/retrain` | Hosted retraining endpoint; direct hosted retraining is disabled |
 
Remote retraining is coordinated through the dashboard and Supabase-backed job queue. Registered worker computers perform the local fine-tuning work. Follow [api/REMOTE_TRAINING_SETUP.md](api/REMOTE_TRAINING_SETUP.md) to configure Supabase, secrets, storage, and worker computers. Keep API tokens, worker credentials, and Supabase service keys private; do not commit them to the repository.

Candidate promotion is based on validation macro-F1. The held-out test split is not used for retraining decisions. A promoted candidate is reviewed and manually published as a GitHub Release before the deployed API is updated to that release.

## Deployment and architecture notes

The live API is hosted on Render. The Streamlit dashboard is deployed separately. The API is also packaged in a CPU-only Docker image and can run behind Nginx with one, two, or five API replicas. The local development workflow does not require Docker Desktop.

## Attribution

Speech Commands was created by Pete Warden and contributors. Please retain the dataset attribution and licence information when redistributing derived materials.
