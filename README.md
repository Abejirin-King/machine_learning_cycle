# Audio-Based Voice Command Recognition System

An end-to-end audio classification project that recognises short spoken commands and supports unknown-word and background-noise detection. The system is designed as a voice-controlled interface for an educational application.

## Project status

**Milestones 1-8: dataset exploration, preprocessing, model training, held-out test evaluation, single/batch prediction, FastAPI service, controlled retraining, and Streamlit dashboard.** The dashboard in `app/dashboard.py` provides API health and model status, single and batch WAV prediction, waveform/Mel/MFCC visualisations, saved test metrics, and a dashboard-controlled remote retraining workflow using registered worker computers.

The repository includes `src/preprocessing.py`, `src/models.py`, `src/deep_learning.py`, `src/prediction.py`, `src/retraining.py`, the FastAPI service in `api/main.py`, and notebooks 01-06. Notebook 03 compares a class-weighted Linear SVM and Random Forest; notebook 04 trains a CNN on log-Mel spectrograms using all 12 classes, including silence. Notebook 05 evaluates all three trained models on the untouched test split and records the validation-selected model.

The project uses the Google Speech Commands v0.02 dataset. The dataset contains 105,829 original one-second audio recordings across 35 spoken-word categories and background-noise recordings. The original dataset is distributed under the Creative Commons Attribution 4.0 licence.

- Dataset card: https://huggingface.co/datasets/google/speech_commands
- Original dataset archive: https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
- Dataset paper: https://arxiv.org/abs/1804.03209

The classification target has 12 classes: ten core commands (`yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`), `unknown`, and `silence`. Auxiliary spoken words are grouped into `unknown`; one-second clips sampled from background-noise recordings form the `silence` class.

## Repository structure

```text
notebooks/       # Dataset exploration, preprocessing, training, evaluation, prediction
src/             # Preprocessing, classical/deep models, prediction, retraining
api/             # FastAPI inference and retraining endpoints
app/             # Streamlit dashboard
data/            # Local source and processed data (not committed)
models/          # Locally generated model artifacts (not committed)
tests/
load_testing/
requirements.txt
```

## Local setup (Windows / Git Bash)

1. Clone the repository:

   ```bash
   git clone https://github.com/Abejirin-King/machine_learning_cycle.git
   cd machine_learning_cycle
   ```

2. Create and activate a virtual environment:

   ```bash
   py -3.11 -m venv .venv
   source .venv/Scripts/activate
   ```

   If the `py` launcher is unavailable, use `python -m venv .venv`.

3. Install dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Run notebooks 01-05 in order to prepare data and train/evaluate the models. Run notebook 06 to test single-file and batch prediction. Put inference WAV files in `data/inference_samples/`.

5. Ensure generated artifacts are present locally before starting the API:
   - `models/final_test_metrics.json`
   - `models/small_audio_cnn.pt`
   - `models/linear_svm.joblib`
   - `models/random_forest.joblib`
   - `notebooks/data/processed/mfcc_features.npz` (needed for retraining)

   These files are intentionally excluded from GitHub because they are generated from the dataset. Run the relevant notebooks to recreate them.

## Run the API and dashboard

Open two Git Bash terminals from the repository root. Activate the virtual environment in both.

**Terminal 1: FastAPI**

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

API documentation: http://127.0.0.1:8000/docs

**Terminal 2: Streamlit dashboard**

```bash
streamlit run app/dashboard.py
```

The dashboard opens in the browser, normally at http://localhost:8501. It connects to the API at http://127.0.0.1:8000 by default. If the API runs elsewhere, change the FastAPI base URL in the dashboard sidebar or set the `API_BASE_URL` environment variable before starting Streamlit.

The dashboard includes:
- API readiness, active model, and observed request count.
- Single WAV prediction and top class scores.
- Batch WAV prediction with downloadable CSV output.
- Waveform, log-Mel spectrogram, and MFCC visualisations.
- Held-out test metrics from the local `models/final_test_metrics.json`.
- Authenticated submission of labelled WAV files and a `filename,label` CSV to a persistent remote training queue.
- Worker registration, job assignment, worker heartbeats, validation results and candidate checkpoint download.
- Local CNN fine-tuning on registered contributor computers; Render remains the lightweight coordinator.

Remote retraining uses Supabase for durable job metadata and private audio storage. Follow [api/REMOTE_TRAINING_SETUP.md](api/REMOTE_TRAINING_SETUP.md) to configure secrets, database tables, storage and worker computers. Candidate promotion is based on validation macro-F1; the held-out test split is not used. A candidate must be reviewed and manually released before deployment.

## Development and deployment notes

The local workflow uses Python directly and does not require Docker Desktop. Container-based deployment and scaling tests will be handled separately in a cloud environment if required by the assignment. The next milestone is cloud deployment and Locust performance testing at different container counts.

## Attribution

Speech Commands was created by Pete Warden and contributors. Please retain the dataset attribution and licence information when redistributing derived materials.
