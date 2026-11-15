"""FastAPI service for single/batch audio prediction and controlled retraining."""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.prediction import AudioCommandPredictor
from src.retraining import retrain_from_labeled_audio

PROJECT_ROOT = Path(os.environ.get(
    "PROJECT_ROOT", Path(__file__).resolve().parents[1]
)).resolve()
STARTED_AT = time.time()
REQUEST_COUNT = 0

app = FastAPI(
    title="Voice Command Recognition API",
    version="1.0.0",
    description=(
        "Predict spoken commands from WAV recordings and trigger controlled "
        "retraining with newly labelled audio."
    ),
)


def _predictor() -> AudioCommandPredictor:
    try:
        return AudioCommandPredictor(PROJECT_ROOT)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


async def _save_upload(upload: UploadFile, destination: Path) -> None:
    name = Path(upload.filename or "").name
    if not name or Path(name).suffix.lower() != ".wav":
        raise HTTPException(status_code=400, detail="Only WAV audio files are accepted.")
    destination = destination / name
    destination.write_bytes(await upload.read())


@app.get("/health")
def health():
    """Return service readiness and model artifact status."""
    model_dir = PROJECT_ROOT / "models"
    required = [
        model_dir / "final_test_metrics.json",
        model_dir / "small_audio_cnn.pt",
        model_dir / "linear_svm.joblib",
        model_dir / "random_forest.joblib",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    deployment = model_dir / "deployment_model.json"
    return {
        "status": "ok" if not missing else "degraded",
        "model_artifacts_ready": not missing,
        "missing_artifacts": missing,
        "deployment_config_present": deployment.is_file(),
    }


@app.get("/metrics")
def metrics():
    """Expose lightweight service metrics for the dashboard and load tests."""
    global REQUEST_COUNT
    model_name = None
    try:
        model_name = _predictor().model_name
    except HTTPException:
        pass
    return {
        "uptime_seconds": round(time.time() - STARTED_AT, 2),
        "requests_observed": REQUEST_COUNT,
        "active_model": model_name,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict one WAV file."""
    global REQUEST_COUNT
    REQUEST_COUNT += 1
    with tempfile.TemporaryDirectory(prefix="voice-predict-") as temp:
        folder = Path(temp)
        await _save_upload(file, folder)
        path = folder / Path(file.filename or "").name
        try:
            return _predictor().predict_one(path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """Predict multiple WAV files in one request."""
    global REQUEST_COUNT
    REQUEST_COUNT += 1
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one WAV file.")
    with tempfile.TemporaryDirectory(prefix="voice-batch-") as temp:
        folder = Path(temp)
        paths = []
        try:
            for upload in files:
                await _save_upload(upload, folder)
                paths.append(folder / Path(upload.filename or "").name)
            results = _predictor().predict_files(paths)
            return JSONResponse(content=json.loads(results.to_json(orient="records")))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/retrain")
async def retrain(
    metadata: UploadFile = File(...),
    audio_files: list[UploadFile] = File(...),
):
    """Upload labelled WAV files and attempt validation-gated model promotion.

    The metadata CSV must have columns: filename,label. Labels must match the
    system's existing 12 classes. The API never evaluates on the held-out test set.
    """
    global REQUEST_COUNT
    REQUEST_COUNT += 1
    if Path(metadata.filename or "").suffix.lower() != ".csv":
        raise HTTPException(status_code=400, detail="Metadata must be a CSV file.")
    with tempfile.TemporaryDirectory(prefix="voice-retrain-") as temp:
        root = Path(temp)
        audio_dir = root / "audio"
        audio_dir.mkdir()
        metadata_path = root / "labels.csv"
        metadata_path.write_bytes(await metadata.read())
        try:
            for upload in audio_files:
                await _save_upload(upload, audio_dir)
            result = retrain_from_labeled_audio(
                metadata_path, audio_dir, PROJECT_ROOT
            )
            return result
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
