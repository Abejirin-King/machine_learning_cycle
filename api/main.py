"""FastAPI service for single/batch audio prediction and controlled retraining."""
from __future__ import annotations

import json
import os
import tempfile
import time
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from src.prediction import AudioCommandPredictor
from src.model_artifacts import ensure_release_checkpoint
from api.training_jobs import router as training_router

PROJECT_ROOT = Path(os.environ.get(
    "PROJECT_ROOT", Path(__file__).resolve().parents[1]
)).resolve()
STARTED_AT = time.time()
REQUEST_COUNT = 0
MODEL_BOOTSTRAP_ERROR: str | None = None

app = FastAPI(
    title="Voice Command Recognition API",
    version="1.0.0",
    description=(
        "Predict spoken commands from WAV recordings and trigger controlled "
        "retraining with newly labelled audio."
    ),
)


def custom_openapi():
    """Make Swagger render repeated multipart uploads as file selectors."""
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # FastAPI versions can omit format=binary from array UploadFile schemas.
    # Swagger UI then renders text fields instead of file upload controls.
    component_schemas = schema.get("components", {}).get("schemas", {})
    for path, field_name in (
        ("/predict/batch", "files"),
        ("/retrain", "audio_files"),
    ):
        operation = schema.get("paths", {}).get(path, {}).get("post", {})
        request_body = operation.get("requestBody", {})
        multipart = request_body.get("content", {}).get("multipart/form-data", {})
        body_schema = multipart.get("schema", {})
        if "$ref" in body_schema:
            component_name = body_schema["$ref"].rsplit("/", 1)[-1]
            body_schema = component_schemas.get(component_name, {})
        properties = body_schema.get("properties", {})
        if field_name in properties:
            properties[field_name] = {
                "title": field_name.replace("_", " ").title(),
                "type": "array",
                "items": {"type": "string", "format": "binary"},
                "description": "Select one or more WAV files.",
            }

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi
app.include_router(training_router)


@app.on_event("startup")
def load_pinned_model_release():
    """Download the configured public release checkpoint if it is absent."""
    global MODEL_BOOTSTRAP_ERROR
    try:
        metrics_path = PROJECT_ROOT / "models" / "final_test_metrics.json"
        if metrics_path.is_file():
            evaluation = json.loads(metrics_path.read_text(encoding="utf-8"))
            deployment_path = PROJECT_ROOT / "models" / "deployment_model.json"
            deployment = (
                json.loads(deployment_path.read_text(encoding="utf-8"))
                if deployment_path.is_file() else {}
            )
            selected = deployment.get(
                "model_name", evaluation.get("validation_selected_model")
            )
            if selected == "small_audio_cnn":
                checkpoint = ensure_release_checkpoint(PROJECT_ROOT / "models")
                print(f"Model checkpoint ready: {checkpoint.name}")
    except Exception as exc:
        MODEL_BOOTSTRAP_ERROR = str(exc)
        print(f"WARNING: model release bootstrap failed: {exc}")



@lru_cache(maxsize=1)
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
    """Check only the artifact required by the currently selected model."""
    model_dir = PROJECT_ROOT / "models"
    metrics_path = model_dir / "final_test_metrics.json"
    deployment_path = model_dir / "deployment_model.json"
    missing = []
    active_model = None
    required_model_path = None

    if not metrics_path.is_file():
        missing.append(str(metrics_path))
    else:
        try:
            with metrics_path.open("r", encoding="utf-8") as handle:
                evaluation = json.load(handle)
            deployment = {}
            if deployment_path.is_file():
                with deployment_path.open("r", encoding="utf-8") as handle:
                    deployment = json.load(handle)

            active_model = str(
                deployment.get(
                    "model_name",
                    evaluation.get("validation_selected_model", ""),
                )
            )
            if active_model == "linear_svm_retrained":
                model_file = deployment.get(
                    "model_file", "linear_svm_retrained.joblib"
                )
            elif active_model in {"linear_svm", "random_forest"}:
                model_file = f"{active_model}.joblib"
            elif active_model == "small_audio_cnn":
                model_file = "small_audio_cnn.pt"
            else:
                model_file = None

            if not model_file:
                missing.append(
                    f"Unsupported or missing selected model name: {active_model!r}"
                )
            else:
                required_model_path = model_dir / model_file
                if not required_model_path.is_file():
                    missing.append(str(required_model_path))
        except (OSError, json.JSONDecodeError, AttributeError, TypeError) as exc:
            missing.append(f"Unable to read model selection metadata: {exc}")

    ready = not missing
    return {
        "status": "ok" if ready else "degraded",
        "model_artifacts_ready": ready,
        "missing_artifacts": missing,
        "active_model": active_model,
        "required_model_artifact": (
            str(required_model_path) if required_model_path else None
        ),
        "deployment_config_present": deployment_path.is_file(),
        "model_bootstrap_error": MODEL_BOOTSTRAP_ERROR,
        "model_release_tag": os.getenv("MODEL_RELEASE_TAG", "v1.0.0"),
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
            predictor = await run_in_threadpool(_predictor)
            return await run_in_threadpool(predictor.predict_one, path)
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
            predictor = await run_in_threadpool(_predictor)
            results = await run_in_threadpool(predictor.predict_files, paths)
            return JSONResponse(content=json.loads(results.to_json(orient="records")))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/retrain")
async def retrain():
    """Retraining is intentionally not run on the memory-limited API host.

    Use the local training command documented in api/README.md. This keeps
    training off Render and prevents promotion without the original train and
    validation audio.
    """
    raise HTTPException(
        status_code=501,
        detail=(
            "CNN retraining is not available on the hosted inference API. "
            "Run the local training workflow with the original Speech Commands "
            "dataset and audio index; see api/README.md."
        ),
    )
