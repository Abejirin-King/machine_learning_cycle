"""Authenticated, Supabase-backed remote training job coordination."""
from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import os
import secrets
import uuid
from datetime import datetime, timezone
from typing import Any

import requests
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

router = APIRouter(prefix="/training", tags=["remote training"])
bearer = HTTPBearer(auto_error=False)
LABELS = {"down", "go", "left", "no", "off", "on", "right", "silence", "stop", "unknown", "up", "yes"}
MAX_AUDIO_FILES = 100
MAX_TOTAL_BYTES = 40 * 1024 * 1024


def _settings() -> tuple[str, str, str]:
    url = os.getenv("SUPABASE_URL", "").rstrip("/")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    bucket = os.getenv("SUPABASE_TRAINING_BUCKET", "training-jobs")
    if not url or not key:
        raise HTTPException(503, "Remote training storage is not configured. Set Supabase URL and service-role key.")
    return url, key, bucket


def _headers(key: str) -> dict[str, str]:
    return {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _db(method: str, table: str, key: str, *, params=None, body=None):
    url, _, _ = _settings()
    response = requests.request(method, f"{url}/rest/v1/{table}", headers=_headers(key), params=params, json=body, timeout=30)
    if not response.ok:
        raise HTTPException(502, f"Training database request failed ({response.status_code}). Check the Supabase schema and API configuration.")
    return response.json() if response.content else None


def _storage_upload(key: str, bucket: str, path: str, content: bytes, content_type: str):
    url, _, _ = _settings()
    response = requests.post(
        f"{url}/storage/v1/object/{bucket}/{path}",
        headers={"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": content_type, "x-upsert": "false"},
        data=content, timeout=90,
    )
    if not response.ok:
        raise HTTPException(502, f"Could not store training upload ({response.status_code}). Check that the storage bucket exists and has sufficient capacity.")


def _signed_url(key: str, bucket: str, path: str) -> str:
    url, _, _ = _settings()
    response = requests.post(
        f"{url}/storage/v1/object/sign/{bucket}/{path}",
        headers=_headers(key), json={"expiresIn": 3600}, timeout=30,
    )
    if not response.ok:
        raise HTTPException(502, "Could not create a temporary download link for the training job.")
    return url + response.json()["signedURL"]


def _dashboard_auth(token: str | None = Header(default=None, alias="X-Dashboard-Token")):
    expected = os.getenv("DASHBOARD_API_TOKEN", "")
    if not expected or not token or not hmac.compare_digest(token, expected):
        raise HTTPException(401, "Dashboard authentication failed.")


def _worker_auth(worker_id: str, credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
    if credentials is None:
        raise HTTPException(401, "Worker bearer token required.")
    _, key, _ = _settings()
    rows = _db("GET", "training_workers", key, params={"id": f"eq.{worker_id}", "select": "id,token_hash,enabled"})
    if not rows or not rows[0].get("enabled"):
        raise HTTPException(401, "Worker is not registered or has been disabled.")
    digest = hashlib.sha256(credentials.credentials.encode()).hexdigest()
    if not hmac.compare_digest(digest, rows[0]["token_hash"]):
        raise HTTPException(401, "Invalid worker token.")
    return worker_id


@router.post("/workers/register")
def register_worker(name: str = Form(...), enrollment_token: str = Header(default="", alias="X-Worker-Enrollment-Token")):
    expected = os.getenv("WORKER_ENROLLMENT_TOKEN", "")
    if not expected or not hmac.compare_digest(enrollment_token, expected):
        raise HTTPException(401, "Worker enrollment token is invalid.")
    if not name.strip() or len(name) > 80:
        raise HTTPException(400, "Worker name must be 1-80 characters.")
    _, key, _ = _settings()
    worker_id, token = str(uuid.uuid4()), secrets.token_urlsafe(40)
    _db("POST", "training_workers", key, params={"select": "id"}, body={
        "id": worker_id, "name": name.strip(), "token_hash": hashlib.sha256(token.encode()).hexdigest(),
        "enabled": True, "last_seen_at": datetime.now(timezone.utc).isoformat(),
    })
    return {"worker_id": worker_id, "worker_token": token, "message": "Save this token securely. It will not be shown again."}


@router.get("/workers")
def list_workers(_: None = Depends(_dashboard_auth)):
    _, key, _ = _settings()
    rows = _db("GET", "training_workers", key, params={"select": "id,name,enabled,last_seen_at,current_job_id", "order": "last_seen_at.desc"})
    return {"workers": rows}


@router.post("/jobs")
async def create_job(
    metadata: UploadFile = File(...),
    audio_files: list[UploadFile] = File(...),
    epochs: int = Form(3),
    _: None = Depends(_dashboard_auth),
):
    if epochs < 1 or epochs > 5:
        raise HTTPException(400, "Epochs must be between 1 and 5 for remotely submitted jobs.")
    if not audio_files or len(audio_files) > MAX_AUDIO_FILES:
        raise HTTPException(400, f"Upload between 1 and {MAX_AUDIO_FILES} WAV files.")
    raw_csv = await metadata.read(1024 * 1024 + 1)
    if len(raw_csv) > 1024 * 1024:
        raise HTTPException(413, "Metadata CSV exceeds 1 MB.")
    try:
        rows = list(csv.DictReader(io.StringIO(raw_csv.decode("utf-8-sig"))))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise HTTPException(400, f"Invalid UTF-8 CSV: {exc}") from exc
    if not rows or not {"filename", "label"}.issubset(rows[0]):
        raise HTTPException(400, "CSV must contain filename,label columns.")
    if len(rows) != len(audio_files):
        raise HTTPException(400, "The CSV row count must match the number of uploaded WAV files.")
    names = [str(row["filename"]).strip() for row in rows]
    if len(set(names)) != len(names) or any(not n or n != n.split("/")[-1] or "\\" in n for n in names):
        raise HTTPException(400, "Filenames must be unique plain filenames without directory paths.")
    if any(str(row["label"]).strip().lower() not in LABELS for row in rows):
        raise HTTPException(400, f"Labels must be from: {', '.join(sorted(LABELS))}.")
    uploads: dict[str, bytes] = {}
    total = len(raw_csv)
    for item in audio_files:
        name = item.filename or ""
        if name not in names or not name.lower().endswith(".wav") or name in uploads:
            raise HTTPException(400, "Uploaded WAV filenames must exactly match the CSV filenames.")
        data = await item.read(MAX_TOTAL_BYTES + 1)
        total += len(data)
        if not data or total > MAX_TOTAL_BYTES:
            raise HTTPException(413, "Combined metadata and audio upload exceeds 40 MB.")
        uploads[name] = data
    if set(uploads) != set(names):
        raise HTTPException(400, "Every CSV filename must have a matching WAV upload.")

    _, key, bucket = _settings()
    job_id = str(uuid.uuid4())
    prefix = f"jobs/{job_id}"
    _storage_upload(key, bucket, f"{prefix}/metadata.csv", raw_csv, "text/csv")
    for name, data in uploads.items():
        _storage_upload(key, bucket, f"{prefix}/audio/{name}", data, "audio/wav")
    _, _, _ = _settings()
    job = {
        "id": job_id, "status": "queued", "epochs": epochs,
        "metadata_path": f"{prefix}/metadata.csv",
        "audio_paths": [f"{prefix}/audio/{name}" for name in names],
        "recording_count": len(names), "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _db("POST", "training_jobs", key, params={"select": "id,status,created_at"}, body=job)
    return {"job_id": job_id, "status": "queued", "recording_count": len(names), "epochs": epochs}


@router.get("/jobs")
def list_jobs(_: None = Depends(_dashboard_auth)):
    _, key, _ = _settings()
    rows = _db("GET", "training_jobs", key, params={"select": "id,status,epochs,recording_count,created_at,started_at,finished_at,worker_id,result,error", "order": "created_at.desc", "limit": "30"})
    return {"jobs": rows}


@router.get("/jobs/{job_id}")
def get_job(job_id: str, _: None = Depends(_dashboard_auth)):
    _, key, _ = _settings()
    rows = _db("GET", "training_jobs", key, params={"id": f"eq.{job_id}", "select": "*"})
    if not rows:
        raise HTTPException(404, "Training job not found.")
    return rows[0]


@router.post("/workers/{worker_id}/heartbeat")
def worker_heartbeat(worker_id: str, _: str = Depends(_worker_auth)):
    _, key, _ = _settings()
    _db("PATCH", "training_workers", key, params={"id": f"eq.{worker_id}"}, body={"last_seen_at": datetime.now(timezone.utc).isoformat()})
    return {"status": "ok"}


@router.post("/workers/{worker_id}/claim")
def claim_job(worker_id: str, _: str = Depends(_worker_auth)):
    _, key, bucket = _settings()
    result = requests.post(f"{os.environ['SUPABASE_URL'].rstrip('/')}/rest/v1/rpc/claim_training_job",
        headers=_headers(key), json={"p_worker_id": worker_id}, timeout=30)
    if not result.ok:
        raise HTTPException(502, "Could not claim a job. Apply api/training_jobs_schema.sql in Supabase.")
    job = result.json()
    if not job:
        return {"job": None}
    job["metadata_url"] = _signed_url(key, bucket, job["metadata_path"])
    job["audio_urls"] = {path.rsplit("/", 1)[-1]: _signed_url(key, bucket, path) for path in job["audio_paths"]}
    return {"job": job}


@router.post("/workers/{worker_id}/jobs/{job_id}/result")
async def submit_result(
    worker_id: str, job_id: str, result_json: str = Form(...),
    candidate: UploadFile | None = File(default=None),
    _: str = Depends(_worker_auth),
):
    _, key, bucket = _settings()
    try:
        result_data = json.loads(result_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "result_json must be valid JSON.") from exc
    candidate_path = None
    if candidate is not None:
        if candidate.filename != "small_audio_cnn_candidate.pt":
            raise HTTPException(400, "Unexpected candidate checkpoint filename.")
        data = await candidate.read(20 * 1024 * 1024 + 1)
        if not data or len(data) > 20 * 1024 * 1024:
            raise HTTPException(413, "Candidate checkpoint must be smaller than 20 MB.")
        candidate_path = f"results/{job_id}/small_audio_cnn_candidate.pt"
        _storage_upload(key, bucket, candidate_path, data, "application/octet-stream")
    status = "candidate_ready" if result_data.get("promoted") and candidate_path else "completed"
    _db("PATCH", "training_jobs", key, params={"id": f"eq.{job_id}", "worker_id": f"eq.{worker_id}", "status": "eq.running"}, body={
        "status": status, "result": result_data, "candidate_path": candidate_path,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    })
    _db("PATCH", "training_workers", key, params={"id": f"eq.{worker_id}"}, body={"current_job_id": None})
    return {"status": status, "candidate_uploaded": bool(candidate_path)}


@router.post("/workers/{worker_id}/jobs/{job_id}/fail")
def fail_job(worker_id: str, job_id: str, error: str = Form(...), _: str = Depends(_worker_auth)):
    _, key, _ = _settings()
    _db("PATCH", "training_jobs", key, params={"id": f"eq.{job_id}", "worker_id": f"eq.{worker_id}", "status": "eq.running"}, body={
        "status": "failed", "error": error[:2000], "finished_at": datetime.now(timezone.utc).isoformat(),
    })
    _db("PATCH", "training_workers", key, params={"id": f"eq.{worker_id}"}, body={"current_job_id": None})
    return {"status": "failed"}
