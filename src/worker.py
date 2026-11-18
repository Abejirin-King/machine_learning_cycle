"""Polling worker for dashboard-submitted CNN retraining jobs.

Run from a clone that already contains the original dataset, audio index and model.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import requests

from src.retraining import retrain_from_labeled_audio


def main():
    parser = argparse.ArgumentParser(description="Connect this computer as an audio CNN training worker.")
    parser.add_argument("--server-url", default=os.getenv("TRAINING_API_URL", "https://voice-command-api.onrender.com"))
    parser.add_argument("--worker-id", default=os.getenv("TRAINING_WORKER_ID"))
    parser.add_argument("--token", default=os.getenv("TRAINING_WORKER_TOKEN"))
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--poll-seconds", type=int, default=20)
    args = parser.parse_args()
    if not args.worker_id or not args.token:
        parser.error("Provide --worker-id and --token, or set TRAINING_WORKER_ID and TRAINING_WORKER_TOKEN.")
    root = Path(args.project_root).resolve()
    headers = {"Authorization": f"Bearer {args.token}"}
    session = requests.Session()

    print(f"Worker connected to {args.server_url}. Press Ctrl+C to stop.")
    while True:
        try:
            session.post(f"{args.server_url}/training/workers/{args.worker_id}/heartbeat", headers=headers, timeout=30).raise_for_status()
            response = session.post(f"{args.server_url}/training/workers/{args.worker_id}/claim", headers=headers, timeout=60)
            response.raise_for_status()
            job = response.json().get("job")
            if not job:
                time.sleep(max(5, args.poll_seconds))
                continue

            job_id = job["id"]
            print(f"Claimed job {job_id}: {job['recording_count']} recordings, {job['epochs']} epochs.")
            with tempfile.TemporaryDirectory(prefix=f"voice-job-{job_id}-") as tmp:
                folder = Path(tmp)
                audio_dir = folder / "audio"
                audio_dir.mkdir()
                metadata_path = folder / "metadata.csv"
                for filename, url in job["audio_urls"].items():
                    data = session.get(url, timeout=90)
                    data.raise_for_status()
                    (audio_dir / Path(filename).name).write_bytes(data.content)
                csv_response = session.get(job["metadata_url"], timeout=60)
                csv_response.raise_for_status()
                metadata_path.write_bytes(csv_response.content)

                result = retrain_from_labeled_audio(
                    metadata_path, audio_dir, root, epochs=int(job["epochs"])
                )
                files = {"candidate": ("small_audio_cnn_candidate.pt",
                    (root / "models" / "small_audio_cnn_candidate.pt").read_bytes(),
                    "application/octet-stream")} if result.get("promoted") else {}
                data = {"result_json": json.dumps(result)}
                submitted = session.post(
                    f"{args.server_url}/training/workers/{args.worker_id}/jobs/{job_id}/result",
                    headers=headers, data=data, files=files or None, timeout=180,
                )
                submitted.raise_for_status()
                print(f"Job {job_id} finished: {submitted.json()['status']}.")
        except KeyboardInterrupt:
            print("Worker stopped.")
            break
        except Exception as exc:
            print(f"Worker error: {exc}")
            time.sleep(max(10, args.poll_seconds))


if __name__ == "__main__":
    main()
