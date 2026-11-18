"""Locust load tests for the deployed voice-command FastAPI.

Configure audio with LOCUST_AUDIO_DIR or LOCUST_AUDIO_FILES. The health task
still runs when no WAV samples are available; prediction tasks are skipped.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

from locust import HttpUser, between, task


def _load_audio_samples() -> list[tuple[str, bytes]]:
    explicit = os.getenv("LOCUST_AUDIO_FILES", "").strip()
    if explicit:
        paths = [Path(item.strip()) for item in explicit.split(",") if item.strip()]
    else:
        audio_dir = Path(os.getenv("LOCUST_AUDIO_DIR", "data/inference_samples"))
        paths = sorted(audio_dir.glob("*.wav")) if audio_dir.exists() else []

    samples: list[tuple[str, bytes]] = []
    for path in paths:
        if path.is_file() and path.suffix.lower() == ".wav":
            try:
                payload = path.read_bytes()
            except OSError:
                continue
            if payload:
                samples.append((path.name, payload))
    return samples


AUDIO_SAMPLES = _load_audio_samples()
try:
    BATCH_SIZE = max(1, min(int(os.getenv("LOCUST_BATCH_SIZE", "3")), 10))
except ValueError:
    BATCH_SIZE = 3


class VoiceCommandApiUser(HttpUser):
    """A simulated client making health and audio-inference requests."""

    wait_time = between(1, 3)

    def on_start(self) -> None:
        self.audio_samples = AUDIO_SAMPLES
        self.batch_size = BATCH_SIZE

    @task(3)
    def health(self) -> None:
        self.client.get("/health", name="/health")

    @task(2)
    def predict_single(self) -> None:
        if not self.audio_samples:
            return
        filename, payload = random.choice(self.audio_samples)
        self.client.post(
            "/predict",
            files={"file": (filename, payload, "audio/wav")},
            name="/predict",
        )

    @task(1)
    def predict_batch(self) -> None:
        if not self.audio_samples:
            return
        count = min(self.batch_size, len(self.audio_samples))
        selected = random.sample(self.audio_samples, count)
        files = [
            ("files", (filename, payload, "audio/wav"))
            for filename, payload in selected
        ]
        self.client.post("/predict/batch", files=files, name="/predict/batch")
