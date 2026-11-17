"""Download the pinned CNN checkpoint from a GitHub Release when absent."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path


REPOSITORY = "Abejirin-King/machine_learning_cycle"
CHECKPOINT_NAME = "small_audio_cnn.pt"


def ensure_release_checkpoint(model_dir: str | Path) -> Path:
    """Ensure the pinned release checkpoint exists locally and matches its digest.

    Set MODEL_RELEASE_TAG to a release tag such as v1.0.0. The default is the
    initial validated model. Existing local checkpoints are left untouched.
    """
    destination_dir = Path(model_dir)
    destination = destination_dir / CHECKPOINT_NAME
    if destination.is_file() and destination.stat().st_size > 0:
        return destination

    tag = os.environ.get("MODEL_RELEASE_TAG", "v1.0.0").strip()
    if not tag:
        raise RuntimeError("MODEL_RELEASE_TAG cannot be empty.")

    api_url = (
        f"https://api.github.com/repos/{REPOSITORY}/releases/tags/{tag}"
    )
    request = urllib.request.Request(
        api_url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "machine-learning-cycle-model-loader",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            release = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Could not retrieve GitHub release {tag!r}: {exc}"
        ) from exc

    asset = next(
        (item for item in release.get("assets", [])
         if item.get("name") == CHECKPOINT_NAME),
        None,
    )
    if asset is None:
        raise RuntimeError(
            f"Release {tag!r} does not contain {CHECKPOINT_NAME}."
        )

    download_url = asset.get("browser_download_url")
    if not download_url:
        raise RuntimeError("The GitHub release asset has no download URL.")

    destination_dir.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    download_request = urllib.request.Request(
        download_url,
        headers={"User-Agent": "machine-learning-cycle-model-loader"},
    )
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(download_request, timeout=120) as response:
            with temporary.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    output.write(chunk)
        expected = asset.get("digest")
        actual = f"sha256:{digest.hexdigest()}"
        if expected and actual.lower() != expected.lower():
            raise RuntimeError(
                f"Checkpoint checksum mismatch for release {tag!r}: "
                f"expected {expected}, received {actual}."
            )
        if temporary.stat().st_size == 0:
            raise RuntimeError("GitHub returned an empty model checkpoint.")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return destination
