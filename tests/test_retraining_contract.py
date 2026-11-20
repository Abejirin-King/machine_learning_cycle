"""Lightweight tests for retraining input validation."""
from pathlib import Path

import pandas as pd
import pytest

from src.retraining import retrain_from_labeled_audio


def _write_metadata(path: Path, filename: str, label: str) -> Path:
    pd.DataFrame({"filename": [filename], "label": [label]}).to_csv(
        path, index=False
    )
    return path


def test_retraining_rejects_metadata_without_required_columns(tmp_path: Path):
    metadata = tmp_path / "labels.csv"
    pd.DataFrame({"audio": ["yes.wav"], "command": ["yes"]}).to_csv(
        metadata, index=False
    )
    with pytest.raises(ValueError, match="filename.*label"):
        retrain_from_labeled_audio(metadata, tmp_path, tmp_path)


def test_retraining_rejects_unsupported_label(tmp_path: Path):
    metadata = _write_metadata(tmp_path / "labels.csv", "sample.wav", "maybe")
    with pytest.raises(ValueError, match="Unsupported labels"):
        retrain_from_labeled_audio(metadata, tmp_path, tmp_path)


def test_retraining_rejects_missing_wav_file(tmp_path: Path):
    metadata = _write_metadata(tmp_path / "labels.csv", "missing.wav", "yes")
    with pytest.raises(ValueError, match="Missing WAV file"):
        retrain_from_labeled_audio(metadata, tmp_path, tmp_path)


def test_retraining_rejects_empty_metadata_file(tmp_path: Path):
    metadata = tmp_path / "labels.csv"
    metadata.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Metadata CSV is empty"):
        retrain_from_labeled_audio(metadata, tmp_path, tmp_path)


def test_retraining_rejects_metadata_with_no_labelled_rows(tmp_path: Path):
    metadata = tmp_path / "labels.csv"
    pd.DataFrame({"filename": [None], "label": [None]}).to_csv(
        metadata, index=False
    )
    with pytest.raises(ValueError, match="No labelled recordings"):
        retrain_from_labeled_audio(metadata, tmp_path, tmp_path)
