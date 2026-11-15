"""Lightweight tests for retraining input validation."""
from pathlib import Path

import pandas as pd
import pytest

from src.retraining import retrain_from_labeled_audio


def test_retraining_rejects_metadata_without_required_columns(tmp_path: Path):
    metadata = tmp_path / "labels.csv"
    pd.DataFrame({"audio": ["yes.wav"], "command": ["yes"]}).to_csv(
        metadata, index=False
    )
    with pytest.raises(ValueError, match="filename.*label"):
        retrain_from_labeled_audio(metadata, tmp_path, tmp_path)
