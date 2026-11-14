"""Audio preprocessing utilities for the Speech Commands project.

The module keeps the official Speech Commands train/validation/test split,
creates one-second silence examples from background recordings, and extracts
fixed-length MFCC summary features for classical machine-learning models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm.auto import tqdm


CORE_COMMANDS = {
    "yes", "no", "up", "down", "left",
    "right", "on", "off", "stop", "go",
}
VALID_SPLITS = {"train", "validation", "test"}
INDEX_COLUMNS = {
    "path", "relative_path", "source_label", "target",
    "speaker_id", "split",
}


def _normalise_split(value: str) -> str:
    """Map the notebook's 'test' split to the canonical test name."""
    value = str(value).strip().lower()
    if value not in VALID_SPLITS:
        raise ValueError(
            f"Unsupported split {value!r}. Expected one of {sorted(VALID_SPLITS)}."
        )
    return value


def build_audio_index(
    index_path: str | Path,
    dataset_dir: str | Path,
    *,
    silence_stride_seconds: float = 1.0,
) -> pd.DataFrame:
    """Load the speech index and append silence segments from background noise.

    Background recordings are assigned to splits by source file, not by clip.
    This prevents segments from one continuous recording leaking across splits.
    With the six standard background recordings, four are assigned to train,
    one to validation, and one to test.
    """
    index_path = Path(index_path)
    dataset_dir = Path(dataset_dir)

    if not index_path.is_file():
        raise FileNotFoundError(
            f"Audio index not found: {index_path}. Run notebook 01 first."
        )
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Extracted dataset directory not found: {dataset_dir}")
    if silence_stride_seconds <= 0:
        raise ValueError("silence_stride_seconds must be greater than zero.")

    speech = pd.read_csv(index_path)
    missing = INDEX_COLUMNS.difference(speech.columns)
    if missing:
        raise ValueError(f"Audio index is missing columns: {sorted(missing)}")

    speech = speech.copy()
    speech["path"] = speech["path"].map(lambda p: str(Path(p)))
    speech["split"] = speech["split"].map(_normalise_split)
    speech["offset_seconds"] = 0.0

    # The source notebook groups all non-core spoken words into "unknown".
    valid_targets = CORE_COMMANDS | {"unknown"}
    unexpected = set(speech["target"].dropna().unique()) - valid_targets
    if unexpected:
        raise ValueError(f"Unexpected speech target labels: {sorted(unexpected)}")

    noise_dir = dataset_dir / "_background_noise_"
    noise_files = sorted(noise_dir.glob("*.wav"))
    if not noise_files:
        raise FileNotFoundError(
            f"No background-noise WAV files found in {noise_dir}"
        )

    # Assign complete source recordings to splits before making segments.
    # This avoids train/test leakage from adjacent chunks of the same noise.
    noise_split: dict[str, str] = {}
    for i, noise_path in enumerate(noise_files):
        if i < max(1, len(noise_files) - 2):
            split = "train"
        elif i == len(noise_files) - 2:
            split = "validation"
        else:
            split = "test"
        noise_split[noise_path.name] = split

    silence_rows: list[dict] = []
    for noise_path in noise_files:
        info = sf.info(str(noise_path))
        duration = float(info.duration)
        split = noise_split[noise_path.name]
        n_segments = int(duration // silence_stride_seconds)

        for segment_number in range(n_segments):
            offset = segment_number * silence_stride_seconds
            silence_rows.append({
                "path": str(noise_path.resolve()),
                "relative_path": (
                    f"_background_noise_/{noise_path.name}"
                    f"#offset={offset:.3f}"
                ),
                "source_label": "_background_noise_",
                "target": "silence",
                "speaker_id": noise_path.stem,
                "split": split,
                "offset_seconds": float(offset),
            })

    if not silence_rows:
        raise ValueError(
            "No one-second silence segments could be created. "
            "Check the background-noise recordings."
        )

    combined = pd.concat(
        [speech, pd.DataFrame(silence_rows)],
        ignore_index=True,
    )
    combined["split"] = combined["split"].astype("category")
    combined["target"] = combined["target"].astype("category")

    # Fail early if any source path is missing.
    missing_paths = [p for p in combined["path"].unique() if not Path(p).is_file()]
    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)} audio paths are missing. First: {missing_paths[0]}"
        )

    return combined


def save_audio_index(audio_index: pd.DataFrame, output_path: str | Path) -> Path:
    """Save the complete speech + silence index as CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio_index.to_csv(output_path, index=False)
    return output_path


def load_audio_segment(
    path: str | Path,
    *,
    sample_rate: int = 16_000,
    duration_seconds: float = 1.0,
    offset_seconds: float = 0.0,
) -> np.ndarray:
    """Load mono audio, resample it, then pad or trim to a fixed duration."""
    if sample_rate <= 0 or duration_seconds <= 0 or offset_seconds < 0:
        raise ValueError("Sample rate/duration must be positive and offset non-negative.")

    audio, _ = librosa.load(
        str(path),
        sr=sample_rate,
        mono=True,
        offset=offset_seconds,
        duration=duration_seconds,
    )
    expected_length = int(round(sample_rate * duration_seconds))

    if len(audio) < expected_length:
        audio = np.pad(audio, (0, expected_length - len(audio)))
    elif len(audio) > expected_length:
        audio = audio[:expected_length]

    return audio.astype(np.float32, copy=False)


def extract_mfcc_features(
    audio_index: pd.DataFrame,
    *,
    sample_rate: int = 16_000,
    duration_seconds: float = 1.0,
    n_mfcc: int = 40,
    n_fft: int = 400,
    hop_length: int = 160,
    max_samples_per_class: int | None = None,
    random_state: int = 42,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Extract mean and standard deviation of MFCCs for every indexed clip.

    Returns:
        X: float32 feature matrix with 2 * n_mfcc columns.
        y: string labels.
        metadata: rows corresponding to X and y, in the same order.

    max_samples_per_class is optional and intended only for quick experiments.
    Leave it as None for the full dataset and final evaluation.
    """
    required = {"path", "target", "split"}
    missing = required.difference(audio_index.columns)
    if missing:
        raise ValueError(f"Audio index is missing columns: {sorted(missing)}")

    if n_mfcc <= 0 or n_fft <= 0 or hop_length <= 0:
        raise ValueError("MFCC parameters must be positive.")

    selected = audio_index.copy()
    if max_samples_per_class is not None:
        if max_samples_per_class < 1:
            raise ValueError("max_samples_per_class must be at least 1.")
        selected = (
            selected.groupby(["split", "target"], observed=True, group_keys=False)
            .apply(
                lambda group: group.sample(
                    n=min(len(group), max_samples_per_class),
                    random_state=random_state,
                ),
                include_groups=True,
            )
            .reset_index(drop=True)
        )

    features: list[np.ndarray] = []
    iterator: Iterable = selected.itertuples(index=False)
    if show_progress:
        iterator = tqdm(iterator, total=len(selected), desc="Extracting MFCCs")

    has_offset = "offset_seconds" in selected.columns
    for row in iterator:
        offset = float(getattr(row, "offset_seconds", 0.0)) if has_offset else 0.0
        audio = load_audio_segment(
            row.path,
            sample_rate=sample_rate,
            duration_seconds=duration_seconds,
            offset_seconds=offset,
        )
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        summary = np.concatenate(
            [mfcc.mean(axis=1), mfcc.std(axis=1)]
        ).astype(np.float32)
        features.append(summary)

    X = np.vstack(features).astype(np.float32)
    y = selected["target"].astype(str).to_numpy()
    metadata = selected.reset_index(drop=True)
    return X, y, metadata


def split_feature_arrays(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
) -> dict[str, dict[str, np.ndarray]]:
    """Separate extracted features into train, validation, and test arrays."""
    if len(X) != len(y) or len(X) != len(metadata):
        raise ValueError("X, y, and metadata must have the same number of rows.")

    output: dict[str, dict[str, np.ndarray]] = {}
    splits = metadata["split"].astype(str).to_numpy()
    for split in ("train", "validation", "test"):
        mask = splits == split
        if not mask.any():
            raise ValueError(f"No samples found for split {split!r}.")
        output[split] = {
            "X": X[mask],
            "y": y[mask],
        }
    return output


def class_distribution(audio_index: pd.DataFrame) -> pd.DataFrame:
    """Return a split-by-class count table, including absent combinations."""
    return (
        audio_index.groupby(["split", "target"], observed=True)
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
