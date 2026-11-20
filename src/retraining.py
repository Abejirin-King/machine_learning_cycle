"""Validation-gated fine-tuning for the deployed SmallAudioCNN.

Run this on a machine containing the original Speech Commands audio and split
index. New labelled recordings join only the training split. The unchanged
validation split controls promotion; the held-out test split is never used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.cnn_model import SmallAudioCNN
from src.deep_learning import (
    SpeechCommandsMelDataset, evaluate_cnn, save_cnn_bundle, train_cnn,
)

VALID_LABELS = {
    "down", "go", "left", "no", "off", "on",
    "right", "silence", "stop", "unknown", "up", "yes",
}


def _load_index(index_path: Path, root: Path) -> pd.DataFrame:
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Original audio index not found: {index_path}. "
            "Run notebooks 01 and 02 on this machine first."
        )
    frame = pd.read_csv(index_path)
    required = {"path", "target", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Audio index is missing columns: {sorted(missing)}")
    if "offset_seconds" not in frame:
        frame["offset_seconds"] = 0.0
    frame["target"] = frame["target"].astype(str).str.lower().str.strip()
    frame["split"] = frame["split"].astype(str).str.lower().str.strip()
    frame["path"] = frame["path"].map(
        lambda value: str((root / value).resolve())
        if not Path(str(value)).is_absolute()
        else str(Path(str(value)).resolve())
    )
    absent = [p for p in frame["path"].unique() if not Path(p).is_file()]
    if absent:
        raise FileNotFoundError(
            f"{len(absent)} original audio paths are unavailable. First: {absent[0]}"
        )
    return frame


def _validate_uploads(metadata_path: Path, audio_dir: Path) -> pd.DataFrame:
    try:
        metadata = pd.read_csv(metadata_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("Metadata CSV is empty or unreadable.") from exc
    if not {"filename", "label"}.issubset(metadata.columns):
        raise ValueError("Metadata CSV must contain filename,label columns.")
    metadata = metadata[["filename", "label"]].dropna().copy()
    metadata["filename"] = metadata["filename"].astype(str).map(
        lambda name: Path(name).name
    )
    metadata["label"] = metadata["label"].astype(str).str.lower().str.strip()
    if metadata.empty:
        raise ValueError("No labelled recordings were provided.")
    if metadata["filename"].duplicated().any():
        raise ValueError("Each uploaded filename must appear only once.")
    invalid = sorted(set(metadata["label"]) - VALID_LABELS)
    if invalid:
        raise ValueError(f"Unsupported labels: {invalid}. Valid labels: {sorted(VALID_LABELS)}")
    rows = []
    for row in metadata.itertuples(index=False):
        path = audio_dir / row.filename
        if not path.is_file() or path.suffix.lower() != ".wav":
            raise ValueError(f"Missing WAV file: {row.filename}")
        info = sf.info(str(path))
        if info.frames <= 0 or info.duration <= 0:
            raise ValueError(f"Audio file is empty: {row.filename}")
        rows.append({
            "path": str(path.resolve()), "target": row.label,
            "split": "train", "offset_seconds": 0.0,
        })
    return pd.DataFrame(rows)


def _make_loaders(index: pd.DataFrame, labels: list[str], batch_size: int = 64):
    label_to_id = {label: i for i, label in enumerate(labels)}
    unknown = sorted(set(index["target"].astype(str)) - set(labels))
    if unknown:
        raise ValueError(f"Training index contains labels absent from checkpoint: {unknown}")
    train_frame = index[index["split"] == "train"].copy()
    validation_frame = index[index["split"] == "validation"].copy()
    if train_frame.empty or validation_frame.empty:
        raise ValueError("Both train and validation splits must contain recordings.")
    train_ds = SpeechCommandsMelDataset(train_frame, label_to_id, 16_000, 64)
    validation_ds = SpeechCommandsMelDataset(validation_frame, label_to_id, 16_000, 64)
    counts = train_frame["target"].astype(str).value_counts()
    weights = train_frame["target"].astype(str).map(
        lambda label: 1.0 / float(counts[label])
    ).to_numpy(dtype=np.float64)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights), replacement=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=0, pin_memory=False,
    )
    validation_loader = DataLoader(
        validation_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )
    return train_loader, validation_loader


def retrain_from_labeled_audio(
    metadata_path: str | Path,
    audio_dir: str | Path,
    project_root: str | Path,
    *,
    index_path: str | Path | None = None,
    epochs: int = 3,
    learning_rate: float = 1e-4,
) -> dict:
    """Fine-tune the current CNN; save a candidate only if validation F1 improves."""
    root = Path(project_root).resolve()
    model_dir = root / "models"
    checkpoint_path = model_dir / "small_audio_cnn.pt"

    if epochs < 1 or epochs > 20:
        raise ValueError("epochs must be between 1 and 20.")

    # Validate the submitted metadata and WAV files before checking training
    # artifacts. Invalid uploads should produce an input error even on a
    # machine that does not have the model checkpoint or original dataset.
    added = _validate_uploads(Path(metadata_path), Path(audio_dir))

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Current CNN checkpoint not found: {checkpoint_path}")
    if not (model_dir / "final_test_metrics.json").is_file():
        raise FileNotFoundError("models/final_test_metrics.json is missing.")

    data_index_path = Path(index_path) if index_path else (
        root / "notebooks" / "data" / "processed" / "audio_index_with_silence.csv"
    )
    original = _load_index(data_index_path, root)
    original_labels = sorted(original["target"].astype(str).unique().tolist())

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    labels = [str(label) for label in checkpoint["labels"]]
    if labels != original_labels:
        raise ValueError(
            "Local audio-index labels do not match checkpoint labels. "
            f"Checkpoint={labels}; index={original_labels}"
        )

    combined = pd.concat([original, added], ignore_index=True)
    combined["split"] = combined["split"].astype(str)
    train_loader, validation_loader = _make_loaders(combined, labels)

    baseline_model = SmallAudioCNN(n_classes=len(labels))
    baseline_model.load_state_dict(checkpoint["state_dict"])
    baseline = evaluate_cnn(baseline_model, validation_loader, labels, device="cpu")

    candidate = SmallAudioCNN(n_classes=len(labels))
    candidate.load_state_dict(checkpoint["state_dict"])
    candidate, history = train_cnn(
        candidate, train_loader, validation_loader,
        epochs=epochs, learning_rate=learning_rate, device="cpu",
    )
    candidate_metrics = evaluate_cnn(
        candidate, validation_loader, labels, device="cpu"
    )
    baseline_f1 = float(baseline["macro_f1"])
    candidate_f1 = float(candidate_metrics["macro_f1"])
    promoted = candidate_f1 > baseline_f1

    model_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = model_dir / "small_audio_cnn_candidate.pt"
    if promoted:
        save_cnn_bundle(candidate, labels, candidate_path, n_mels=64)

    result = {
        "candidate_model": "small_audio_cnn",
        "added_recordings": int(len(added)),
        "training_samples_before": int((original["split"] == "train").sum()),
        "training_samples_after": int((combined["split"] == "train").sum()),
        "baseline_validation_macro_f1": baseline_f1,
        "candidate_validation_macro_f1": candidate_f1,
        "promoted": promoted,
        "candidate_checkpoint": str(candidate_path) if promoted else None,
        "epochs": int(epochs),
        "training_history": history.to_dict(orient="records"),
        "validation_metrics": {
            key: float(candidate_metrics[key]) for key in (
                "accuracy", "balanced_accuracy", "macro_precision",
                "macro_recall", "macro_f1",
            )
        },
        "note": (
            "Promotion decision used the unchanged validation split only. "
            "The held-out test split was not loaded or used."
        ),
    }
    (model_dir / "latest_retraining_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune the current CNN using newly labelled WAV files."
    )
    parser.add_argument("--metadata", required=True, help="CSV with filename,label")
    parser.add_argument("--audio-dir", required=True, help="Directory containing WAVs")
    parser.add_argument("--project-root", default=".", help="Repository root")
    parser.add_argument("--index", default=None, help="Optional audio index CSV")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    result = retrain_from_labeled_audio(
        args.metadata, args.audio_dir, args.project_root,
        index_path=args.index, epochs=args.epochs,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
