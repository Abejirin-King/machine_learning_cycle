"""Controlled retraining for the audio command classifier.

New recordings must be labelled in a CSV with columns: filename,label.
The candidate Linear SVM is trained on the existing training split plus the
new labelled recordings, then evaluated on the unchanged validation split.
The held-out test split is never used for retraining or promotion decisions.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models import build_classical_models, evaluate_classifier, save_model_bundle
from src.preprocessing import extract_mfcc_features


def retrain_from_labeled_audio(
    metadata_path: str | Path,
    audio_dir: str | Path,
    project_root: str | Path,
) -> dict:
    """Train a candidate SVM and promote it only if validation macro-F1 improves."""
    root = Path(project_root).resolve()
    metadata_path = Path(metadata_path)
    audio_dir = Path(audio_dir)
    model_dir = root / "models"
    processed_dir = root / "notebooks" / "data" / "processed"
    feature_path = processed_dir / "mfcc_features.npz"
    metrics_path = model_dir / "final_test_metrics.json"

    for required in (feature_path, metrics_path):
        if not required.is_file():
            raise FileNotFoundError(
                f"Required training artifact is missing: {required}. "
                "Run notebooks 02-05 first."
            )

    metadata = pd.read_csv(metadata_path)
    required_columns = {"filename", "label"}
    if not required_columns.issubset(metadata.columns):
        raise ValueError("Metadata CSV must contain 'filename' and 'label' columns.")
    metadata = metadata[["filename", "label"]].dropna().copy()
    metadata["filename"] = metadata["filename"].astype(str).map(lambda x: Path(x).name)
    metadata["label"] = metadata["label"].astype(str).str.strip().str.lower()
    if metadata.empty:
        raise ValueError("The metadata CSV contains no labelled recordings.")
    if metadata["filename"].duplicated().any():
        raise ValueError("Each audio filename must appear only once in the metadata CSV.")

    with np.load(feature_path, allow_pickle=True) as data:
        X_train = data["X_train"].astype(np.float32)
        y_train = np.asarray(data["y_train"], dtype=str)
        X_validation = data["X_validation"].astype(np.float32)
        y_validation = np.asarray(data["y_validation"], dtype=str)

    known_labels = sorted(np.unique(y_train).tolist())
    unknown_labels = sorted(set(metadata["label"]) - set(known_labels))
    if unknown_labels:
        raise ValueError(
            f"Unknown labels: {unknown_labels}. Use one of: {known_labels}"
        )

    rows = []
    for row in metadata.itertuples(index=False):
        audio_path = audio_dir / row.filename
        if not audio_path.is_file() or audio_path.suffix.lower() != ".wav":
            raise ValueError(f"Missing WAV file for metadata row: {row.filename}")
        rows.append({
            "path": str(audio_path.resolve()),
            "target": row.label,
            "split": "train",
            "offset_seconds": 0.0,
        })

    added_frame = pd.DataFrame(rows)
    X_added, y_added, _ = extract_mfcc_features(
        added_frame, show_progress=False
    )
    X_combined = np.vstack([X_train, X_added]).astype(np.float32)
    y_combined = np.concatenate([y_train, y_added.astype(str)])

    candidate = build_classical_models()["linear_svm"]
    candidate.fit(X_combined, y_combined)
    evaluation = evaluate_classifier(candidate, X_validation, y_validation)
    candidate_f1 = float(evaluation["metrics"]["macro_f1"])

    final_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    validation_rows = final_metrics.get("validation_comparison", [])
    selected_name = final_metrics.get("validation_selected_model", "")
    selected_row = next(
        (item for item in validation_rows if item.get("model") == selected_name), None
    )
    baseline_f1 = (
        float(selected_row["macro_f1"]) if selected_row and "macro_f1" in selected_row
        else -1.0
    )

    promoted = candidate_f1 > baseline_f1
    result = {
        "candidate_model": "linear_svm_retrained",
        "added_recordings": int(len(metadata)),
        "training_samples_before": int(len(X_train)),
        "training_samples_after": int(len(X_combined)),
        "validation_macro_f1": candidate_f1,
        "previous_deployed_model": selected_name,
        "previous_validation_macro_f1": baseline_f1,
        "promoted": promoted,
        "validation_metrics": evaluation["metrics"],
        "note": "Promotion is based on validation data only; the held-out test set was not used.",
    }

    if promoted:
        bundle_path = model_dir / "linear_svm_retrained.joblib"
        save_model_bundle(
            candidate, bundle_path, model_name="linear_svm_retrained",
            feature_names=[f"mfcc_{i}" for i in range(X_combined.shape[1])],
        )
        deployment_path = model_dir / "deployment_model.json"
        deployment_path.write_text(
            json.dumps({
                "model_name": "linear_svm_retrained",
                "model_file": bundle_path.name,
                "validation_macro_f1": candidate_f1,
                "promoted_from": selected_name,
            }, indent=2),
            encoding="utf-8",
        )
        result["model_path"] = str(bundle_path)
        result["deployment_config"] = str(deployment_path)

    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "latest_retraining_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result
