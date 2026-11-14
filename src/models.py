"""Traditional machine-learning models and evaluation helpers for audio commands."""

from __future__ import annotations

from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def build_classical_models(random_state: int = 42) -> dict[str, object]:
    """Create reproducible, imbalance-aware classical baselines."""
    return {
        "linear_svm": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LinearSVC(
                class_weight="balanced",
                random_state=random_state,
                max_iter=5000,
                dual="auto",
            )),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def evaluate_classifier(model, X: np.ndarray, y_true: np.ndarray) -> dict:
    """Return aggregate metrics, per-class report, and confusion matrix."""
    predictions = model.predict(X)
    labels = sorted(np.unique(np.concatenate([y_true, predictions])).tolist())
    return {
        "metrics": {
            "accuracy": float(accuracy_score(y_true, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
            "macro_precision": float(precision_score(
                y_true, predictions, average="macro", zero_division=0
            )),
            "macro_recall": float(recall_score(
                y_true, predictions, average="macro", zero_division=0
            )),
            "macro_f1": float(f1_score(
                y_true, predictions, average="macro", zero_division=0
            )),
            "weighted_f1": float(f1_score(
                y_true, predictions, average="weighted", zero_division=0
            )),
        },
        "report": classification_report(
            y_true, predictions, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=labels),
        "labels": labels,
        "predictions": predictions,
    }


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    random_state: int = 42,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Fit both baselines and compare them on validation data only."""
    models = build_classical_models(random_state=random_state)
    fitted: dict[str, object] = {}
    rows = []

    for name, model in models.items():
        started = time.perf_counter()
        model.fit(X_train, y_train)
        training_seconds = time.perf_counter() - started

        prediction_started = time.perf_counter()
        result = evaluate_classifier(model, X_validation, y_validation)
        prediction_seconds = time.perf_counter() - prediction_started

        fitted[name] = model
        rows.append({
            "model": name,
            **result["metrics"],
            "training_seconds": training_seconds,
            "validation_prediction_seconds": prediction_seconds,
        })

    comparison = pd.DataFrame(rows).sort_values(
        ["macro_f1", "balanced_accuracy"], ascending=False
    ).reset_index(drop=True)
    return fitted, comparison


def save_model_bundle(model, path: str | Path, *, model_name: str, feature_names=None) -> Path:
    """Persist a fitted estimator with metadata needed for inference."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "model_name": model_name,
        "feature_names": list(feature_names) if feature_names is not None else None,
    }, path)
    return path


def save_metrics(comparison: pd.DataFrame, path: str | Path) -> Path:
    """Write model comparison metrics to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(comparison.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    return path
