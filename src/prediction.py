"""Single-file and batch inference for the trained Speech Commands models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio

from src.deep_learning import SmallAudioCNN
from src.preprocessing import load_audio_segment


class AudioCommandPredictor:
    """Load the validation-selected model and predict one or many WAV files.

    Confidence is a model score, not a calibrated probability. For LinearSVC,
    scores are converted to a relative softmax distribution for ranking only.
    """

    def __init__(self, project_root: str | Path, model_name: str | None = None,
                 device: str | None = None):
        self.project_root = Path(project_root).resolve()
        self.model_dir = self.project_root / "models"
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        selection_path = self.model_dir / "final_test_metrics.json"
        if not selection_path.is_file():
            raise FileNotFoundError(
                f"Missing {selection_path}. Run notebook 05 before inference."
            )
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        deployment_path = self.model_dir / "deployment_model.json"
        deployment = (
            json.loads(deployment_path.read_text(encoding="utf-8"))
            if deployment_path.is_file() else {}
        )
        selected = str(
            deployment.get("model_name", selection["validation_selected_model"])
        )
        self.model_name = model_name or selected
        self.deployment_model_file = (
            deployment.get("model_file") if self.model_name == selected else None
        )
        self.labels: list[str]
        self.model = None

        if self.model_name in {"linear_svm", "random_forest", "linear_svm_retrained"}:
            bundle_name = self.deployment_model_file or f"{self.model_name}.joblib"
            bundle_path = self.model_dir / bundle_name
            if not bundle_path.is_file():
                raise FileNotFoundError(f"Missing model bundle: {bundle_path}")
            bundle = joblib.load(bundle_path)
            self.model = bundle["model"]
            self.labels = sorted(str(label) for label in self.model.classes_)
        elif self.model_name == "small_audio_cnn":
            checkpoint_path = self.model_dir / "small_audio_cnn.pt"
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Missing CNN checkpoint: {checkpoint_path}")
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            self.labels = [str(label) for label in checkpoint["labels"]]
            self.n_mels = int(checkpoint.get("n_mels", 64))
            self.sample_rate = int(checkpoint.get("sample_rate", 16_000))
            self.model = SmallAudioCNN(n_classes=len(self.labels)).to(self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.eval()
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate, n_fft=400, hop_length=160,
                n_mels=self.n_mels, power=2.0
            )
        else:
            raise ValueError(
                f"Unknown model {self.model_name!r}. Choose linear_svm, "
                "random_forest, or small_audio_cnn."
            )

    def _classical_features(self, audio_path: str | Path) -> np.ndarray:
        audio = load_audio_segment(audio_path, sample_rate=16_000)
        mfcc = librosa.feature.mfcc(
            y=audio, sr=16_000, n_mfcc=40, n_fft=400, hop_length=160
        )
        return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]).astype(
            np.float32
        ).reshape(1, -1)

    def _cnn_input(self, audio_path: str | Path) -> torch.Tensor:
        audio = load_audio_segment(
            audio_path, sample_rate=self.sample_rate, duration_seconds=1.0
        )
        waveform = torch.from_numpy(audio.copy()).unsqueeze(0)
        mel = self.mel_transform(waveform).clamp_min(1e-9).log()
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        return mel.unsqueeze(0).to(self.device)

    def predict_one(self, audio_path: str | Path, top_k: int = 3) -> dict:
        """Predict a single WAV file and return its label and top-k scores."""
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")
        if path.suffix.lower() != ".wav":
            raise ValueError(f"Expected a .wav file, received: {path.name}")
        info = sf.info(str(path))
        if info.frames == 0:
            raise ValueError(f"Audio file is empty: {path.name}")

        if self.model_name == "small_audio_cnn":
            with torch.no_grad():
                logits = self.model(self._cnn_input(path))
                scores = torch.softmax(logits, dim=1)[0].cpu().numpy()
        else:
            features = self._classical_features(path)
            prediction = str(self.model.predict(features)[0])
            if hasattr(self.model, "predict_proba"):
                scores = self.model.predict_proba(features)[0]
            else:
                decision = np.asarray(self.model.decision_function(features))[0]
                decision = decision - np.max(decision)
                exp_scores = np.exp(decision)
                scores = exp_scores / exp_scores.sum()
            # Use the model's native predicted class as the final label.
            predicted_label = prediction

        order = np.argsort(scores)[::-1]
        top_k = max(1, min(int(top_k), len(self.labels)))
        ranked = [
            {"label": self.labels[int(i)], "score": float(scores[int(i)])}
            for i in order[:top_k]
        ]
        label = predicted_label if self.model_name != "small_audio_cnn" else ranked[0]["label"]
        confidence = next(item["score"] for item in ranked if item["label"] == label)
        return {
            "file": path.name,
            "model": self.model_name,
            "predicted_command": label,
            "confidence_score": float(confidence),
            "confidence_note": "Model score; not a calibrated probability.",
            "top_predictions": ranked,
        }

    def predict_files(self, audio_paths, output_csv: str | Path | None = None) -> pd.DataFrame:
        """Predict multiple WAV files and optionally save results as CSV."""
        rows = [self.predict_one(path) for path in audio_paths]
        frame = pd.DataFrame([{
            "file": row["file"],
            "model": row["model"],
            "predicted_command": row["predicted_command"],
            "confidence_score": row["confidence_score"],
        } for row in rows])
        if output_csv is not None:
            output_path = Path(output_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(output_path, index=False)
        return frame
