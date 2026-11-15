"""PyTorch CNN utilities for Speech Commands audio classification."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchaudio


class SpeechCommandsMelDataset(Dataset):
    """Load one-second audio and convert it to log-Mel spectrograms."""

    def __init__(self, frame: pd.DataFrame, label_to_id: dict[str, int],
                 sample_rate: int = 16_000, n_mels: int = 64,
                 n_fft: int = 400, hop_length: int = 160):
        self.frame = frame.reset_index(drop=True).copy()
        self.label_to_id = label_to_id
        self.sample_rate = sample_rate
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, power=2.0
        )

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        offset = float(row.get("offset_seconds", 0.0))
        audio, sr = sf.read(str(row["path"]), start=int(offset * self.sample_rate),
                            frames=self.sample_rate, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sr != self.sample_rate:
            waveform = torch.from_numpy(audio.copy()).unsqueeze(0)
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            audio = waveform.squeeze(0).numpy()
        target_length = self.sample_rate
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
        waveform = torch.from_numpy(audio.copy()).unsqueeze(0)
        mel = self.mel(waveform).clamp_min(1e-9).log()
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        label = self.label_to_id[str(row["target"])]
        return mel, torch.tensor(label, dtype=torch.long)


from src.cnn_model import SmallAudioCNN
def make_loaders(index: pd.DataFrame, batch_size: int = 64,
                 sample_rate: int = 16_000, n_mels: int = 64):
    """Create train/validation loaders and a stable label mapping."""
    labels = sorted(index["target"].astype(str).unique().tolist())
    label_to_id = {label: i for i, label in enumerate(labels)}
    train_frame = index[index["split"].astype(str) == "train"].copy()
    val_frame = index[index["split"].astype(str) == "validation"].copy()
    train_ds = SpeechCommandsMelDataset(train_frame, label_to_id, sample_rate, n_mels)
    val_ds = SpeechCommandsMelDataset(val_frame, label_to_id, sample_rate, n_mels)

    counts = train_frame["target"].astype(str).value_counts()
    sample_weights = train_frame["target"].astype(str).map(
        lambda label: 1.0 / float(counts[label])
    ).to_numpy(dtype=np.float64)
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights), replacement=True
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)
    return train_loader, val_loader, labels


def train_cnn(model, train_loader, val_loader, *, epochs: int = 8,
              learning_rate: float = 1e-3, device: str = "cpu"):
    """Train with weighted sampling and retain the best validation macro-F1."""
    from sklearn.metrics import f1_score, accuracy_score

    device = torch.device(device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    history = []
    best_f1 = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * targets.size(0)
            train_correct += (logits.argmax(1) == targets).sum().item()
            train_total += targets.size(0)

        model.eval()
        val_loss, y_true, y_pred = 0.0, [], []
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                logits = model(features)
                loss = criterion(logits, targets)
                val_loss += loss.item() * targets.size(0)
                y_true.extend(targets.cpu().numpy().tolist())
                y_pred.extend(logits.argmax(1).cpu().numpy().tolist())

        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(train_total, 1),
            "train_accuracy": train_correct / max(train_total, 1),
            "validation_loss": val_loss / max(len(y_true), 1),
            "validation_accuracy": accuracy_score(y_true, y_pred),
            "validation_macro_f1": macro_f1,
            "epoch_seconds": time.perf_counter() - started,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss {row['train_loss']:.4f} | "
            f"val loss {row['validation_loss']:.4f} | "
            f"val accuracy {row['validation_accuracy']:.4f} | "
            f"val macro-F1 {macro_f1:.4f} | "
            f"{row['epoch_seconds']:.1f}s"
        )
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def evaluate_cnn(model, loader, labels, device: str = "cpu"):
    """Return validation metrics, report, confusion matrix and predictions."""
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
        classification_report, confusion_matrix, f1_score, precision_score,
        recall_score)

    model.eval()
    y_true, y_pred = [], []
    device = torch.device(device)
    with torch.no_grad():
        for features, targets in loader:
            logits = model(features.to(device))
            y_true.extend(targets.numpy().tolist())
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "report": classification_report(y_true, y_pred, labels=list(range(len(labels))),
                                        target_names=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(labels)))),
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred),
    }


def save_cnn_bundle(model, labels, path: str | Path, *, n_mels: int = 64):
    """Save model weights and inference metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "labels": list(labels),
        "n_mels": n_mels,
        "sample_rate": 16_000,
        "model_name": "small_audio_cnn",
    }, path)
    return path
