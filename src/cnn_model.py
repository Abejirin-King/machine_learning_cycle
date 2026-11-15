"""Small CNN architecture shared by training and lightweight inference."""

from __future__ import annotations

import torch
from torch import nn


class SmallAudioCNN(nn.Module):
    """Compact 2D CNN for one-second log-Mel spectrograms."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.35),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.25), nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
