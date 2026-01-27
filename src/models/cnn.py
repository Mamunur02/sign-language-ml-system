from __future__ import annotations

import torch
from torch import nn


class SimpleCNN(nn.Module):
    """
    Minimal CNN for image classification.
    Input:  (B, 3, H, W)
    Output: (B, num_classes) logits
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2
        )

        # Adaptive pooling makes this work for any image_size
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)              # (B, 128, 1, 1)
        x = torch.flatten(x, 1)       # (B, 128)
        logits = self.classifier(x)   # (B, num_classes)
        return logits
