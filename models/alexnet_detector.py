"""AlexNet-based detector heads implemented in PyTorch.

This module follows the architecture described by Krizhevsky et al. (2012) and adapts it
for object detection tasks by adding parallel classification and bounding-box regression
heads on top of the shared fully connected representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from torch import Tensor, nn


__all__ = ["AlexNetFeatureExtractor", "AlexNetDetector", "AlexNetDetectorOutput"]


def _conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: Optional[int] = None,
    groups: int = 1,
) -> nn.Sequential:
    """Utility conv -> ReLU block with AlexNet defaults."""
    if padding is None:
        padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups),
        nn.ReLU(inplace=True),
    )


class AlexNetFeatureExtractor(nn.Module):
    """Backbone portion of AlexNet tailored for detection pipelines."""

    def __init__(self, in_channels: int = 3, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            _conv_block(256, 384, kernel_size=3),
            _conv_block(384, 384, kernel_size=3, groups=2),
            _conv_block(384, 256, kernel_size=3, groups=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.flatten = nn.Flatten()
        self.fc6 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
        )
        self.fc7 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )

        self.output_dim = 4096

    def forward(self, x: Tensor) -> Tensor:
        """Return the shared representation used for detection heads."""
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc6(x)
        x = self.fc7(x)
        return x


@dataclass
class AlexNetDetectorOutput:
    """Convenience structure returned by :class:`AlexNetDetector`."""

    logits: Tensor
    bbox_deltas: Tensor
    features: Tensor

    def to_dict(self) -> Dict[str, Tensor]:
        return {"logits": self.logits, "bbox_deltas": self.bbox_deltas, "features": self.features}


class AlexNetDetector(nn.Module):
    """AlexNet-based dense detector head.

    Parameters
    ----------
    num_classes:
        Number of target classes (including background if required).
    dropout:
        Dropout probability applied to the fully connected layers.
    bbox_per_class:
        Whether to learn class-specific box regressors (default) or class-agnostic (if False).
    """

    def __init__(self, num_classes: int, dropout: float = 0.5, bbox_per_class: bool = True) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be a positive integer.")

        self.feature_extractor = AlexNetFeatureExtractor(dropout=dropout)
        self.representation_size = self.feature_extractor.output_dim
        self.num_classes = num_classes
        self.bbox_per_class = bbox_per_class

        self.classifier = nn.Linear(self.representation_size, num_classes)
        bbox_dim = num_classes * 4 if bbox_per_class else 4
        self.box_regressor = nn.Linear(self.representation_size, bbox_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.classifier.bias, 0.0)
        nn.init.normal_(self.box_regressor.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.box_regressor.bias, 0.0)

    def forward(self, x: Tensor) -> AlexNetDetectorOutput:
        """Forward pass producing classification logits and bounding-box deltas."""
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        bbox = self.box_regressor(features)
        if self.bbox_per_class:
            bbox = bbox.view(bbox.size(0), self.num_classes, 4)
        return AlexNetDetectorOutput(logits=logits, bbox_deltas=bbox, features=features)

    def predict(self, x: Tensor) -> Dict[str, Tensor]:
        """Return dictionary outputs for ease of integration with detection pipelines."""
        output = self.forward(x)
        return output.to_dict()
