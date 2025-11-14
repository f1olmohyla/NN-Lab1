"""You Only Look Once (YOLO v1) detector architecture implemented in PyTorch.

This module mirrors the configuration presented by Redmon et al. (2016) for the original
YOLO model, producing predictions on an S x S grid with B bounding boxes per cell.
"""

from __future__ import annotations

from typing import Dict

from torch import Tensor, nn

__all__ = ["YOLOv1"]


def _conv(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int | None = None) -> nn.Sequential:
    if padding is None:
        padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.LeakyReLU(0.1, inplace=True),
    )


class YOLOv1(nn.Module):
    """PyTorch implementation of the original YOLO detection network."""

    def __init__(self, num_classes: int, grid_size: int = 7, num_bboxes: int = 2) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be a positive integer.")
        if grid_size <= 0:
            raise ValueError("grid_size must be a positive integer.")
        if num_bboxes <= 0:
            raise ValueError("num_bboxes must be a positive integer.")

        layers = [
            _conv(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            _conv(64, 192, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            _conv(192, 128, kernel_size=1, padding=0),
            _conv(128, 256, kernel_size=3),
            _conv(256, 256, kernel_size=1, padding=0),
            _conv(256, 512, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]

        for _ in range(4):
            layers.extend((_conv(512, 256, kernel_size=1, padding=0), _conv(256, 512, kernel_size=3)))
        layers.extend((_conv(512, 512, kernel_size=1, padding=0), _conv(512, 1024, kernel_size=3)))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        for _ in range(2):
            layers.extend((_conv(1024, 512, kernel_size=1, padding=0), _conv(512, 1024, kernel_size=3)))
        layers.append(_conv(1024, 1024, kernel_size=3))
        layers.append(_conv(1024, 1024, kernel_size=3, stride=2, padding=1))
        layers.append(_conv(1024, 1024, kernel_size=3))
        layers.append(_conv(1024, 1024, kernel_size=3))

        self.features = nn.Sequential(*layers)
        self.spatial_pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))

        prediction_dim = grid_size * grid_size * (num_classes + num_bboxes * 5)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * grid_size * grid_size, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, prediction_dim),
        )

        self.num_classes = num_classes
        self.grid_size = grid_size
        self.num_bboxes = num_bboxes

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.1, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.features(x)
        x = self.spatial_pool(x)
        x = self.classifier(x)
        predictions = x.view(-1, self.grid_size, self.grid_size, self.num_classes + self.num_bboxes * 5)
        return {"predictions": predictions}
