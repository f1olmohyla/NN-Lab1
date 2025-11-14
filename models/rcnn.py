"""Region-based Convolutional Neural Network (R-CNN) implementation in PyTorch.

The architecture follows the formulation presented by Girshick et al. (2014), where
external region proposals are cropped, resized, and then forwarded through a high-capacity
CNN (AlexNet here by default) to produce class scores and bounding box regressions.
"""

from __future__ import annotations

from typing import Dict, Optional

from torch import Tensor, nn

from .alexnet_detector import AlexNetFeatureExtractor

__all__ = ["RCNN"]


class RCNN(nn.Module):
    """Region-based CNN detector with a configurable feature extractor."""

    def __init__(
        self,
        num_classes: int,
        feature_extractor: Optional[nn.Module] = None,
        representation_size: Optional[int] = None,
        use_bbox_regression: bool = True,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be a positive integer.")

        self.feature_extractor = feature_extractor or AlexNetFeatureExtractor()

        if representation_size is None:
            if hasattr(self.feature_extractor, "output_dim"):
                representation_size = int(getattr(self.feature_extractor, "output_dim"))
            else:
                raise ValueError(
                    "representation_size must be provided when feature_extractor does not expose output_dim."
                )

        self.num_classes = num_classes
        self.representation_size = representation_size
        self.use_bbox_regression = use_bbox_regression

        self.classifier = nn.Linear(self.representation_size, num_classes)
        self.regressor = nn.Linear(self.representation_size, num_classes * 4) if use_bbox_regression else None

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.classifier.bias, 0.0)
        if self.regressor is not None:
            nn.init.normal_(self.regressor.weight, mean=0.0, std=0.001)
            nn.init.constant_(self.regressor.bias, 0.0)

    def extract_features(self, x: Tensor) -> Tensor:
        """Return the shared representation before the detection heads."""
        return self.feature_extractor(x)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Run the R-CNN head on a batch of cropped proposal images."""
        features = self.extract_features(x)
        scores = self.classifier(features)
        output: Dict[str, Tensor] = {"logits": scores, "features": features}
        if self.regressor is not None:
            bbox_deltas = self.regressor(features).view(x.size(0), self.num_classes, 4)
            output["bbox_deltas"] = bbox_deltas
        return output
