import torch
import torch.nn as nn
import torchvision.models as models


class AlexNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        alexnet = models.alexnet(pretrained=pretrained)
        self.features = alexnet.features
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x


class AlexNetDetector(nn.Module):
    def __init__(self, num_classes=95, num_boxes=10):
        super().__init__()
        self.backbone = AlexNetFeatureExtractor(pretrained=True)

        feature_dim = 256 * 6 * 6

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(feature_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_boxes * num_classes)
        )

        self.bbox_regressor = nn.Sequential(
            nn.Dropout(),
            nn.Linear(feature_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_boxes * 4)
        )

        self.num_classes = num_classes
        self.num_boxes = num_boxes

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)

        class_logits = self.classifier(features)
        class_logits = class_logits.view(-1, self.num_boxes, self.num_classes)

        bbox_pred = self.bbox_regressor(features)
        bbox_pred = bbox_pred.view(-1, self.num_boxes, 4)

        return class_logits, bbox_pred
