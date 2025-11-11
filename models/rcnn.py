import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.ops as ops
import cv2


def nms(boxes, scores, iou_threshold=0.5):
    keep = ops.nms(boxes, scores, iou_threshold)
    return keep


class SelectiveSearchProposer:
    def __init__(self, max_proposals=2000):
        self.max_proposals = max_proposals
        self.ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    def generate_proposals(self, image):
        self.ss.setBaseImage(image)
        self.ss.switchToSelectiveSearchFast()
        rects = self.ss.process()

        proposals = rects[:self.max_proposals]
        return proposals


class RCNN(nn.Module):
    def __init__(self, num_classes=95, backbone='alexnet'):
        super().__init__()
        self.num_classes = num_classes

        if backbone == 'alexnet':
            alexnet = models.alexnet(pretrained=True)
            self.features = alexnet.features
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            feature_dim = 256 * 6 * 6

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(feature_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes + 1)
        )

        self.bbox_regressor = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4 * num_classes)
        )

        self.proposer = SelectiveSearchProposer()

    def forward(self, x, proposals=None):
        if proposals is None:
            pass

        return None, None

    def extract_features(self, roi):
        x = self.features(roi.unsqueeze(0))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
