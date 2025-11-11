import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
    def __init__(self, num_classes=95):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.num_classes = num_classes

    def forward(self, predictions, targets):
        batch_size, num_boxes, num_classes = predictions.shape
        predictions = predictions.view(batch_size * num_boxes, num_classes)
        targets = targets.view(batch_size * num_boxes)
        return self.ce_loss(predictions, targets)


class LocalizationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, pred_boxes, target_boxes, valid_mask=None):
        loss = self.smooth_l1(pred_boxes, target_boxes)
        if valid_mask is not None:
            loss = loss * valid_mask.unsqueeze(-1)
            return loss.sum() / (valid_mask.sum() + 1e-6)
        return loss.mean()


class DetectionLoss(nn.Module):
    def __init__(self, num_classes=95, lambda_cls=1.0, lambda_loc=1.0):
        super().__init__()
        self.cls_loss = ClassificationLoss(num_classes)
        self.loc_loss = LocalizationLoss()
        self.lambda_cls = lambda_cls
        self.lambda_loc = lambda_loc

    def forward(self, pred_classes, pred_boxes, target_classes, target_boxes):
        cls_loss = self.cls_loss(pred_classes, target_classes)

        valid_mask = (target_classes != -100).float()
        loc_loss = self.loc_loss(pred_boxes, target_boxes, valid_mask)

        total_loss = self.lambda_cls * cls_loss + self.lambda_loc * loc_loss

        return {
            'total': total_loss,
            'cls': cls_loss,
            'loc': loc_loss
        }
