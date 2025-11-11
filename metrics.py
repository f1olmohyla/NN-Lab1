import numpy as np


def compute_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_width = max(0, inter_xmax - inter_xmin)
    inter_height = max(0, inter_ymax - inter_ymin)
    inter_area = inter_width * inter_height

    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def compute_precision_recall(pred_boxes, pred_scores, pred_labels,
                             gt_boxes, gt_labels, iou_threshold=0.5):
    if len(pred_boxes) == 0:
        return 0.0, 0.0

    if len(gt_boxes) == 0:
        return 0.0, 1.0 if len(pred_boxes) == 0 else 0.0

    sorted_indices = np.argsort(pred_scores)[::-1]
    pred_boxes = [pred_boxes[i] for i in sorted_indices]
    pred_labels = [pred_labels[i] for i in sorted_indices]

    gt_matched = [False] * len(gt_boxes)
    true_positives = 0

    for pred_box, pred_label in zip(pred_boxes, pred_labels):
        best_iou = 0
        best_gt_idx = -1

        for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            if gt_matched[gt_idx]:
                continue

            if pred_label != gt_label:
                continue

            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx != -1:
            gt_matched[best_gt_idx] = True
            true_positives += 1

    precision = true_positives / len(pred_boxes) if len(pred_boxes) > 0 else 0.0
    recall = true_positives / len(gt_boxes) if len(gt_boxes) > 0 else 0.0

    return precision, recall


def compute_map(all_pred_boxes, all_pred_scores, all_pred_labels,
                all_gt_boxes, all_gt_labels, iou_thresholds=[0.5]):
    all_classes = set()
    for labels in all_gt_labels:
        all_classes.update(labels)
    for labels in all_pred_labels:
        all_classes.update(labels)

    if len(all_classes) == 0:
        return 0.0

    all_classes = sorted(list(all_classes))
    aps_per_threshold = []

    for iou_threshold in iou_thresholds:
        aps = []

        for cls in all_classes:
            cls_pred_boxes = []
            cls_pred_scores = []
            cls_gt_boxes = []

            for img_idx in range(len(all_pred_boxes)):
                pred_boxes = all_pred_boxes[img_idx]
                pred_scores = all_pred_scores[img_idx]
                pred_labels = all_pred_labels[img_idx]
                gt_boxes = all_gt_boxes[img_idx]
                gt_labels = all_gt_labels[img_idx]

                for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                    if label == cls:
                        cls_pred_boxes.append((img_idx, box, score))

                for box, label in zip(gt_boxes, gt_labels):
                    if label == cls:
                        cls_gt_boxes.append((img_idx, box))

            if len(cls_gt_boxes) == 0:
                continue

            cls_pred_boxes = sorted(cls_pred_boxes, key=lambda x: x[2], reverse=True)
            gt_matched = [False] * len(cls_gt_boxes)
            tp = []
            fp = []

            for pred_img_idx, pred_box, pred_score in cls_pred_boxes:
                best_iou = 0
                best_gt_idx = -1

                for gt_idx, (gt_img_idx, gt_box) in enumerate(cls_gt_boxes):
                    if gt_matched[gt_idx]:
                        continue

                    if pred_img_idx != gt_img_idx:
                        continue

                    iou = compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= iou_threshold and best_gt_idx != -1:
                    gt_matched[best_gt_idx] = True
                    tp.append(1)
                    fp.append(0)
                else:
                    tp.append(0)
                    fp.append(1)

            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)

            recalls = tp_cumsum / len(cls_gt_boxes)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

            recalls = np.concatenate(([0.0], recalls, [1.0]))
            precisions = np.concatenate(([0.0], precisions, [0.0]))

            for i in range(len(precisions) - 2, -1, -1):
                precisions[i] = max(precisions[i], precisions[i + 1])

            indices = np.where(recalls[1:] != recalls[:-1])[0]
            ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
            aps.append(ap)

        if len(aps) > 0:
            aps_per_threshold.append(np.mean(aps))
        else:
            aps_per_threshold.append(0.0)

    return np.mean(aps_per_threshold)
