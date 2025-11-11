import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def visualize_predictions(image, pred_boxes, pred_labels, pred_scores,
                         gt_boxes=None, gt_labels=None, class_names=None,
                         score_threshold=0.5):
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    image = np.clip(image, 0, 1)

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        if score < score_threshold:
            continue

        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin

        rect = patches.Rectangle(
            (xmin, ymin), width, height,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)

        label_text = f"{class_names[label]}: {score:.2f}" if class_names else f"{label}: {score:.2f}"
        ax.text(xmin, ymin - 5, label_text, color='red', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    if gt_boxes is not None:
        for box, label in zip(gt_boxes, gt_labels):
            xmin, ymin, xmax, ymax = box
            width = xmax - xmin
            height = ymax - ymin

            rect = patches.Rectangle(
                (xmin, ymin), width, height,
                linewidth=2, edgecolor='green', facecolor='none'
            )
            ax.add_patch(rect)

    plt.axis('off')
    plt.tight_layout()
    plt.show()


def save_checkpoint(model, optimizer, epoch, map_score, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'map_score': map_score
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(model, optimizer, filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['map_score']
