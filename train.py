import torch
import torch.optim as optim
from tqdm import tqdm
import os
import json

from dataset import create_dataloaders
from models.alexnet_detector import AlexNetDetector
from losses import DetectionLoss
from metrics import compute_map


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc="Training")
    for images, targets in pbar:
        images = images.to(device)

        class_logits, bbox_pred = model(images)

        batch_size = len(targets)

        target_classes = torch.full((batch_size, model.num_boxes), -100, dtype=torch.long).to(device)
        target_boxes = torch.zeros(batch_size, model.num_boxes, 4).to(device)

        for i, target in enumerate(targets):
            num_boxes = min(len(target['labels']), model.num_boxes)
            target_classes[i, :num_boxes] = target['labels'][:num_boxes].to(device)
            target_boxes[i, :num_boxes] = target['boxes'][:num_boxes].to(device)

        loss_dict = criterion(class_logits, bbox_pred, target_classes, target_boxes)
        loss = loss_dict['total']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_labels = []
    all_gt_boxes = []
    all_gt_labels = []

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            class_logits, bbox_pred = model(images)

            probs = torch.softmax(class_logits, dim=-1)
            scores, labels = torch.max(probs, dim=-1)

            for i in range(len(targets)):
                pred_boxes_i = bbox_pred[i].cpu().numpy()
                pred_scores_i = scores[i].cpu().numpy()
                pred_labels_i = labels[i].cpu().numpy()

                all_pred_boxes.append(pred_boxes_i)
                all_pred_scores.append(pred_scores_i)
                all_pred_labels.append(pred_labels_i)

                gt_boxes_i = targets[i]['boxes'].numpy()
                gt_labels_i = targets[i]['labels'].numpy()

                all_gt_boxes.append(gt_boxes_i)
                all_gt_labels.append(gt_labels_i)

    map_score = compute_map(
        all_pred_boxes, all_pred_scores, all_pred_labels,
        all_gt_boxes, all_gt_labels
    )

    return map_score


def main():
    with open('hyperparameters.json', 'r') as f:
        config = json.load(f)

    train_config = config['training']
    alexnet_config = config['alexnet_detector']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = create_dataloaders(
        'aircraft_dataset/labels_with_split.csv',
        batch_size=train_config['batch_size'],
        num_workers=train_config['num_workers']
    )

    model = AlexNetDetector(
        num_classes=alexnet_config['num_classes'],
        num_boxes=alexnet_config['num_boxes']
    )
    model = model.to(device)

    criterion = DetectionLoss(
        num_classes=alexnet_config['num_classes'],
        lambda_cls=alexnet_config['lambda_cls'],
        lambda_loc=alexnet_config['lambda_loc']
    )
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_config['learning_rate'],
        weight_decay=train_config['weight_decay']
    )

    os.makedirs('checkpoints', exist_ok=True)

    best_map = 0.0
    for epoch in range(train_config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{train_config['num_epochs']}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")

        map_score = evaluate(model, val_loader, device)
        print(f"Validation mAP: {map_score:.4f}")

        if map_score > best_map:
            best_map = map_score
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            print(f"Saved best model with mAP: {best_map:.4f}")


if __name__ == '__main__':
    main()
