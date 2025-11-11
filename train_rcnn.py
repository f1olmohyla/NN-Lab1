import torch
import torch.optim as optim
from tqdm import tqdm
import os
import json

from dataset import create_dataloaders
from models.rcnn import RCNN
from losses import DetectionLoss
from metrics import compute_map


def train_rcnn():
    with open('hyperparameters.json', 'r') as f:
        config = json.load(f)

    train_config = config['training']
    rcnn_config = config['rcnn']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = create_dataloaders(
        'aircraft_dataset/labels_with_split.csv',
        batch_size=train_config['batch_size'],
        num_workers=train_config['num_workers']
    )

    model = RCNN(
        num_classes=rcnn_config['num_classes'],
        backbone=rcnn_config['backbone']
    )
    model = model.to(device)

    criterion = DetectionLoss(num_classes=rcnn_config['num_classes'])
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_config['learning_rate'],
        weight_decay=train_config['weight_decay']
    )

    os.makedirs('checkpoints', exist_ok=True)

    print("R-CNN training requires multi-stage approach")
    print("Stage 1: Feature extraction fine-tuning")
    print("Stage 2: Classifier training")
    print("Stage 3: Bbox regressor training")


if __name__ == '__main__':
    train_rcnn()
