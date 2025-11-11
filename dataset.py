import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from dataset_preprocessing import ImageNormalizer, DataAugmenter, BoundingBoxValidator


class AircraftDetectionDataset(Dataset):
    def __init__(self, csv_path, split='train', transform=None):
        full_data = pd.read_csv(csv_path)

        self.classes = sorted(full_data['class'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.data = full_data[full_data['split'] == split]
        self.split = split
        self.transform = transform

        self.image_groups = self.data.groupby('filename')
        self.image_files = list(self.image_groups.groups.keys())

        self.normalizer = ImageNormalizer(target_size=(640, 640), do_normalize=True)
        self.augmenter = DataAugmenter()
        self.bbox_validator = BoundingBoxValidator()
        self.target_size = (640, 640)
        self.image_dir = "aircraft_dataset/dataset"
        self.ext_priority = ("jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff")

    def __len__(self):
        return len(self.image_files)

    def _resolve_image_path(self, filename):
        base = Path(filename)
        root = Path(self.image_dir)

        candidate = root / filename
        if base.suffix and candidate.exists():
            return candidate

        stem = base.stem

        for ext in self.ext_priority:
            for variant in (ext, ext.upper(), ext.capitalize()):
                cand = root / f"{stem}.{variant}"
                if cand.exists():
                    return cand

        matches = sorted((p for p in root.glob(f"{stem}.*") if p.is_file()))
        if matches:
            return matches[0]

        raise FileNotFoundError(f"No image found for stem '{stem}' in {root}")

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        group = self.image_groups.get_group(filename)

        image_path = self._resolve_image_path(filename)
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Failed to read image at {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = group[['xmin', 'ymin', 'xmax', 'ymax']].values.astype(np.float32)
        labels = group['class'].map(self.class_to_idx).values

        if self.split == 'train':
            image, boxes = self.augmenter.augment(image, boxes)

        h, w = image.shape[:2]
        boxes = self.bbox_validator.validate_coordinates(boxes)
        boxes = self.bbox_validator.clip_to_image(boxes, w, h)

        target_w, target_h = self.target_size
        scale_x = target_w / float(w)
        scale_y = target_h / float(h)
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        min_area = 100
        w_boxes = boxes[:, 2] - boxes[:, 0]
        h_boxes = boxes[:, 3] - boxes[:, 1]
        areas = w_boxes * h_boxes
        valid_mask = (w_boxes > 0) & (h_boxes > 0) & (areas >= min_area)
        boxes = boxes[valid_mask]
        labels = labels[valid_mask]

        if len(boxes) == 0:
            boxes = np.array([[0, 0, 1, 1]], dtype=np.float32)
            labels = np.array([0])

        image = self.normalizer.resize_image(image)
        image = self.normalizer.apply_normalization(image)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        boxes_tensor = torch.from_numpy(boxes).float()
        labels_tensor = torch.from_numpy(labels).long()

        target = {
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'image_id': torch.tensor([idx])
        }

        return image_tensor, target


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets


def create_dataloaders(csv_path, batch_size=8, num_workers=4):
    train_dataset = AircraftDetectionDataset(csv_path, split='train')
    val_dataset = AircraftDetectionDataset(csv_path, split='validation')
    test_dataset = AircraftDetectionDataset(csv_path, split='test')

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader
