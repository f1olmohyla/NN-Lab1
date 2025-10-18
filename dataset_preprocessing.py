#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import random
from typing import Tuple, List, Dict, Optional, Sequence
from pathlib import Path
import cv2


# -----------------------------
# Image normalization utilities
# -----------------------------
class ImageNormalizer:
    """
    target_size is (width, height) to match cv2.resize.
    """
    def __init__(self, target_size: Tuple[int, int] = (640, 640), do_normalize: bool = True):
        self.target_size = (int(target_size[0]), int(target_size[1]))  # (W, H)
        self.do_normalize = do_normalize
        # RGB ImageNet stats
        self.imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.imagenet_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        # cv2 expects (width, height)
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)

    def apply_normalization(self, image: np.ndarray) -> np.ndarray:
        # expects uint8 RGB in [0,255]
        image = image.astype(np.float32) / 255.0
        return (image - self.imagenet_mean) / self.imagenet_std

    def process(self, image: np.ndarray) -> np.ndarray:
        image = self.resize_image(image)
        if self.do_normalize:
            image = self.apply_normalization(image)
        return image


# -----------------------------
# Data augmentation (box-aware)
# -----------------------------
class DataAugmenter:
    def __init__(self, flip_prob: float = 0.5, rotation_range: int = 15,
                 brightness_range: Tuple[float, float] = (0.8, 1.2),
                 contrast_range: Tuple[float, float] = (0.8, 1.2)):
        self.flip_prob = flip_prob
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def horizontal_flip(self, image: np.ndarray, boxes: Optional[np.ndarray] = None
                        ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if random.random() < self.flip_prob:
            h, w = image.shape[:2]
            image = cv2.flip(image, 1)
            if boxes is not None and len(boxes) > 0:
                # Using [xmin, ymin, xmax, ymax] with xmax exclusive
                # new_xmin = w - old_xmax; new_xmax = w - old_xmin
                boxes = boxes.copy().astype(np.float32)
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        return image, boxes

    def rotate(self, image: np.ndarray, boxes: Optional[np.ndarray] = None
               ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Random in-plane rotation by angle in [-rotation_range, rotation_range].
        Boxes are transformed by rotating their 4 corners and taking the AABB.
        """
        angle = float(random.uniform(-self.rotation_range, self.rotation_range))
        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated_img = cv2.warpAffine(
            image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )

        if boxes is None or len(boxes) == 0:
            return rotated_img, boxes

        # Build 4 corners per box: (xmin,ymin), (xmax,ymin), (xmax,ymax), (xmin,ymax)
        boxes = boxes.astype(np.float32)
        corners = np.stack([
            boxes[:, [0, 1]],
            boxes[:, [2, 1]],
            boxes[:, [2, 3]],
            boxes[:, [0, 3]],
        ], axis=1)  # (N, 4, 2)

        # Convert to homogeneous coords and apply affine
        N = corners.shape[0]
        corners_flat = corners.reshape(-1, 2)  # (N*4, 2)
        ones = np.ones((corners_flat.shape[0], 1), dtype=np.float32)
        corners_h = np.hstack([corners_flat, ones])  # (N*4, 3)
        rot_flat = corners_h @ M.T  # (N*4, 2)
        rot_corners = rot_flat.reshape(N, 4, 2)

        x_min = rot_corners[..., 0].min(axis=1)
        y_min = rot_corners[..., 1].min(axis=1)
        x_max = rot_corners[..., 0].max(axis=1)
        y_max = rot_corners[..., 1].max(axis=1)

        new_boxes = np.stack([x_min, y_min, x_max, y_max], axis=1).astype(np.float32)
        return rotated_img, new_boxes

    def adjust_brightness(self, image: np.ndarray) -> np.ndarray:
        pil_image = Image.fromarray(image)  # expects RGB uint8
        factor = float(random.uniform(*self.brightness_range))
        enhancer = ImageEnhance.Brightness(pil_image)
        return np.array(enhancer.enhance(factor))

    def adjust_contrast(self, image: np.ndarray) -> np.ndarray:
        pil_image = Image.fromarray(image)  # expects RGB uint8
        factor = float(random.uniform(*self.contrast_range))
        enhancer = ImageEnhance.Contrast(pil_image)
        return np.array(enhancer.enhance(factor))

    def augment(self, image: np.ndarray, boxes: Optional[np.ndarray] = None
                ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        image, boxes = self.horizontal_flip(image, boxes)
        image, boxes = self.rotate(image, boxes)
        image = self.adjust_brightness(image)
        image = self.adjust_contrast(image)
        return image, boxes


# -----------------------------
# Class balancing utilities
# -----------------------------
class ClassBalancer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.class_counts = df['class'].value_counts()
        self.max_count = self.class_counts.max()
        self.weights = self._calculate_weights()

    def _calculate_weights(self) -> Dict[str, float]:
        return {cls: self.max_count / cnt for cls, cnt in self.class_counts.items()}

    def get_sample_weights(self) -> np.ndarray:
        return np.array([self.weights[c] for c in self.df['class']], dtype=np.float32)

    def oversample_minority_classes(self, threshold: int = 500) -> pd.DataFrame:
        balanced_dfs = []
        for class_name, group in self.df.groupby('class'):
            count = len(group)
            if count < threshold:
                repeat_factor = threshold // count
                remainder = threshold % count
                repeated = pd.concat([group] * repeat_factor, ignore_index=True)
                if remainder > 0:
                    sampled = group.sample(n=remainder, replace=True, random_state=42)
                    repeated = pd.concat([repeated, sampled], ignore_index=True)
                balanced_dfs.append(repeated)
            else:
                balanced_dfs.append(group)
        return pd.concat(balanced_dfs, ignore_index=True)


# -----------------------------
# Box validation / clipping
# -----------------------------
class BoundingBoxValidator:
    """
    Assumes [xmin, ymin, xmax, ymax] with xmax/ymax being exclusive bounds,
    i.e., valid range is 0 <= x < width, 0 <= y < height.
    """
    def validate_coordinates(self, boxes: np.ndarray) -> np.ndarray:
        boxes = boxes.copy().astype(np.float32)
        boxes[:, 0] = np.maximum(boxes[:, 0], 0)  # xmin >= 0
        boxes[:, 1] = np.maximum(boxes[:, 1], 0)  # ymin >= 0
        # Ensure positive width/height
        boxes[:, 2] = np.maximum(boxes[:, 2], boxes[:, 0] + 1)
        boxes[:, 3] = np.maximum(boxes[:, 3], boxes[:, 1] + 1)
        return boxes

    def clip_to_image(self, boxes: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
        boxes = boxes.copy().astype(np.float32)
        # xmin/ymin in [0, w-1] / [0, h-1]
        boxes[:, 0] = np.clip(boxes[:, 0], 0, image_width - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, image_height - 1)
        # xmax/ymax in [0, w] / [0, h] (exclusive upper bound convention)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, image_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, image_height)
        return boxes

    def filter_valid_boxes(self, boxes: np.ndarray, min_area: int = 100) -> np.ndarray:
        w = (boxes[:, 2] - boxes[:, 0])
        h = (boxes[:, 3] - boxes[:, 1])
        areas = w * h
        valid_mask = (w > 0) & (h > 0) & (areas >= min_area)
        return boxes[valid_mask]


# -----------------------------
# Dataset preprocessing
# -----------------------------
class DatasetPreprocessor:
    """
    target_size is (width, height) to match cv2.resize.
    """
    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        self.target_size = (int(target_size[0]), int(target_size[1]))  # (W, H)
        self.normalizer = ImageNormalizer(self.target_size, do_normalize=True)
        self.augmenter = DataAugmenter()
        self.bbox_validator = BoundingBoxValidator()
        self._ext_priority: Sequence[str] = ("jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff")

    def _resolve_image_path(self, image_dir: str, filename: str) -> Path:
        """
        Try, in order:
          1) exact filename (if it has an extension and exists)
          2) <stem>.<ext> for preferred ext list (case variants)
          3) first match <stem>.* as fallback
        """
        base = Path(filename)
        root = Path(image_dir)

        # 1) direct hit with extension
        candidate = root / filename
        if base.suffix and candidate.exists():
            return candidate

        stem = base.stem

        # 2) try preferred extensions with case variants
        for ext in self._ext_priority:
            for variant in (ext, ext.upper(), ext.capitalize()):
                cand = root / f"{stem}.{variant}"
                if cand.exists():
                    return cand

        # 3) fallback: any match
        matches = sorted((p for p in root.glob(f"{stem}.*") if p.is_file()))
        if matches:
            return matches[0]

        raise FileNotFoundError(f"No image found for stem '{stem}' in {root}")

    def process_dataset_row(self, row: pd.Series, image_dir: str, augment: bool = False
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads image (RGB), optionally applies box-aware augmentation,
        resizes to target (W,H), normalizes (ImageNet), scales + validates boxes.
        Returns: (image, boxes) with image shape (H, W, 3) if normalized float32, or same but float32.
        """
        image_path = self._resolve_image_path(image_dir, str(row["filename"]))

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image at {image_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Boxes for this row (single-box rows; extend to multiple as needed)
        boxes = np.array([[row['xmin'], row['ymin'], row['xmax'], row['ymax']]], dtype=np.float32)

        # Augment at original resolution
        if augment:
            image, boxes = self.augmenter.augment(image, boxes)

        # Validate/clip at original resolution
        h, w = image.shape[:2]
        boxes = self.bbox_validator.validate_coordinates(boxes)
        boxes = self.bbox_validator.clip_to_image(boxes, w, h)

        # Resize image; scale boxes to target (W,H)
        target_w, target_h = self.target_size
        scale_x = target_w / float(w)
        scale_y = target_h / float(h)
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        # Filter tiny/invalid after scaling
        boxes = self.bbox_validator.filter_valid_boxes(boxes)

        # Normalize image for model input
        image = self.normalizer.resize_image(image)
        if self.normalizer.do_normalize:
            image = self.normalizer.apply_normalization(image)

        return image, boxes


# -----------------------------
# Split & stats helpers
# -----------------------------
def create_train_val_test_splits(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Accepts either 'validation' or 'val' for the validation split.
    """
    split_col = 'split'
    val_mask = df[split_col].astype(str).str.lower().isin(['validation', 'val'])
    train_df = df[df[split_col] == 'train'].copy()
    val_df = df[val_mask].copy()
    test_df = df[df[split_col] == 'test'].copy()
    return train_df, val_df, test_df


def calculate_class_distribution(df: pd.DataFrame) -> Dict[str, int]:
    return df['class'].value_counts().to_dict()


# -----------------------------
# Main
# -----------------------------
def main():
    csv_path = "aircraft_dataset/labels_with_split.csv"
    image_dir = "aircraft_dataset/dataset"

    df = pd.read_csv(csv_path)
    print(f"Original dataset size: {len(df)}")

    train_df, val_df, test_df = create_train_val_test_splits(df)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    if len(train_df) == 0:
        print("No training samples found; aborting sample processing.")
        return

    balancer = ClassBalancer(train_df)
    sample_weights = balancer.get_sample_weights()
    print(f"Sample weights shape: {sample_weights.shape}")

    balanced_df = balancer.oversample_minority_classes(threshold=200)
    print(f"Balanced dataset size: {len(balanced_df)}")

    preprocessor = DatasetPreprocessor(target_size=(640, 640))

    sample_row = train_df.iloc[0]
    try:
        processed_image, processed_boxes = preprocessor.process_dataset_row(
            sample_row, image_dir, augment=True
        )
        print(f"Processed image shape: {processed_image.shape}")
        print(f"Processed boxes shape: {processed_boxes.shape}")
        if processed_boxes.size:
            print(f"Sample box (scaled): {processed_boxes[0].round(2)}")
    except Exception as e:
        print(f"Error processing sample: {e}")

    class_dist = calculate_class_distribution(train_df)
    if class_dist:
        print(f"Number of classes: {len(class_dist)}")
        most_cls = max(class_dist, key=class_dist.get)
        least_cls = min(class_dist, key=class_dist.get)
        print(f"Most common class: {most_cls} ({class_dist[most_cls]} samples)")
        print(f"Least common class: {least_cls} ({class_dist[least_cls]} samples)")


if __name__ == "__main__":
    main()