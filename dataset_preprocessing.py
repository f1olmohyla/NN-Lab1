#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import random
from typing import Tuple, List, Dict, Optional, Sequence
from pathlib import Path
import cv2

class ImageNormalizer:
    """
    target_size is (width, height) to match cv2.resize.
    """
    def __init__(self, target_size: Tuple[int, int] = (640, 640), do_normalize: bool = True):
        self.target_size = (int(target_size[0]), int(target_size[1]))  # (W, H)
        self.do_normalize = do_normalize

        self.imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.imagenet_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)

    def apply_normalization(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32) / 255.0
        return (image - self.imagenet_mean) / self.imagenet_std

    def process(self, image: np.ndarray) -> np.ndarray:
        image = self.resize_image(image)
        if self.do_normalize:
            image = self.apply_normalization(image)
        return image


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
                boxes = boxes.copy().astype(np.float32)
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        return image, boxes

    def rotate(self, image: np.ndarray, boxes: Optional[np.ndarray] = None
               ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        angle = float(random.uniform(-self.rotation_range, self.rotation_range))
        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated_img = cv2.warpAffine(
            image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )

        if boxes is None or len(boxes) == 0:
            return rotated_img, boxes

        boxes = boxes.astype(np.float32)
        corners = np.stack([
            boxes[:, [0, 1]],
            boxes[:, [2, 1]],
            boxes[:, [2, 3]],
            boxes[:, [0, 3]],
        ], axis=1)  # (N, 4, 2)

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

class BoundingBoxValidator:
    """
    Assumes [xmin, ymin, xmax, ymax] with xmax/ymax being exclusive bounds,
    i.e., valid range is 0 <= x < width, 0 <= y < height.
    """
    def validate_coordinates(self, boxes: np.ndarray) -> np.ndarray:
        boxes = boxes.copy().astype(np.float32)
        boxes[:, 0] = np.maximum(boxes[:, 0], 0)  # xmin >= 0
        boxes[:, 1] = np.maximum(boxes[:, 1], 0)  # ymin >= 0

        boxes[:, 2] = np.maximum(boxes[:, 2], boxes[:, 0] + 1)
        boxes[:, 3] = np.maximum(boxes[:, 3], boxes[:, 1] + 1)
        return boxes

    def clip_to_image(self, boxes: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
        boxes = boxes.copy().astype(np.float32)

        boxes[:, 0] = np.clip(boxes[:, 0], 0, image_width - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, image_height - 1)

        boxes[:, 2] = np.clip(boxes[:, 2], 0, image_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, image_height)
        return boxes

    def filter_valid_boxes(self, boxes: np.ndarray, min_area: int = 100) -> np.ndarray:
        w = (boxes[:, 2] - boxes[:, 0])
        h = (boxes[:, 3] - boxes[:, 1])
        areas = w * h
        valid_mask = (w > 0) & (h > 0) & (areas >= min_area)
        return boxes[valid_mask]

class DatasetPreprocessor:
    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        self.target_size = (int(target_size[0]), int(target_size[1]))  # (W, H)
        self.normalizer = ImageNormalizer(self.target_size, do_normalize=True)
        self.augmenter = DataAugmenter()
        self.bbox_validator = BoundingBoxValidator()
        self._ext_priority: Sequence[str] = ("jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff")

    def _resolve_image_path(self, image_dir: str, filename: str) -> Path:
        base = Path(filename)
        root = Path(image_dir)

        candidate = root / filename
        if base.suffix and candidate.exists():
            return candidate

        stem = base.stem

        for ext in self._ext_priority:
            for variant in (ext, ext.upper(), ext.capitalize()):
                cand = root / f"{stem}.{variant}"
                if cand.exists():
                    return cand

        matches = sorted((p for p in root.glob(f"{stem}.*") if p.is_file()))
        if matches:
            return matches[0]

        raise FileNotFoundError(f"No image found for stem '{stem}' in {root}")

    def process_dataset_row(self, row: pd.Series, image_dir: str, augment: bool = False
                            ) -> Tuple[np.ndarray, np.ndarray]:
        image_path = self._resolve_image_path(image_dir, str(row["filename"]))

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image at {image_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        boxes = np.array([[row['xmin'], row['ymin'], row['xmax'], row['ymax']]], dtype=np.float32)

        if augment:
            image, boxes = self.augmenter.augment(image, boxes)

        h, w = image.shape[:2]
        boxes = self.bbox_validator.validate_coordinates(boxes)
        boxes = self.bbox_validator.clip_to_image(boxes, w, h)

        target_w, target_h = self.target_size
        scale_x = target_w / float(w)
        scale_y = target_h / float(h)
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        boxes = self.bbox_validator.filter_valid_boxes(boxes)

        image = self.normalizer.resize_image(image)
        if self.normalizer.do_normalize:
            image = self.normalizer.apply_normalization(image)

        return image, boxes


def create_train_val_test_splits(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_col = 'split'
    val_mask = df[split_col].astype(str).str.lower().isin(['validation', 'val'])
    train_df = df[df[split_col] == 'train'].copy()
    val_df = df[val_mask].copy()
    test_df = df[df[split_col] == 'test'].copy()
    return train_df, val_df, test_df


def calculate_class_distribution(df: pd.DataFrame) -> Dict[str, int]:
    return df['class'].value_counts().to_dict()


def _denorm_to_uint8_rgb(img: np.ndarray, normalizer: ImageNormalizer) -> np.ndarray:
    """
    Convert a possibly ImageNet-normalized float32 RGB image to uint8 RGB for display.
    """
    out = img
    if out.dtype != np.uint8:
        if normalizer.do_normalize:
            out = (out * normalizer.imagenet_std) + normalizer.imagenet_mean
        out = np.clip(out, 0.0, 1.0)
        out = (out * 255.0).round().astype(np.uint8)
    return out


def _draw_boxes_bgr(img_bgr: np.ndarray, boxes: Optional[np.ndarray], color=(0, 255, 0), thickness: int = 2) -> np.ndarray:
    if boxes is not None and boxes.size:
        for (x1, y1, x2, y2) in boxes.astype(int):
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)
    return img_bgr


def show_processed_samples(
    preprocessor: DatasetPreprocessor,
    df: pd.DataFrame,
    image_dir: str,
    num: int = 4,
    augment: bool = True,
    delay_ms: int = 0,
    save_dir: Optional[str] = None,
    show_original: bool = True,
    seed: Optional[int] = None,         
    unique_by: Optional[str] = 'filename'
) -> None:
    if len(df) == 0:
        print("show_processed_samples: dataframe is empty.")
        return

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    pool = df.drop_duplicates(unique_by) if unique_by else df

    rows = pool.sample(n=min(num, len(pool)), random_state=None)

    idx = 0
    for _, row in rows.iterrows():
        idx += 1

        orig_bgr = None
        orig_boxes_clipped = None
        if show_original:
            image_path = preprocessor._resolve_image_path(image_dir, str(row["filename"]))
            orig_bgr = cv2.imread(str(image_path))
            if orig_bgr is None:
                print(f"Could not read original image: {image_path}")
            else:
                h0, w0 = orig_bgr.shape[:2]
                # Original boxes from row; clip for safety
                orig_boxes = np.array([[row['xmin'], row['ymin'], row['xmax'], row['ymax']]], dtype=np.float32)
                orig_boxes = preprocessor.bbox_validator.validate_coordinates(orig_boxes)
                orig_boxes_clipped = preprocessor.bbox_validator.clip_to_image(orig_boxes, w0, h0)

        proc_img, proc_boxes = preprocessor.process_dataset_row(row, image_dir, augment=augment)
        vis_rgb = _denorm_to_uint8_rgb(proc_img, preprocessor.normalizer)
        proc_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
        proc_bgr = _draw_boxes_bgr(proc_bgr, proc_boxes, color=(0, 255, 0), thickness=2)

        cls = str(row.get('class', ''))
        proc_title = f"processed_{idx}_{cls}"
        cv2.namedWindow(proc_title, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(proc_title, proc_bgr)

        if show_original and orig_bgr is not None:
            orig_bgr_draw = orig_bgr.copy()
            orig_bgr_draw = _draw_boxes_bgr(orig_bgr_draw, orig_boxes_clipped, color=(255, 0, 0), thickness=2)
            orig_title = f"original_{idx}_{cls}"
            cv2.namedWindow(orig_title, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(orig_title, orig_bgr_draw)

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path(save_dir) / f"{proc_title}.jpg"), proc_bgr)
            if show_original and orig_bgr is not None:
                cv2.imwrite(str(Path(save_dir) / f"original_{idx}_{cls}.jpg"), orig_bgr_draw)

        key = cv2.waitKey(delay_ms if delay_ms > 0 else 0)
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()


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

    try:
        show_processed_samples(
            preprocessor,
            train_df,
            image_dir,
            num=2,          # how many images to preview
            augment=True,   # show with augmentation applied
            delay_ms=0,     # 0 = wait for keypress per image
            save_dir=None   # e.g., "debug_previews" to also save .jpgs
        )
    except Exception as e:
        print(f"Preview error: {e}")


if __name__ == "__main__":
    main()