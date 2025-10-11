#!/usr/bin/env python3

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import random
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import cv2


class ImageNormalizer:
    def __init__(self, target_size: Tuple[int, int] = (640, 640), 
                 normalize_pixels: bool = True):
        self.target_size = target_size
        self.normalize_pixels = normalize_pixels
        self.imagenet_mean = np.array([0.485, 0.456, 0.406])
        self.imagenet_std = np.array([0.229, 0.224, 0.225])
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, self.target_size)
    
    def normalize_pixels(self, image: np.ndarray) -> np.ndarray:
        if self.normalize_pixels:
            image = image.astype(np.float32) / 255.0
            image = (image - self.imagenet_mean) / self.imagenet_std
        return image
    
    def process(self, image: np.ndarray) -> np.ndarray:
        image = self.resize_image(image)
        image = self.normalize_pixels(image)
        return image


class DataAugmenter:
    def __init__(self, flip_prob: float = 0.5, rotation_range: int = 15,
                 brightness_range: Tuple[float, float] = (0.8, 1.2),
                 contrast_range: Tuple[float, float] = (0.8, 1.2)):
        self.flip_prob = flip_prob
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
    
    def horizontal_flip(self, image: np.ndarray, boxes: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if random.random() < self.flip_prob:
            image = cv2.flip(image, 1)
            if boxes is not None:
                h, w = image.shape[:2]
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        return image, boxes
    
    def rotate(self, image: np.ndarray) -> np.ndarray:
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h))
    
    def adjust_brightness(self, image: np.ndarray) -> np.ndarray:
        pil_image = Image.fromarray(image)
        factor = random.uniform(*self.brightness_range)
        enhancer = ImageEnhance.Brightness(pil_image)
        return np.array(enhancer.enhance(factor))
    
    def adjust_contrast(self, image: np.ndarray) -> np.ndarray:
        pil_image = Image.fromarray(image)
        factor = random.uniform(*self.contrast_range)
        enhancer = ImageEnhance.Contrast(pil_image)
        return np.array(enhancer.enhance(factor))
    
    def augment(self, image: np.ndarray, boxes: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        image, boxes = self.horizontal_flip(image, boxes)
        image = self.rotate(image)
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
        weights = {}
        for class_name, count in self.class_counts.items():
            weights[class_name] = self.max_count / count
        return weights
    
    def get_sample_weights(self) -> np.ndarray:
        return np.array([self.weights[class_name] for class_name in self.df['class']])
    
    def oversample_minority_classes(self, threshold: int = 500) -> pd.DataFrame:
        balanced_dfs = []
        
        for class_name, group in self.df.groupby('class'):
            count = len(group)
            if count < threshold:
                repeat_factor = threshold // count
                remainder = threshold % count
                
                repeated = pd.concat([group] * repeat_factor, ignore_index=True)
                if remainder > 0:
                    sampled = group.sample(n=remainder, replace=True)
                    repeated = pd.concat([repeated, sampled], ignore_index=True)
                balanced_dfs.append(repeated)
            else:
                balanced_dfs.append(group)
        
        return pd.concat(balanced_dfs, ignore_index=True)


class BoundingBoxValidator:
    def __init__(self):
        pass
    
    def validate_coordinates(self, boxes: np.ndarray) -> np.ndarray:
        boxes[:, 0] = np.maximum(boxes[:, 0], 0)
        boxes[:, 1] = np.maximum(boxes[:, 1], 0)
        
        boxes[:, 2] = np.maximum(boxes[:, 2], boxes[:, 0] + 1)
        boxes[:, 3] = np.maximum(boxes[:, 3], boxes[:, 1] + 1)
        
        return boxes
    
    def clip_to_image(self, boxes: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
        boxes[:, 0] = np.clip(boxes[:, 0], 0, image_width - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, image_height - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, image_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, image_height)
        
        return boxes
    
    def filter_valid_boxes(self, boxes: np.ndarray, min_area: int = 100) -> np.ndarray:
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        valid_mask = areas >= min_area
        return boxes[valid_mask]


class DatasetPreprocessor:
    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        self.normalizer = ImageNormalizer(target_size)
        self.augmenter = DataAugmenter()
        self.bbox_validator = BoundingBoxValidator()
        self.target_size = target_size
    
    def preprocess_image(self, image_path: str, augment: bool = False) -> np.ndarray:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if augment:
            image, _ = self.augmenter.augment(image)
        
        image = self.normalizer.process(image)
        return image
    
    def preprocess_boxes(self, boxes: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
        orig_h, orig_w = original_size
        target_h, target_w = self.target_size
        
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        
        boxes = self.bbox_validator.validate_coordinates(boxes)
        boxes = self.bbox_validator.clip_to_image(boxes, target_w, target_h)
        boxes = self.bbox_validator.filter_valid_boxes(boxes)
        
        return boxes
    
    def process_dataset_row(self, row: pd.Series, image_dir: str, augment: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        image_path = Path(image_dir) / row['filename']
        image = self.preprocess_image(str(image_path), augment)
        
        boxes = np.array([[row['xmin'], row['ymin'], row['xmax'], row['ymax']]])
        original_size = (row['height'], row['width'])
        boxes = self.preprocess_boxes(boxes, original_size)
        
        return image, boxes


def create_train_val_test_splits(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'validation'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    return train_df, val_df, test_df


def calculate_class_distribution(df: pd.DataFrame) -> Dict[str, int]:
    return df['class'].value_counts().to_dict()


def main():
    csv_path = ""
    image_dir = ""
    
    df = pd.read_csv(csv_path)
    print(f"Original dataset size: {len(df)}")
    
    train_df, val_df, test_df = create_train_val_test_splits(df)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
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
    except Exception as e:
        print(f"Error processing sample: {e}")
    
    class_dist = calculate_class_distribution(train_df)
    print(f"Number of classes: {len(class_dist)}")
    print(f"Most common class: {max(class_dist, key=class_dist.get)} ({max(class_dist.values())} samples)")
    print(f"Least common class: {min(class_dist, key=class_dist.get)} ({min(class_dist.values())} samples)")


if __name__ == "__main__":
    main()
