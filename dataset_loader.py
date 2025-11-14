"""Utilities for loading the aircraft detection dataset into a pandas DataFrame."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import pandas as pd


AIRCRAFT_DATASET_RELATIVE_PATH = Path("aircraft_dataset") / "dataset"
IMAGE_EXTENSIONS: Iterable[str] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _default_dataset_dir() -> Path:
    """Return the default dataset directory relative to this file."""
    return Path(__file__).resolve().parent / AIRCRAFT_DATASET_RELATIVE_PATH


def _resolve_image_path(stem: str, dataset_dir: Path) -> Optional[Path]:
    """Return the first existing image path that matches the provided stem."""
    for extension in IMAGE_EXTENSIONS:
        candidate = dataset_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def load_aircraft_dataset(
    dataset_dir: Optional[str | Path] = None,
    allowed_classes: Optional[Sequence[str]] = None,
    area_range: Optional[Tuple[float, float]] = None,
    allow_multiple_annotations: bool = True,
) -> pd.DataFrame:
    """Load all annotations in the aircraft dataset directory into a DataFrame.

    Parameters
    ----------
    dataset_dir
        Root directory that contains the paired `.csv` annotation files and image files.
        When omitted, the loader looks for `aircraft_dataset/dataset` relative to this module.
    allowed_classes
        Optional iterable of class names to keep. When provided, any annotations whose
        `class` value is not in this collection are dropped from the resulting DataFrame.
    area_range
        Optional tuple (min_ratio, max_ratio) specifying the inclusive range of bounding-box
        areas relative to the full image (0.0–1.0). Boxes outside this range are removed.
    allow_multiple_annotations
        If False, only images with a single remaining annotation are retained. When True,
        all images are kept regardless of how many annotations they contain.

    Returns
    -------
    pandas.DataFrame
        Columns:
            - filename: image file stem (string)
            - width, height: original image dimensions (int)
            - class: aircraft class label (string)
            - xmin, ymin, xmax, ymax: bounding box coordinates (int)
            - image_path: absolute path to the corresponding image file (string)
            - csv_path: absolute path to the source annotation CSV (string)

        If no CSV annotations are found, an empty DataFrame with the expected column schema is returned.
    """
    dataset_directory = Path(dataset_dir) if dataset_dir is not None else _default_dataset_dir()

    if not dataset_directory.exists():
        raise FileNotFoundError(f"Dataset directory '{dataset_directory}' does not exist.")

    allowed_class_set = None
    if allowed_classes is not None:
        allowed_class_set = {str(cls_name) for cls_name in allowed_classes}

    area_min, area_max = None, None
    if area_range is not None:
        if len(area_range) != 2:
            raise ValueError("area_range must be a tuple of (min_ratio, max_ratio).")
        area_min, area_max = area_range
        if not (0.0 <= area_min <= 1.0 and 0.0 <= area_max <= 1.0 and area_min <= area_max):
            raise ValueError("area_range values must satisfy 0.0 <= min <= max <= 1.0.")

    csv_files = sorted(dataset_directory.glob("*.csv"))

    rows: list[pd.DataFrame] = []
    for csv_path in csv_files:
        annotation_frame = pd.read_csv(csv_path)

        expected_columns = {"filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"}
        if not expected_columns.issubset(annotation_frame.columns):
            missing = ", ".join(sorted(expected_columns - set(annotation_frame.columns)))
            raise ValueError(f"Annotation file '{csv_path}' is missing required columns: {missing}")

        # Normalize field types.
        annotation_frame["filename"] = annotation_frame["filename"].astype(str)
        annotation_frame["class"] = annotation_frame["class"].astype(str)
        numeric_columns = ["width", "height", "xmin", "ymin", "xmax", "ymax"]
        annotation_frame[numeric_columns] = annotation_frame[numeric_columns].apply(pd.to_numeric, errors="coerce")

        if allowed_class_set is not None:
            annotation_frame = annotation_frame[annotation_frame["class"].isin(allowed_class_set)]
            if annotation_frame.empty:
                continue

        if area_min is not None and area_max is not None:
            widths = (annotation_frame["xmax"] - annotation_frame["xmin"]).clip(lower=0)
            heights = (annotation_frame["ymax"] - annotation_frame["ymin"]).clip(lower=0)
            image_area = (annotation_frame["width"] * annotation_frame["height"]).replace(0, pd.NA)
            bbox_area_ratio = (widths * heights) / image_area
            annotation_frame = annotation_frame[(bbox_area_ratio >= area_min) & (bbox_area_ratio <= area_max)]
            if annotation_frame.empty:
                continue

        image_path = _resolve_image_path(csv_path.stem, dataset_directory)
        if image_path is None:
            warnings.warn(f"No image found for annotation '{csv_path.stem}' in '{dataset_directory}'.", stacklevel=2)
        annotation_frame["image_path"] = image_path.as_posix() if image_path else None
        annotation_frame["csv_path"] = csv_path.as_posix()

        rows.append(annotation_frame)

    if rows:
        dataset = pd.concat(rows, ignore_index=True)
    else:
        dataset = pd.DataFrame(
            columns=[
                "filename",
                "width",
                "height",
                "class",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
                "image_path",
                "csv_path",
            ]
        )

    if not allow_multiple_annotations and not dataset.empty:
        counts = dataset["filename"].value_counts()
        singletons = counts[counts == 1].index
        dataset = dataset[dataset["filename"].isin(singletons)].reset_index(drop=True)

    dataset.attrs["dataset_dir"] = dataset_directory.as_posix()
    if allowed_class_set is not None:
        dataset.attrs["allowed_classes"] = sorted(allowed_class_set)
    if area_range is not None:
        dataset.attrs["area_range"] = area_range
    dataset.attrs["allow_multiple_annotations"] = allow_multiple_annotations
    return dataset


__all__ = ["load_aircraft_dataset"]

