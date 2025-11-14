"""Utilities for loading the aircraft detection dataset into a pandas DataFrame."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, Optional, Sequence

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

    dataset.attrs["dataset_dir"] = dataset_directory.as_posix()
    if allowed_class_set is not None:
        dataset.attrs["allowed_classes"] = sorted(allowed_class_set)
    return dataset


__all__ = ["load_aircraft_dataset"]

