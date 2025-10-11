#!/usr/bin/env python3
"""
Usage:
  python dataset_info.py \
    --csv "aircraft_dataset/labels_with_split.csv" \
    --out "eda_outputs/dataset_info.txt"
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize dataset metadata from labels_with_split.csv")
    parser.add_argument(
        "--csv",
        type=str,
        default="aircraft_dataset/labels_with_split.csv",
        help="Path to labels_with_split.csv",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional path to write the summary text. If omitted, only prints to console.",
    )
    parser.add_argument(
        "--max-show-classes",
        type=int,
        default=200,
        help="Maximum number of class names to print in the class list section.",
    )
    return parser.parse_args()


def read_labels(csv_path: str) -> pd.DataFrame:
    dtypes = {
        "filename": "string",
        "width": "Int64",
        "height": "Int64",
        "class": "string",
        "xmin": "Int64",
        "ymin": "Int64",
        "xmax": "Int64",
        "ymax": "Int64",
        "split": "string",
    }
    try:
        df = pd.read_csv(csv_path, dtype=dtypes)
    except Exception:
        df = pd.read_csv(csv_path)
    return df


def bytes_to_mb(num_bytes: int) -> float:
    return float(num_bytes) / (1024.0 * 1024.0)


def print_header(title: str) -> None:
    print(title)


def summarize(df: pd.DataFrame, max_show_classes: int = 200) -> None:
    print_header("Columns and dtypes:")
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        print(f"- {col}: {dtype_str}")
    print()

    num_rows = len(df)
    num_unique_images = df["filename"].nunique(dropna=True)
    print_header("Row and image counts:")
    print(f"- rows (boxes): {num_rows}")
    print(f"- unique images (overall): {num_unique_images}")
    print()

    print_header("Splits and counts:")
    split_counts = df["split"].value_counts(dropna=False).sort_index()
    for split_name, count in split_counts.items():
        split_name_str = "<NA>" if pd.isna(split_name) else str(split_name)
        print(f"- {split_name_str}: rows={count}")
    
    if "split" in df.columns:
        print("\nUnique images per split:")
        uniq_per_split = (
            df.groupby("split")["filename"].nunique(dropna=True).sort_index()
        )
        for split_name, count in uniq_per_split.items():
            split_name_str = "<NA>" if pd.isna(split_name) else str(split_name)
            print(f"- {split_name_str}: unique_images={count}")
    print()

    classes = sorted([c for c in df["class"].dropna().unique().tolist()])
    print_header(f"Classes present ( {len(classes)} ):")
    to_show = classes[:max_show_classes]
    print(", ".join(to_show))
    if len(classes) > len(to_show):
        print(f"... and {len(classes) - len(to_show)} more")
    print()

    print_header("Boxes per class (desc):")
    boxes_per_class = (
        df.groupby("class").size().sort_values(ascending=False)
    )
    for cls, count in boxes_per_class.items():
        print(f"- {cls}: {count}")
    print()

    print_header("Unique images per class (desc):")
    uniq_imgs_per_class = (
        df.groupby("class")["filename"].nunique(dropna=True).sort_values(ascending=False)
    )
    for cls, count in uniq_imgs_per_class.items():
        print(f"- {cls}: {count}")
    print()

    print_header("Image size summary (width/height):")
    for col in ["width", "height"]:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            min_v = series.min()
            p50_v = series.quantile(0.5)
            p90_v = series.quantile(0.9)
            max_v = series.max()
            print(f"- {col}: min={min_v}, median={p50_v}, p90={p90_v}, max={max_v}")
    print()

    mem_bytes = int(df.memory_usage(deep=True).sum())
    print_header("Memory usage of DataFrame:")
    print(f"- {bytes_to_mb(mem_bytes):.2f} MB")


def tee_to_file_if_requested(out_path_str: str) -> Optional[object]:
    if not out_path_str:
        return None
    out_path = Path(out_path_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    class _Tee:
        def __init__(self, *streams):
            self._streams = streams
        def write(self, data):
            for s in self._streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self._streams:
                s.flush()

    logfile = out_path.open("w", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, logfile)
    sys.stderr = _Tee(sys.stderr, logfile)
    print(f"Writing summary to: {out_path}")
    return logfile


def main() -> None:
    args = parse_args()
    log_file = tee_to_file_if_requested(args.out)
    try:
        print("Reading labels...")
        df = read_labels(args.csv)
        print()
        summarize(df, max_show_classes=args.max_show_classes)
        print("Done.")
    finally:
        if log_file is not None:
            try:
                log_file.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()


