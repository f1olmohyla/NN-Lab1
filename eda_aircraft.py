#!/usr/bin/env python3
"""
BBox Area Analysis (in pixels) for aircraft_dataset.

Outputs:
- Distribution of bounding box areas (in pixels) per class (Matplotlib histograms)
- Per-class summary CSV with bbox area stats (count, min, mean, median, p90, max)

Usage:
  python eda_aircraft.py \
    --csv "/Users/f1ol/workspace/ukma/NNs/LAB1/aircraft_dataset/labels_with_split.csv" \
    --out-dir "/Users/f1ol/workspace/ukma/NNs/LAB1/aircraft_dataset/eda_outputs"
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BBox area (pixels) analysis for aircraft_dataset")
    parser.add_argument(
        "--csv",
        type=str,
        default="/Users/f1ol/workspace/ukma/NNs/LAB1/aircraft_dataset/labels_with_split.csv",
        help="Path to labels_with_split.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/Users/f1ol/workspace/ukma/NNs/LAB1/aircraft_dataset/eda_outputs",
        help="Directory to save plots and any outputs",
    )
    parser.add_argument(
        "--max-classes-for-plots",
        type=int,
        default=60,
        help="If there are many classes, limit number of subplots per figure by splitting into multiple figures",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins for area distributions",
    )
    return parser.parse_args()


def ensure_out_dir(path: str) -> Path:
    out_path = Path(path)
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


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
    # Let pandas infer if the exact nullable ints fail; this keeps memory reasonable
    try:
        df = pd.read_csv(csv_path, dtype=dtypes)
    except Exception:
        df = pd.read_csv(csv_path)
    return df


def compute_bbox_area_px(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure numeric and compute raw bbox area in pixels
    for col in ["xmin", "ymin", "xmax", "ymax"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bw = (df["xmax"] - df["xmin"]).clip(lower=0)
    bh = (df["ymax"] - df["ymin"]).clip(lower=0)
    area_px = (bw * bh).astype("float64")
    df = df.assign(bbox_area_px=area_px)
    return df


def plot_bbox_area_distributions_px(
    df: pd.DataFrame, out_dir: Path, max_classes_per_fig: int = 12, bins: int = 50
) -> None:
    """Create histograms per class for bbox area in pixels."""
    classes = sorted(df["class"].dropna().unique().tolist())
    if len(classes) == 0:
        print("No classes found to plot.")
        return

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    for idx, chunk in enumerate(chunks(classes, max_classes_per_fig)):
        sub = df[df["class"].isin(chunk)]
        n_classes = len(chunk)

        # Single-row grid: bbox area (px)
        fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 3.5), squeeze=False)

        for j, cls in enumerate(chunk):
            cls_df = sub[sub["class"] == cls]
            axes[0, j].hist(
                cls_df["bbox_area_px"].dropna().values, bins=bins, color="#2ca02c", alpha=0.85
            )
            axes[0, j].set_title(f"{cls}")
            axes[0, j].set_xlabel("bbox area (px)")
            axes[0, j].set_ylabel("count")

        fig.tight_layout()
        fig_path = out_dir / f"bbox_area_px_part_{idx+1}.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {fig_path}")


def main() -> None:
    args = parse_args()
    out_dir = ensure_out_dir(args.out_dir)

    # Tee all stdout/stderr to eda.txt in the out_dir
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

    log_path = out_dir / "eda.txt"
    logfile = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, logfile)
    sys.stderr = _Tee(sys.stderr, logfile)
    print(f"Logging console output to: {log_path}")

    print("Reading labels...")
    df = read_labels(args.csv)
    
    # Compute bbox pixel area and plot distributions
    df = compute_bbox_area_px(df)
    print("\nPlotting bbox area distributions (pixels)...")
    plot_bbox_area_distributions_px(df, out_dir=out_dir, max_classes_per_fig=12, bins=args.bins)

    # Save per-class summary stats for bbox area in pixels
    def q(df_col, p):
        return df_col.quantile(p)

    summary = df.groupby("class").agg(
        num_boxes=("bbox_area_px", "size"),
        area_px_min=("bbox_area_px", "min"),
        area_px_mean=("bbox_area_px", "mean"),
        area_px_median=("bbox_area_px", "median"),
        area_px_p90=("bbox_area_px", lambda s: q(s, 0.90)),
        area_px_max=("bbox_area_px", "max"),
    ).reset_index()
    summary_path = out_dir / "per_class_bbox_area_px_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved per-class bbox area summary: {summary_path}")

    # Print concise per-class stats to console (also captured in eda.txt)
    def _fmt_int(x):
        return "nan" if pd.isna(x) else str(int(round(float(x))))

    summary_sorted = summary.sort_values(by="area_px_median", ascending=False)
    print("\nPer-class bbox area (px) stats:")
    for _, row in summary_sorted.iterrows():
        cls = row["class"]
        cnt = _fmt_int(row["num_boxes"])
        amin = _fmt_int(row["area_px_min"])
        amed = _fmt_int(row["area_px_median"])
        ap90 = _fmt_int(row["area_px_p90"])
        amax = _fmt_int(row["area_px_max"])
        print(f"- {cls}: boxes={cnt}, area_px[min/median/p90/max]={amin}/{amed}/{ap90}/{amax}")

    print("\nDone.")


if __name__ == "__main__":
    main()


