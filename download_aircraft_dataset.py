#!/usr/bin/env python3
from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from typing import List


DATASET_ID: str = "a2015003713/militaryaircraftdetectiondataset"
TARGET_DIR: Path = Path("aircraft_dataset").resolve()
DOWNLOAD_DIR: Path = (TARGET_DIR.parent / ".kaggle_downloads").resolve()


def find_downloaded_zip(download_dir: Path) -> Path:
    zip_files: List[Path] = list(download_dir.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No zip files found in download directory: {download_dir}")
    zip_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return zip_files[0]


def unzip_to_target(zip_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)


def download_with_kaggle_api(dataset: str, download_dir: Path) -> None:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "The 'kaggle' package is required. Install it with: pip install kaggle"
        ) from exc

    api = KaggleApi()
    api.authenticate()

    download_dir.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(dataset, path=str(download_dir), unzip=False, quiet=False)


def main() -> int:
    print(f"Downloading Kaggle dataset '{DATASET_ID}'...")
    download_with_kaggle_api(DATASET_ID, DOWNLOAD_DIR)

    zip_path = find_downloaded_zip(DOWNLOAD_DIR)
    print(f"Downloaded zip: {zip_path}")

    print(f"Extracting into: {TARGET_DIR}")
    unzip_to_target(zip_path, TARGET_DIR)
    print("Done.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)