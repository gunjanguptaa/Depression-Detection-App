"""Download DAIC-WOZ Kaggle dataset using Kaggle API.

Prerequisite:
1. Create/download kaggle.json from your Kaggle account.
2. Place it at ~/.kaggle/kaggle.json on Linux/Mac or C:\\Users\\<you>\\.kaggle\\kaggle.json on Windows.
3. Run: python scripts/download_dataset.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_SLUG = "gunjangggupta/daic-woz"
TARGET_ROOT = ROOT / "DAIC_WOZ"


def main() -> None:
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Kaggle dataset: {DATASET_SLUG}")
    cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET_SLUG,
        "-p",
        str(TARGET_ROOT),
        "--unzip",
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            "Kaggle download failed. Check Kaggle API installation and kaggle.json credentials. "
            "You can also manually download the dataset and place it in DAIC_WOZ/."
        ) from exc
    print("Download finished. Now run: python scripts/verify_dataset.py")


if __name__ == "__main__":
    main()
