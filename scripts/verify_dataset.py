from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT / "DAIC_WOZ" / "data" / "audio"
TRANSCRIPT_DIR = ROOT / "DAIC_WOZ" / "data" / "transcripts"
LABEL_DIR = ROOT / "DAIC_WOZ" / "data" / "labels"
EXPECTED_LABELS = [
    "train_split_Depression_AVEC2017.csv",
    "dev_split_Depression_AVEC2017.csv",
    "test_split_Depression_AVEC2017.csv",
]


def count(pattern: str, path: Path) -> int:
    return len(list(path.glob(pattern))) if path.exists() else 0


def main() -> None:
    print("Verifying DAIC-WOZ project dataset structure...\n")
    checks = {
        "audio_dir_exists": AUDIO_DIR.exists(),
        "transcript_dir_exists": TRANSCRIPT_DIR.exists(),
        "label_dir_exists": LABEL_DIR.exists(),
        "wav_count": count("*.wav", AUDIO_DIR),
        "transcript_csv_count": count("*.csv", TRANSCRIPT_DIR),
        "label_csv_count": count("*.csv", LABEL_DIR),
    }
    for key, value in checks.items():
        print(f"{key}: {value}")

    print("\nRequired label files:")
    ok = True
    for name in EXPECTED_LABELS:
        exists = (LABEL_DIR / name).exists()
        print(f"  {name}: {'FOUND' if exists else 'MISSING'}")
        ok = ok and exists

    if checks["wav_count"] == 0:
        ok = False
        print("\nERROR: No WAV audio files found in DAIC_WOZ/data/audio")
    if checks["label_csv_count"] == 0:
        ok = False
        print("\nERROR: No label CSV files found in DAIC_WOZ/data/labels")

    if ok:
        print("\nDataset looks ready ✅")
        sample_label = LABEL_DIR / EXPECTED_LABELS[0]
        if sample_label.exists():
            df = pd.read_csv(sample_label)
            print("\nTrain label preview:")
            print(df.head())
    else:
        print("\nDataset is not ready yet. Please match the folder structure in README.md.")
        sys.exit(1)


if __name__ == "__main__":
    main()
