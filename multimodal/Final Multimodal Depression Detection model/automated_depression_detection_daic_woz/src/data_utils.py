from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .audio_features import extract_mfcc_features, load_audio, split_audio_segments


LABEL_CANDIDATES = [
    "PHQ8_Binary",
    "PHQ_Binary",
    "PHQ8_binary",
    "depression",
    "Depression",
    "label",
    "Label",
    "target",
]
ID_CANDIDATES = ["Participant_ID", "participant_ID", "participant_id", "Participant", "id", "ID"]


def find_column(df: pd.DataFrame, candidates: List[str]) -> str:
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    raise ValueError(f"Could not find any of columns {candidates}. Existing columns: {list(df.columns)}")


def load_split_csv(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing label CSV: {csv_path}")
    return pd.read_csv(csv_path)


def participant_to_audio_path(participant_id, audio_dir: str | Path) -> Path:
    """Return correct WAV path for Participant_ID.

    Pandas iterrows can convert 303 into 303.0, so we normalize it back to 303.
    """
    raw_pid = str(participant_id).strip()

    try:
        clean_pid = str(int(float(raw_pid)))
        pid_variants = [clean_pid, raw_pid]
    except (TypeError, ValueError):
        pid_variants = [raw_pid]

    audio_dir = Path(audio_dir)
    candidates = []

    for pid in dict.fromkeys(pid_variants):
        candidates.extend(
            [
                audio_dir / f"{pid}.wav",
                audio_dir / f"{pid}_AUDIO.wav",
                audio_dir / f"{pid}_audio.wav",
            ]
        )

    for path in candidates:
        if path.exists():
            return path

    return candidates[0]


def build_examples_from_split(
    split_csv: str | Path,
    audio_dir: str | Path,
    cfg_features: dict,
    cache_path: str | Path | None = None,
    overwrite_cache: bool = False,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Create feature tensors from one split CSV.

    Returns:
        X: (num_segments, fixed_frames, feature_dim)
        y: (num_segments,)
        metadata: rows mapping each segment to participant id and source file
    """
    cache_path = Path(cache_path) if cache_path else None
    meta_cache_path = cache_path.with_suffix(".metadata.csv") if cache_path else None
    if cache_path and cache_path.exists() and meta_cache_path and meta_cache_path.exists() and not overwrite_cache:
        arr = np.load(cache_path, allow_pickle=False)
        return arr["X"], arr["y"], pd.read_csv(meta_cache_path)

    df = load_split_csv(split_csv)
    id_col = find_column(df, ID_CANDIDATES)
    label_col = find_column(df, LABEL_CANDIDATES)

    sr = int(cfg_features.get("sample_rate", 16000))
    segment_seconds = float(cfg_features.get("segment_seconds", 20))
    stride_seconds = float(cfg_features.get("segment_stride_seconds", segment_seconds))
    max_segments = int(cfg_features.get("max_segments_per_file", 6))

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    meta_rows: List[Dict[str, object]] = []
    missing_audio: List[str] = []

    for _, row in df.iterrows():
        participant_id = row[id_col]
        label = int(row[label_col])
        audio_path = participant_to_audio_path(participant_id, audio_dir)
        if not audio_path.exists():
            missing_audio.append(str(audio_path))
            continue

        y_wave, sr_loaded = load_audio(
            audio_path,
            sample_rate=sr,
            normalize=bool(cfg_features.get("normalize_audio", True)),
            trim_silence=bool(cfg_features.get("trim_silence", False)),
        )
        segments = split_audio_segments(
            y_wave,
            sr_loaded,
            segment_seconds=segment_seconds,
            stride_seconds=stride_seconds,
            max_segments=max_segments,
        )
        for seg_idx, segment in enumerate(segments):
            feat = extract_mfcc_features(segment, sr_loaded, cfg_features)
            X_list.append(feat)
            y_list.append(label)
            meta_rows.append(
                {
                    "participant_id": participant_id,
                    "segment_index": seg_idx,
                    "label": label,
                    "audio_path": str(audio_path),
                }
            )

    if missing_audio:
        print(f"Warning: {len(missing_audio)} audio files were missing. First missing file: {missing_audio[0]}")

    if not X_list:
        raise RuntimeError(
            "No audio examples were created. Check DAIC_WOZ/data/audio and label CSV participant IDs."
        )

    X = np.stack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    metadata = pd.DataFrame(meta_rows)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, X=X, y=y)
        metadata.to_csv(meta_cache_path, index=False)
    return X, y, metadata


def fit_scaler(X_train: np.ndarray, scaler_path: str | Path) -> StandardScaler:
    """Fit a StandardScaler feature-wise over all frames."""
    scaler = StandardScaler()
    n, t, f = X_train.shape
    scaler.fit(X_train.reshape(n * t, f))
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    return scaler


def load_scaler(scaler_path: str | Path) -> StandardScaler | None:
    p = Path(scaler_path)
    if p.exists():
        return joblib.load(p)
    return None


def transform_with_scaler(X: np.ndarray, scaler: StandardScaler | None) -> np.ndarray:
    if scaler is None:
        return X.astype(np.float32)
    n, t, f = X.shape
    X_scaled = scaler.transform(X.reshape(n * t, f)).reshape(n, t, f)
    return X_scaled.astype(np.float32)


def compute_dataset_summary(cfg: dict) -> Dict[str, object]:
    data_dir = Path(cfg["paths"]["data_dir"])
    audio_dir = Path(cfg["paths"]["audio_dir"])
    transcript_dir = Path(cfg["paths"]["transcript_dir"])
    label_dir = Path(cfg["paths"]["label_dir"])
    summary = {
        "data_dir_exists": data_dir.exists(),
        "audio_count": len(list(audio_dir.glob("*.wav"))) if audio_dir.exists() else 0,
        "transcript_count": len(list(transcript_dir.glob("*.csv"))) if transcript_dir.exists() else 0,
        "label_files": [p.name for p in label_dir.glob("*.csv")] if label_dir.exists() else [],
    }
    return summary
