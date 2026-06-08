from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable, List, Tuple

import librosa
import numpy as np
import soundfile as sf


def load_audio(path_or_bytes, sample_rate: int = 16000, normalize: bool = True, trim_silence: bool = False) -> Tuple[np.ndarray, int]:
    """Load audio from a file path or bytes object as mono float32 waveform."""
    if isinstance(path_or_bytes, (str, Path)):
        y, sr = librosa.load(str(path_or_bytes), sr=sample_rate, mono=True)
    else:
        if hasattr(path_or_bytes, "read"):
            raw = path_or_bytes.read()
        else:
            raw = path_or_bytes
        data, sr_native = sf.read(io.BytesIO(raw), always_2d=False)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        y = data.astype(np.float32)
        if sr_native != sample_rate:
            y = librosa.resample(y, orig_sr=sr_native, target_sr=sample_rate)
        sr = sample_rate

    if trim_silence and y.size > 0:
        y, _ = librosa.effects.trim(y, top_db=30)

    if normalize and y.size > 0:
        peak = float(np.max(np.abs(y)))
        if peak > 0:
            y = y / peak
    return y.astype(np.float32), sr


def split_audio_segments(
    y: np.ndarray,
    sr: int,
    segment_seconds: float,
    stride_seconds: float,
    max_segments: int | None = None,
) -> List[np.ndarray]:
    """Split a waveform into fixed-duration windows. Pads short audio."""
    segment_len = int(segment_seconds * sr)
    stride_len = int(stride_seconds * sr)
    if segment_len <= 0:
        raise ValueError("segment_seconds must be positive")
    if y.size == 0:
        y = np.zeros(segment_len, dtype=np.float32)

    if y.size < segment_len:
        return [np.pad(y, (0, segment_len - y.size)).astype(np.float32)]

    segments: List[np.ndarray] = []
    for start in range(0, y.size - segment_len + 1, max(1, stride_len)):
        segments.append(y[start : start + segment_len].astype(np.float32))
        if max_segments is not None and len(segments) >= max_segments:
            break
    if not segments:
        segments.append(y[:segment_len].astype(np.float32))
    return segments


def pad_or_truncate_features(x: np.ndarray, fixed_frames: int) -> np.ndarray:
    """Convert feature matrix to shape (fixed_frames, feature_dim)."""
    # x arrives as (feature_dim, time), convert to (time, feature_dim)
    x = x.T.astype(np.float32)
    frames, dims = x.shape
    if frames < fixed_frames:
        pad = np.zeros((fixed_frames - frames, dims), dtype=np.float32)
        x = np.vstack([x, pad])
    elif frames > fixed_frames:
        x = x[:fixed_frames]
    return x


def extract_mfcc_features(y: np.ndarray, sr: int, cfg_features: dict) -> np.ndarray:
    """Extract MFCC + optional delta features as a fixed 2D matrix."""
    n_mfcc = int(cfg_features.get("n_mfcc", 40))
    n_fft = int(cfg_features.get("n_fft", 1024))
    hop_length = int(cfg_features.get("hop_length", 512))
    win_length = int(cfg_features.get("win_length", 1024))
    fixed_frames = int(cfg_features.get("fixed_frames", 600))

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )
    features = [mfcc]
    if cfg_features.get("use_delta", True):
        features.append(librosa.feature.delta(mfcc))
    if cfg_features.get("use_delta_delta", True):
        features.append(librosa.feature.delta(mfcc, order=2))

    stacked = np.vstack(features)
    return pad_or_truncate_features(stacked, fixed_frames)


def extract_log_mel_for_display(y: np.ndarray, sr: int, cfg_features: dict) -> np.ndarray:
    """Log-mel spectrogram for visualization in Streamlit."""
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=int(cfg_features.get("n_fft", 1024)),
        hop_length=int(cfg_features.get("hop_length", 512)),
        n_mels=80,
        power=2.0,
    )
    return librosa.power_to_db(mel, ref=np.max)


def features_from_audio_source(path_or_bytes, cfg_features: dict) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return model-ready feature tensor for one full input audio and its waveform."""
    sr = int(cfg_features.get("sample_rate", 16000))
    y, sr = load_audio(
        path_or_bytes,
        sample_rate=sr,
        normalize=bool(cfg_features.get("normalize_audio", True)),
        trim_silence=bool(cfg_features.get("trim_silence", False)),
    )
    # During inference use one segment from the beginning/middle depending on duration.
    segment_seconds = float(cfg_features.get("segment_seconds", 20))
    segment_len = int(segment_seconds * sr)
    if y.size > segment_len:
        start = max(0, (y.size - segment_len) // 2)
        segment = y[start : start + segment_len]
    else:
        segment = np.pad(y, (0, max(0, segment_len - y.size)))
    x = extract_mfcc_features(segment, sr, cfg_features)
    return x[np.newaxis, ...].astype(np.float32), y, sr
