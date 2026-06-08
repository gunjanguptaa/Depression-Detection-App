from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from .audio_features import extract_log_mel_for_display, features_from_audio_source
from .config import load_config
from .data_utils import load_scaler, transform_with_scaler
from .explainability import get_attention_weights, gradcam_1d, saliency_map
from .model import load_trained_model


def load_artifacts(config_path: str | None = None):
    cfg = load_config(config_path) if config_path else load_config()
    model_path = Path(cfg["paths"]["best_model"])
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Train first with: python -m src.train")
    model = load_trained_model(str(model_path))
    scaler = load_scaler(cfg["paths"]["scaler"])
    return cfg, model, scaler


def predict_audio(path_or_bytes, cfg: dict, model, scaler=None) -> Dict[str, object]:
    X, waveform, sr = features_from_audio_source(path_or_bytes, cfg["features"])
    X_scaled = transform_with_scaler(X, scaler)
    depressed_prob = float(model.predict(X_scaled, verbose=0).reshape(-1)[0])
    non_depressed_prob = 1.0 - depressed_prob
    threshold = float(cfg["training"].get("threshold", 0.5))
    label = "Depressed" if depressed_prob >= threshold else "Non-depressed"

    attention = get_attention_weights(model, X_scaled)[0]
    gradcam = gradcam_1d(model, X_scaled)[0]
    saliency = saliency_map(model, X_scaled)[0]
    log_mel = extract_log_mel_for_display(waveform, sr, cfg["features"])

    return {
        "label": label,
        "depressed_probability": depressed_prob,
        "non_depressed_probability": non_depressed_prob,
        "threshold": threshold,
        "features": X_scaled[0],
        "waveform": waveform,
        "sample_rate": sr,
        "attention": attention,
        "gradcam": gradcam,
        "saliency": saliency,
        "log_mel": log_mel,
    }
