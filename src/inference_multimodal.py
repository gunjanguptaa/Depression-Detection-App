from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict

import numpy as np

from .audio_features import extract_log_mel_for_display, features_from_audio_source
from .config import load_config
from .data_utils import load_scaler, transform_with_scaler
from .multimodal_explainability import get_multimodal_attention_weights, multimodal_gradcam_1d, multimodal_saliency_map
from .multimodal_model import load_multimodal_model
from .text_utils import encode_texts, zero_text_embedding


def load_multimodal_artifacts(config_path: str | None = None):
    cfg = load_config(config_path) if config_path else load_config()
    model_path = Path(cfg["paths"]["model_dir"]) / "best_multimodal_attention_lstm.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Multimodal model not found at {model_path}. Train first with: python -m src.train_multimodal")
    model = load_multimodal_model(str(model_path))
    scaler = load_scaler(cfg["paths"]["scaler"])
    return cfg, model, scaler


def transcribe_audio_with_whisper(audio_bytes: bytes, suffix: str = ".wav") -> str:
    """Transcribe recorded/uploaded audio using faster-whisper or openai-whisper. Returns empty string on failure."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            from faster_whisper import WhisperModel
            model = WhisperModel("tiny", device="cpu", compute_type="int8")
            segments, _ = model.transcribe(tmp_path, beam_size=1)
            text = " ".join(seg.text.strip() for seg in segments).strip()
            return text
        except Exception:
            pass

        try:
            import whisper
            model = whisper.load_model("tiny")
            result = model.transcribe(tmp_path, verbose=False)
            text = str(result.get("text", "")).strip()
            return text
        except Exception:
            return ""
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def decide_label(depressed_prob: float, threshold: float = 0.65) -> str:
    if depressed_prob >= threshold:
        return "Depression Risk Detected"
    if depressed_prob >= 0.45:
        return "Borderline / Needs Further Screening"
    return "Non-depressed"


def predict_multimodal_audio(path_or_bytes, transcript_text: str, cfg: dict, model, scaler=None) -> Dict[str, object]:
    X_audio, waveform, sr = features_from_audio_source(path_or_bytes, cfg["features"])
    X_audio_scaled = transform_with_scaler(X_audio, scaler)

    if transcript_text and transcript_text.strip():
        X_text = encode_texts([transcript_text.strip()])
    else:
        X_text = zero_text_embedding()[np.newaxis, :]

    depressed_prob = float(model.predict([X_audio_scaled, X_text], verbose=0).reshape(-1)[0])
    non_depressed_prob = 1.0 - depressed_prob
    threshold = float(cfg["training"].get("threshold", 0.65))
    label = decide_label(depressed_prob, threshold)

    attention = get_multimodal_attention_weights(model, X_audio_scaled, X_text)[0]
    gradcam = multimodal_gradcam_1d(model, X_audio_scaled, X_text)[0]
    saliency = multimodal_saliency_map(model, X_audio_scaled, X_text)[0]
    log_mel = extract_log_mel_for_display(waveform, sr, cfg["features"])

    return {
        "label": label,
        "depressed_probability": depressed_prob,
        "non_depressed_probability": non_depressed_prob,
        "threshold": threshold,
        "transcript_text": transcript_text,
        "features": X_audio_scaled[0],
        "waveform": waveform,
        "sample_rate": sr,
        "attention": attention,
        "gradcam": gradcam,
        "saliency": saliency,
        "log_mel": log_mel,
    }
