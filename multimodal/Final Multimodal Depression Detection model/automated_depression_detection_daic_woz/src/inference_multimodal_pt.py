"""
PyTorch inference module — replaces src/inference_multimodal.py.
Drop this file into src/ as inference_multimodal_pt.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from .audio_features import extract_log_mel_for_display, features_from_audio_source
from .config import load_config
from .data_utils import load_scaler, transform_with_scaler
from .text_utils import encode_texts, zero_text_embedding


def load_multimodal_artifacts_pt(config_path=None):
    """Load PyTorch model + scaler. Returns (cfg, model, scaler)."""
    from .multimodal_model_pt import load_model

    cfg = load_config(config_path) if config_path else load_config()
    model_path = Path(cfg["paths"]["model_dir"]) / "best_multimodal_pt.pth"
    if not model_path.exists():
        raise FileNotFoundError(
            f"PyTorch model not found at {model_path}. "
            "Train first with: python -m pytorch_src.train_multimodal_pt"
        )

    # Load preprocess config for audio_feature_dim
    preprocess_cfg_path = Path(cfg["paths"].get("preprocess_config", ""))
    audio_feature_dim = 120  # default: 40 MFCCs × 3
    if preprocess_cfg_path.exists():
        import json
        with preprocess_cfg_path.open() as f:
            pcfg = json.load(f)
        n_mfcc = int(pcfg.get("features", {}).get("n_mfcc", 40))
        use_delta = pcfg.get("features", {}).get("use_delta", True)
        use_delta_delta = pcfg.get("features", {}).get("use_delta_delta", True)
        audio_feature_dim = n_mfcc * (1 + int(use_delta) + int(use_delta_delta))

    model = load_model(str(model_path), cfg, audio_feature_dim=audio_feature_dim)
    scaler = load_scaler(cfg["paths"]["scaler"])
    return cfg, model, scaler


def transcribe_audio_with_whisper(audio_bytes: bytes, suffix: str = ".wav") -> str:
    """Transcribe with faster-whisper. Returns empty string on failure."""
    try:
        from faster_whisper import WhisperModel
    except Exception:
        return ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(tmp_path, beam_size=1)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        Path(tmp_path).unlink(missing_ok=True)
        return text
    except Exception:
        return ""


def decide_label(depressed_prob: float, threshold: float = 0.65) -> str:
    if depressed_prob >= threshold:
        return "Depression Risk Detected"
    if depressed_prob >= 0.45:
        return "Borderline / Needs Further Screening"
    return "Non-depressed"


def _get_attention(model, x_audio_t: torch.Tensor, x_text_t: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        _, attn = model(x_audio_t, x_text_t)
    return attn.cpu().numpy()


def _gradcam(model, x_audio_t: torch.Tensor, x_text_t: torch.Tensor) -> np.ndarray:
    """Grad-CAM equivalent on the Conv1D layer."""
    model.eval()
    # Hook to capture conv output
    conv_output = {}
    conv_grad = {}

    def fwd_hook(m, inp, out):
        conv_output["val"] = out

    def bwd_hook(m, grad_in, grad_out):
        conv_grad["val"] = grad_out[0]

    h1 = model.conv1d.register_forward_hook(fwd_hook)
    h2 = model.conv1d.register_full_backward_hook(bwd_hook)

    x_audio_t = x_audio_t.requires_grad_(True)
    prob, _ = model(x_audio_t, x_text_t)
    prob.sum().backward()

    h1.remove()
    h2.remove()

    # conv_output["val"]: (batch, conv_filters, time)
    conv_out = conv_output["val"]          # (batch, filters, time)
    grads    = conv_grad["val"]            # (batch, filters, time)
    pooled   = grads.mean(dim=2)           # (batch, filters)
    weighted = (conv_out * pooled.unsqueeze(2)).sum(dim=1)  # (batch, time)
    heatmap  = torch.relu(weighted).detach().cpu().numpy()
    # normalise
    for i in range(heatmap.shape[0]):
        m = heatmap[i].max()
        if m > 0:
            heatmap[i] /= m
    return heatmap


def _saliency(model, x_audio_t: torch.Tensor, x_text_t: torch.Tensor) -> np.ndarray:
    model.eval()
    x_audio_t = x_audio_t.clone().detach().requires_grad_(True)
    prob, _ = model(x_audio_t, x_text_t)
    prob.sum().backward()
    saliency = x_audio_t.grad.abs().max(dim=-1).values.detach().cpu().numpy()
    for i in range(saliency.shape[0]):
        m = saliency[i].max()
        if m > 0:
            saliency[i] /= m
    return saliency


def predict_multimodal_audio_pt(
    path_or_bytes,
    transcript_text: str,
    cfg: dict,
    model,
    scaler=None,
) -> Dict:
    """Run PyTorch multimodal inference. Returns same dict as the TF version."""
    X_audio, waveform, sr = features_from_audio_source(path_or_bytes, cfg["features"])
    X_audio_scaled = transform_with_scaler(X_audio, scaler)

    if transcript_text and transcript_text.strip():
        X_text = encode_texts([transcript_text.strip()])
    else:
        X_text = zero_text_embedding()[np.newaxis, :]

    x_audio_t = torch.tensor(X_audio_scaled, dtype=torch.float32)
    x_text_t  = torch.tensor(X_text,         dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        prob_t, _ = model(x_audio_t, x_text_t)
    depressed_prob     = float(prob_t.numpy().reshape(-1)[0])
    non_depressed_prob = 1.0 - depressed_prob

    threshold = float(cfg["training"].get("threshold", 0.65))
    label     = decide_label(depressed_prob, threshold)

    attention = _get_attention(model, x_audio_t, x_text_t)[0]
    gradcam   = _gradcam(model, x_audio_t, x_text_t)[0]
    saliency  = _saliency(model, x_audio_t, x_text_t)[0]
    log_mel   = extract_log_mel_for_display(waveform, sr, cfg["features"])

    return {
        "label":                  label,
        "depressed_probability":  depressed_prob,
        "non_depressed_probability": non_depressed_prob,
        "threshold":              threshold,
        "transcript_text":        transcript_text,
        "features":               X_audio_scaled[0],
        "waveform":               waveform,
        "sample_rate":            sr,
        "attention":              attention,
        "gradcam":                gradcam,
        "saliency":               saliency,
        "log_mel":                log_mel,
    }
