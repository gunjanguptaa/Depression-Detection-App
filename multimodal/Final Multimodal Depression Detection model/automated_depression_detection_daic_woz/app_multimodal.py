"""
app_multimodal.py — Multimodal Depression Risk Detection (PyTorch version)
Identical UI to the original; only the backend model is PyTorch instead of TF/Keras.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import load_config
from src.data_utils import compute_dataset_summary
from src.inference_multimodal_pt import (
    load_multimodal_artifacts_pt,
    predict_multimodal_audio_pt,
    transcribe_audio_with_whisper,
)

st.set_page_config(
    page_title="Multimodal Depression Risk Detection",
    page_icon="🎙️",
    layout="wide",
)


# ── Plot helpers (unchanged from original) ──────────────────────────────────

def fig_waveform(waveform: np.ndarray, sr: int):
    fig, ax = plt.subplots(figsize=(10, 3))
    times = np.arange(len(waveform)) / sr
    ax.plot(times, waveform)
    ax.set_title("Input Audio Waveform")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    return fig


def fig_matrix(matrix: np.ndarray, title: str, xlabel: str = "Time Frames", ylabel: str = "Features"):
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(matrix, aspect="auto", origin="lower")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    return fig


def fig_line(values: np.ndarray, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(values)
    ax.set_title(title)
    ax.set_xlabel("Time Frames")
    ax.set_ylabel(ylabel)
    return fig


# ── Cached model loader ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def cached_load_multimodal_artifacts():
    return load_multimodal_artifacts_pt()


# ── Config & paths ───────────────────────────────────────────────────────────

cfg       = load_config()
plots_dir = Path(cfg["paths"]["plots_dir"])
reports_dir = Path(cfg["paths"]["reports_dir"])
model_path  = Path(cfg["paths"]["model_dir"]) / "best_multimodal_pt.pth"

# ── Page layout ──────────────────────────────────────────────────────────────

st.title("🎙️ Multimodal Depression Risk Detection")
st.caption(
    "Audio CNN-BiLSTM-Attention + transcript NLP fusion (PyTorch). "
    "Research prototype only — not a medical diagnosis tool."
)

with st.sidebar:
    st.header("Model Status")
    if model_path.exists():
        st.success("✅ PyTorch multimodal model found")
    else:
        st.error("❌ Model not trained yet")
        st.info(
            "Train on Kaggle with:\n"
            "```\npython train_multimodal_pt.py\n```\n"
            "Then upload `best_multimodal_pt.pth` to `artifacts/models/`"
        )
    st.divider()
    st.write("Expected dataset folders:")
    st.code("DAIC_WOZ/data/audio\nDAIC_WOZ/data/transcripts\nDAIC_WOZ/data/labels")

prediction_tab, dashboard_tab, data_tab, guide_tab = st.tabs(
    ["Prediction", "Multimodal Dashboard", "Dataset Explorer", "Guide"]
)

# ── Prediction tab ───────────────────────────────────────────────────────────

with prediction_tab:
    st.subheader("Predict Depression Risk from Audio + Words")
    col_a, col_b = st.columns(2)
    with col_a:
        uploaded_file = st.file_uploader(
            "Option 1: Upload audio", type=["wav", "mp3", "ogg", "flac", "m4a"]
        )
    with col_b:
        if hasattr(st, "audio_input"):
            recorded_audio = st.audio_input("Option 2: Record audio")
        else:
            recorded_audio = None
            st.warning("Your Streamlit version does not support st.audio_input. Use upload option.")

    audio_source = None
    source_name  = None
    suffix       = ".wav"
    if uploaded_file is not None:
        audio_source = uploaded_file.getvalue()
        source_name  = uploaded_file.name
        suffix       = Path(uploaded_file.name).suffix or ".wav"
    elif recorded_audio is not None:
        audio_source = recorded_audio.getvalue()
        source_name  = "recorded_audio.wav"
        suffix       = ".wav"

    manual_text = st.text_area(
        "Optional: type transcript manually or correct Whisper transcript here",
        placeholder="Example: I completed my project today and I am feeling happy and confident.",
        height=100,
    )
    use_whisper = st.checkbox(
        "Auto-transcribe audio using Whisper if manual transcript is empty", value=True
    )

    if audio_source is not None:
        st.audio(audio_source, format="audio/wav")

        if not model_path.exists():
            st.error(
                "PyTorch model file is missing. "
                "Train on Kaggle and upload `best_multimodal_pt.pth` to `artifacts/models/`."
            )
        else:
            transcript_text = manual_text.strip()
            if not transcript_text and use_whisper:
                with st.spinner("Transcribing audio using Whisper..."):
                    transcript_text = transcribe_audio_with_whisper(audio_source, suffix=suffix)

            st.markdown("### Transcript used by text branch")
            if transcript_text:
                st.success(transcript_text)
            else:
                st.warning(
                    "No transcript was produced. Text branch will use an empty embedding. "
                    "Install `faster-whisper` and `sentence-transformers` for auto-transcription."
                )

            with st.spinner("Running multimodal audio + transcript model..."):
                cfg_loaded, model, scaler = cached_load_multimodal_artifacts()
                result = predict_multimodal_audio_pt(
                    audio_source, transcript_text, cfg_loaded, model, scaler
                )

            st.markdown(f"### Prediction for `{source_name}`")
            c1, c2, c3 = st.columns(3)
            c1.metric("Final Decision",             result["label"])
            c2.metric("Depressed Probability",      f"{result['depressed_probability'] * 100:.2f}%")
            c3.metric("Non-depressed Probability",  f"{result['non_depressed_probability'] * 100:.2f}%")

            st.caption("Decision rule: <45% = Non-depressed · 45–65% = Borderline · >65% = Depression risk.")
            prob_df = pd.DataFrame({
                "Class":           ["Non-depressed", "Depression Risk"],
                "Probability (%)": [
                    result["non_depressed_probability"] * 100,
                    result["depressed_probability"]     * 100,
                ],
            })
            st.bar_chart(prob_df.set_index("Class"))

            st.divider()
            st.subheader("Audio Visualizations")
            st.pyplot(fig_waveform(result["waveform"], result["sample_rate"]))
            st.pyplot(fig_matrix(result["log_mel"],        "Log-Mel Spectrogram",           ylabel="Mel Bands"))
            st.pyplot(fig_matrix(result["features"].T,     "MFCC + Delta + Delta-Delta",    ylabel="Feature Index"))

            st.divider()
            st.subheader("Explainability")
            st.pyplot(fig_line(result["attention"], "Attention Weights Over Time", "Attention Weight"))
            st.pyplot(fig_line(result["gradcam"],   "Grad-CAM Style Time Heatmap",  "Importance"))
            st.pyplot(fig_line(result["saliency"],  "Input Gradient Saliency",      "Saliency"))
    else:
        st.info("Upload or record audio. For best results, provide or correct the transcript too.")

# ── Dashboard tab ────────────────────────────────────────────────────────────

with dashboard_tab:
    st.subheader("Multimodal Training and Evaluation Graphs")
    plot_files = [
        "multimodal/loss_curve.png",
        "multimodal/accuracy_curve.png",
        "multimodal/auc_curve.png",
        "multimodal_class_distribution_train.png",
        "multimodal_roc_curve.png",
        "multimodal_confusion_matrix.png",
        "multimodal_prediction_probability_distribution.png",
        "multimodal_example_mfcc_heatmap.png",
        "multimodal_example_attention_weights.png",
        "multimodal_example_gradcam_1d.png",
    ]
    existing = [p for p in plot_files if (plots_dir / p).exists()]
    if not existing:
        st.warning("No multimodal plots found yet. Train and evaluate the model first.")
    else:
        for name in existing:
            st.markdown(f"#### {Path(name).name.replace('_', ' ').replace('.png', '').title()}")
            st.image(str(plots_dir / name), use_container_width=True)

    st.divider()
    st.subheader("Metrics")
    metric_files = sorted(reports_dir.glob("*multimodal*metrics*.json"))
    if metric_files:
        for mf in metric_files:
            with mf.open("r", encoding="utf-8") as f:
                st.json(json.load(f))
    else:
        st.info("Multimodal metrics JSON will appear here after evaluation.")

# ── Dataset Explorer tab ─────────────────────────────────────────────────────

with data_tab:
    st.subheader("DAIC-WOZ Dataset Explorer")
    summary = compute_dataset_summary(cfg)
    col1, col2, col3 = st.columns(3)
    col1.metric("Audio WAV Files",       summary["audio_count"])
    col2.metric("Transcript CSV Files",  summary["transcript_count"])
    col3.metric("Label CSV Files",       len(summary["label_files"]))
    st.write("Detected label files:")
    st.write(summary["label_files"] if summary["label_files"] else "No label CSV files detected")

# ── Guide tab ────────────────────────────────────────────────────────────────

with guide_tab:
    st.subheader("How This Multimodal Project Works")
    st.markdown("""
**Audio branch:** WAV → MFCC + delta features → Conv1D → BiLSTM → Temporal Attention → audio embedding  
**Text branch:** transcript → SentenceTransformer (all-MiniLM-L6-v2) → Dense layers → text embedding  
**Fusion:** audio + text embeddings → Dense(128) → Dense(64) → sigmoid → depression probability

**Training (run on Kaggle with GPU):**
```bash
python train_multimodal_pt.py
```
Then download `artifacts/models/best_multimodal_pt.pth` and upload it to your GitHub repo.

**Local run:**
```bash
streamlit run app_multimodal.py
```

For best demo results, record 20–30 seconds of audio and type/correct the transcript.
""")
