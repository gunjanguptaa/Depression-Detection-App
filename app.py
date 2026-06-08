from __future__ import annotations

import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import load_config
from src.data_utils import compute_dataset_summary
from src.inference import load_artifacts, predict_audio


st.set_page_config(
    page_title="Depression Detection from Speech",
    page_icon="🎙️",
    layout="wide",
)


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


@st.cache_resource(show_spinner=False)
def cached_load_artifacts():
    return load_artifacts()


cfg = load_config()
plots_dir = Path(cfg["paths"]["plots_dir"])
reports_dir = Path(cfg["paths"]["reports_dir"])
model_path = Path(cfg["paths"]["best_model"])

st.title("🎙️ Automated Depression Detection from Speech")
st.caption("CNN + BiLSTM + Attention mechanism trained on DAIC-WOZ-style speech features. Research/education use only; not a medical diagnosis.")

with st.sidebar:
    st.header("Project Controls")
    st.write("Dataset path expected by this project:")
    st.code("DAIC_WOZ/data/audio\nDAIC_WOZ/data/labels\nDAIC_WOZ/data/transcripts")
    st.write("Model status:")
    if model_path.exists():
        st.success("Trained model found")
    else:
        st.error("Model not trained yet")
    st.divider()
    st.info("Train first with: python -m src.train && python -m src.evaluate")

prediction_tab, dashboard_tab, data_tab, guide_tab = st.tabs(
    ["Prediction", "Model Dashboard", "Dataset Explorer", "Training Guide"]
)

with prediction_tab:
    st.subheader("Predict Depression Probability")
    col_a, col_b = st.columns(2)
    with col_a:
        uploaded_file = st.file_uploader("Option 1: Upload a WAV audio file", type=["wav", "mp3", "ogg", "flac", "m4a"])
    with col_b:
        if hasattr(st, "audio_input"):
            recorded_audio = st.audio_input("Option 2: Record audio from microphone")
        else:
            recorded_audio = None
            st.warning("Your Streamlit version does not support st.audio_input. Upgrade Streamlit or use file upload.")

    audio_source = None
    source_name = None
    if uploaded_file is not None:
        audio_source = uploaded_file.getvalue()
        source_name = uploaded_file.name
    elif recorded_audio is not None:
        audio_source = recorded_audio.getvalue()
        source_name = "recorded_audio.wav"

    if audio_source is not None:
        st.audio(audio_source, format="audio/wav")
        if not model_path.exists():
            st.error("Model file is missing. Please train the model first using `python -m src.train` and then run evaluation.")
        else:
            with st.spinner("Running feature extraction, LSTM-attention model, and explainability..."):
                cfg_loaded, model, scaler = cached_load_artifacts()
                result = predict_audio(audio_source, cfg_loaded, model, scaler)

            st.markdown(f"### Prediction for `{source_name}`")
            c1, c2, c3 = st.columns(3)
            c1.metric("Final Class", result["label"])
            c2.metric("Depressed Probability", f"{result['depressed_probability'] * 100:.2f}%")
            c3.metric("Non-depressed Probability", f"{result['non_depressed_probability'] * 100:.2f}%")

            prob_df = pd.DataFrame(
                {
                    "Class": ["Non-depressed", "Depressed"],
                    "Probability (%)": [
                        result["non_depressed_probability"] * 100,
                        result["depressed_probability"] * 100,
                    ],
                }
            )
            st.bar_chart(prob_df.set_index("Class"))

            st.divider()
            st.subheader("Audio and Feature Visualizations")
            st.pyplot(fig_waveform(result["waveform"], result["sample_rate"]))
            st.pyplot(fig_matrix(result["log_mel"], "Log-Mel Spectrogram", ylabel="Mel Bands"))
            st.pyplot(fig_matrix(result["features"].T, "MFCC + Delta + Delta-Delta Heatmap", ylabel="Feature Index"))

            st.divider()
            st.subheader("Explainability")
            st.pyplot(fig_line(result["attention"], "Attention Weights Over Time", "Attention Weight"))
            st.pyplot(fig_line(result["gradcam"], "Grad-CAM Style Time Heatmap", "Importance"))
            st.pyplot(fig_line(result["saliency"], "Input Gradient Saliency Over Time", "Saliency"))
    else:
        st.info("Upload a WAV/MP3 file or record audio to get a prediction.")

with dashboard_tab:
    st.subheader("Training and Evaluation Graphs")
    plot_files = [
        "loss_curve.png",
        "accuracy_curve.png",
        "auc_curve.png",
        "class_distribution_train.png",
        "roc_curve.png",
        "confusion_matrix.png",
        "prediction_probability_distribution.png",
        "example_mfcc_heatmap.png",
        "example_attention_weights.png",
        "example_gradcam_1d.png",
    ]
    existing = [p for p in plot_files if (plots_dir / p).exists()]
    if not existing:
        st.warning("No plots found yet. Run `python -m src.train` and `python -m src.evaluate` first.")
    else:
        for name in existing:
            st.markdown(f"#### {name.replace('_', ' ').replace('.png', '').title()}")
            st.image(str(plots_dir / name), use_container_width=True)

    st.divider()
    st.subheader("Metrics")
    metric_files = sorted(reports_dir.glob("*_metrics.json"))
    if metric_files:
        for metric_file in metric_files:
            with metric_file.open("r", encoding="utf-8") as f:
                st.json(json.load(f))
    else:
        st.info("Metrics JSON will appear here after evaluation.")

with data_tab:
    st.subheader("DAIC-WOZ Dataset Explorer")
    summary = compute_dataset_summary(cfg)
    col1, col2, col3 = st.columns(3)
    col1.metric("Audio WAV Files", summary["audio_count"])
    col2.metric("Transcript CSV Files", summary["transcript_count"])
    col3.metric("Label CSV Files", len(summary["label_files"]))

    st.write("Detected label files:")
    if summary["label_files"]:
        st.write(summary["label_files"])
    else:
        st.warning("No label CSV files detected. Place train/dev/test split CSV files in DAIC_WOZ/data/labels.")

    label_dir = Path(cfg["paths"]["label_dir"])
    for csv_file in sorted(label_dir.glob("*.csv")):
        with st.expander(f"Preview: {csv_file.name}"):
            try:
                df = pd.read_csv(csv_file)
                st.dataframe(df.head(20), use_container_width=True)
            except Exception as exc:
                st.error(f"Could not read {csv_file.name}: {exc}")

with guide_tab:
    st.subheader("How to Run This Project")
    st.markdown(
        """
1. Put the Kaggle dataset into the exact folder structure below:

```text
DAIC_WOZ/
└── data/
    ├── audio/        # 300.wav, 301.wav, ...
    ├── transcripts/  # 300_TRANSCRIPT.csv, ...
    └── labels/       # train/dev/test split CSV files
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Verify the dataset:

```bash
python scripts/verify_dataset.py
```

4. Train the Attention + LSTM model:

```bash
python -m src.train
```

5. Generate ROC curve, confusion matrix, Grad-CAM style heatmap, and reports:

```bash
python -m src.evaluate
```

6. Run the Streamlit app:

```bash
streamlit run app.py
```

The app provides upload audio and record audio options, then shows depressed/non-depressed percentages and explainability plots.
        """
    )
