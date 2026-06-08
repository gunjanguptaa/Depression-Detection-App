# ============================================================
# Combined Depression Detection App
# Classical ML (5 models) + Multimodal (LSTM + Attention)
# ============================================================

import streamlit as st
import sys
import os

st.set_page_config(
    page_title="Depression Detection — Full Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── SIDEBAR NAVIGATION ───────────────────────────────────────
with st.sidebar:
    st.title("🧠 Depression Detection")
    st.divider()

    mode = st.radio(
        "Select Module",
        [
            "🏠 Home",
            "📊 Classical ML Models",
            "🤖 Multimodal (LSTM + Attention)",
        ],
        index=0
    )
    st.divider()
    st.caption("Classical ML: SVM, RF, XGBoost, LR, MLP")
    st.caption("Multimodal: CNN-BiLSTM-Attention + NLP")

# ── HOME PAGE ────────────────────────────────────────────────
if mode == "🏠 Home":
    st.title("🧠 Automated Depression Detection from Speech and Text")
    st.markdown("### DAIC-WOZ Dataset")

    st.markdown("""
    <div style='background:#EFF6FF;padding:20px;border-radius:12px;
                border-left:5px solid #3B82F6;margin-bottom:20px'>
    ⚠️ <strong>Disclaimer:</strong> This is an academic research prototype only.
    It is <strong>NOT</strong> a clinical diagnostic tool and should never replace
    professional mental health assessment.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 📊 Module 1 — Classical ML
        Compare 5 traditional machine learning models trained on
        194-dimensional acoustic features (MFCCs, chroma, spectral).

        **Models:**
        - Logistic Regression (baseline)
        - SVM with RBF kernel ⭐ best
        - Random Forest
        - XGBoost
        - MLP (shallow neural net)

        **Results:**
        - Best AUC: **0.873** (SVM RBF)
        - Features: MFCC + delta + spectral
        - Preprocessing: StandardScaler + SMOTE
        """)

    with col2:
        st.markdown("""
        ### 🤖 Module 2 — Multimodal
        Deep learning model combining audio and transcript text
        for richer depression detection.

        **Architecture:**
        - Audio: CNN → BiLSTM → Attention
        - Text: SentenceTransformer embeddings
        - Fusion: Audio + Text → Dense → Prediction

        **Features:**
        - Attention weight visualization
        - Grad-CAM time heatmap
        - Whisper auto-transcription
        - Live audio recording support
        """)

    st.divider()

    st.markdown("### 📈 Why Two Approaches?")
    st.markdown("""
    | | Classical ML | Multimodal Deep Learning |
    |---|---|---|
    | **Dataset size needed** | Small (100–200 samples) ✅ | Large (1000+ samples) ⚠️ |
    | **Interpretability** | High (feature importance) ✅ | Low (black box) |
    | **Input** | Audio only | Audio + transcript text |
    | **Best model AUC** | 0.873 (SVM) | Depends on training |
    | **Speed** | Fast inference | Slower (deep model) |
    | **Overfitting risk** | Low | Higher on small data |

    The classical ML module demonstrates that well-engineered acoustic features
    with appropriate models outperform deep learning on small datasets like DAIC-WOZ.
    The multimodal module shows the direction for future work with larger datasets.
    """)

    st.divider()
    st.markdown("### 🚀 How to Use")
    col_a, col_b = st.columns(2)
    with col_a:
        st.info("👈 Select **Classical ML Models** to upload audio and get predictions from all 5 models with comparison charts.")
    with col_b:
        st.info("👈 Select **Multimodal** to use the deep learning model with attention visualization and Whisper transcription.")

# ── MODULE 1: CLASSICAL ML ───────────────────────────────────
elif mode == "📊 Classical ML Models":
    # Add comparison project to path
    comparison_path = os.path.join(os.path.dirname(__file__), 'comparison')
    if comparison_path not in sys.path:
        sys.path.insert(0, comparison_path)

    # Change working directory so relative paths (models/, plots/) work
    original_dir = os.getcwd()
    os.chdir(comparison_path)

    try:
        # Dynamically load and execute the comparison app
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "comparison_app",
            os.path.join(comparison_path, "app.py")
        )
        module = importlib.util.module_from_spec(spec)

        # Override set_page_config since we already called it
        import unittest.mock as mock
        with mock.patch('streamlit.set_page_config'):
            spec.loader.exec_module(module)
    except Exception as e:
        st.error(f"Error loading Classical ML module: {e}")
        st.exception(e)
    finally:
        os.chdir(original_dir)

# ── MODULE 2: MULTIMODAL ─────────────────────────────────────
elif mode == "🤖 Multimodal (LSTM + Attention)":
    multimodal_path = os.path.dirname(os.path.abspath(__file__))

    if multimodal_path not in sys.path:
        sys.path.insert(0, multimodal_path)

    original_dir = os.getcwd()
    os.chdir(multimodal_path)

    # Set environment variable so ffmpeg is found regardless of context
    ffmpeg_path = r"C:\ffmpeg\bin"
    if ffmpeg_path not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ.get("PATH", "")

    try:
        import importlib.util
        import unittest.mock as mock

        spec = importlib.util.spec_from_file_location(
            "multimodal_app",
            os.path.join(multimodal_path, "app_multimodal.py")
        )
        module = importlib.util.module_from_spec(spec)

        with mock.patch('streamlit.set_page_config'):
            spec.loader.exec_module(module)

    except Exception as e:
        st.error(f"Error loading Multimodal module: {e}")
        st.exception(e)
    finally:
        os.chdir(original_dir)
