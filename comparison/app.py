# ============================================================
# Streamlit App — Depression Detection from Speech
# ============================================================
# Deploy: streamlit run app.py
# Requires: models/ folder with .pkl files from training
# ============================================================

import streamlit as st
import numpy as np
import librosa
import joblib
import os
import tempfile
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics import roc_curve
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title='Depression Detection from Speech',
    page_icon='🧠',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ── CONSTANTS ────────────────────────────────────────────────
MODEL_DIR   = 'models'   # folder containing .pkl files saved from training
N_MFCC      = 40
HOP_LENGTH  = 512
N_FFT       = 2048
SR          = 16000

MODEL_FILES = {
    'Logistic Regression': os.path.join(MODEL_DIR, 'logistic_regression.pkl'),
    'SVM (RBF)':           os.path.join(MODEL_DIR, 'svm_(rbf).pkl'),
    'Random Forest':       os.path.join(MODEL_DIR, 'random_forest.pkl'),
    'XGBoost':             os.path.join(MODEL_DIR, 'xgboost.pkl'),
    'MLP':                 os.path.join(MODEL_DIR, 'mlp.pkl'),
}
SCALER_PATH  = os.path.join(MODEL_DIR, 'scaler.pkl')
PLOT_DIR     = 'plots'   # optional: pre-generated plots from training

FEAT_NAMES = (
    [f'MFCC {i+1} mean' for i in range(N_MFCC)] +
    [f'MFCC {i+1} std'  for i in range(N_MFCC)] +
    [f'ΔMFCC {i+1} mean' for i in range(N_MFCC)] +
    [f'ΔMFCC {i+1} std'  for i in range(N_MFCC)] +
    ['Centroid μ','Centroid σ','Bandwidth μ','Bandwidth σ',
     'Rolloff μ','Rolloff σ','ZCR μ','ZCR σ','RMS μ','RMS σ'] +
    [f'Chroma {i+1} μ' for i in range(12)] +
    [f'Chroma {i+1} σ' for i in range(12)]
)

# ── CACHE: LOAD MODELS ───────────────────────────────────────
@st.cache_resource
def load_models():
    models, scaler = {}, None
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    for name, path in MODEL_FILES.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            st.warning(f'Model not found: {path}')
    return models, scaler

# ── FEATURE EXTRACTION ───────────────────────────────────────
def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=SR, mono=True,
                             duration=120, res_type='kaiser_fast')
    except Exception as e:
        return None, str(e)

    if len(y) < sr:
        return None, 'Audio too short (< 1 second).'

    feats = []

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                  n_fft=N_FFT, hop_length=HOP_LENGTH)
    feats.extend(np.mean(mfccs, axis=1).tolist())
    feats.extend(np.std(mfccs, axis=1).tolist())

    delta = librosa.feature.delta(mfccs)
    feats.extend(np.mean(delta, axis=1).tolist())
    feats.extend(np.std(delta, axis=1).tolist())

    for fn in [librosa.feature.spectral_centroid,
               librosa.feature.spectral_bandwidth,
               librosa.feature.spectral_rolloff]:
        f = fn(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH)
        feats += [float(np.mean(f)), float(np.std(f))]

    for fn in [librosa.feature.zero_crossing_rate, librosa.feature.rms]:
        f = fn(y=y, hop_length=HOP_LENGTH)
        feats += [float(np.mean(f)), float(np.std(f))]

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT,
                                          hop_length=HOP_LENGTH)
    feats.extend(np.mean(chroma, axis=1).tolist())
    feats.extend(np.std(chroma, axis=1).tolist())

    return np.array(feats, dtype=np.float32), None


def predict_on_features(feat_vec, models, scaler, selected_model):
    feat_scaled = scaler.transform(feat_vec.reshape(1, -1)) if scaler else feat_vec.reshape(1, -1)
    model = models[selected_model]
    prob  = model.predict_proba(feat_scaled)[0]
    pred  = int(np.argmax(prob))
    return pred, prob, feat_scaled


# ── PLOTS ────────────────────────────────────────────────────
def plot_probability_gauge(prob_depressed):
    fig, ax = plt.subplots(figsize=(5, 2.5))
    bar_color = '#EF4444' if prob_depressed > 0.5 else '#10B981'
    ax.barh([''], [prob_depressed], color=bar_color, height=0.4, alpha=0.85)
    ax.barh([''], [1.0], color='#E5E7EB', height=0.4, alpha=0.4)
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color='gray', lw=1, ls='--', alpha=0.6)
    ax.set_xlabel('Probability', fontsize=10)
    ax.set_title(f'Depression probability: {prob_depressed:.1%}',
                 fontsize=11, fontweight='bold')
    ax.text(prob_depressed + 0.02, 0, f'{prob_depressed:.1%}',
            va='center', fontsize=10, color='black')
    fig.tight_layout()
    return fig


def plot_mfcc_heatmap(feat_vec):
    mfcc_mean = feat_vec[:N_MFCC]
    mfcc_std  = feat_vec[N_MFCC:2*N_MFCC]
    data = np.vstack([mfcc_mean, mfcc_std])

    fig, ax = plt.subplots(figsize=(12, 3))
    sns.heatmap(
        data, ax=ax, cmap='RdYlGn_r',
        xticklabels=[str(i+1) for i in range(N_MFCC)],
        yticklabels=['Mean', 'Std'],
        linewidths=0.3, linecolor='white',
        annot=False, cbar_kws={'shrink': 0.7}
    )
    ax.set_title('MFCC Feature Heatmap (coefficients 1–40)',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('MFCC Coefficient Index')
    fig.tight_layout()
    return fig


def plot_feature_bar(feat_vec, top_k=20):
    abs_feats = np.abs(feat_vec)
    top_idx   = np.argsort(abs_feats)[-top_k:][::-1]
    top_vals  = feat_vec[top_idx]
    top_names = [FEAT_NAMES[i] if i < len(FEAT_NAMES) else f'feat_{i}'
                 for i in top_idx]

    colors = ['#EF4444' if v > 0 else '#3B82F6' for v in top_vals]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(top_k), top_vals[::-1],
            color=colors[::-1], alpha=0.85, edgecolor='none')
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top_names[::-1], fontsize=8)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_title(f'Top {top_k} Feature Values (scaled)',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Scaled Value')
    fig.tight_layout()
    return fig


def plot_all_models_comparison(feat_vec, models, scaler):
    names, dep_probs = [], []
    feat_scaled = scaler.transform(feat_vec.reshape(1, -1)) if scaler else feat_vec.reshape(1, -1)

    for name, model in models.items():
        prob = model.predict_proba(feat_scaled)[0][1]
        names.append(name)
        dep_probs.append(prob)

    colors = ['#EF4444' if p > 0.5 else '#10B981' for p in dep_probs]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, dep_probs, color=colors, alpha=0.85, edgecolor='none')
    ax.axhline(0.5, color='gray', lw=1, ls='--', alpha=0.7,
               label='Decision threshold (0.5)')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('P(Depressed)', fontsize=10)
    ax.set_title('Depression Probability Across All Models',
                 fontsize=11, fontweight='bold')
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    for bar, p in zip(bars, dep_probs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.02, f'{p:.2f}',
                ha='center', va='bottom', fontsize=9)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_waveform_and_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=SR, mono=True, duration=60)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))

    t = np.linspace(0, len(y) / sr, len(y))
    axes[0].plot(t, y, color='#3B82F6', lw=0.5, alpha=0.8)
    axes[0].set_title('Waveform', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.2)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=N_FFT)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, sr=sr, hop_length=HOP_LENGTH,
                                   x_axis='time', y_axis='mel', ax=axes[1],
                                   cmap='magma')
    axes[1].set_title('Mel Spectrogram', fontsize=11, fontweight='bold')
    fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
    fig.tight_layout()
    return fig


# ── CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
.stApp { max-width: 1200px; margin: auto; }
.result-box {
    padding: 20px; border-radius: 12px; margin: 16px 0;
    text-align: center; font-size: 1.4rem; font-weight: 600;
}
.depressed    { background: #FEE2E2; color: #991B1B; border: 2px solid #F87171; }
.not-depressed{ background: #D1FAE5; color: #065F46; border: 2px solid #34D399; }
.disclaimer   { background: #FEF3C7; color: #92400E; padding: 12px;
                border-radius: 8px; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ───────────────────────────────────────────────────
st.title('🧠 Depression Detection from Speech')
st.markdown('DAIC-WOZ Dataset | Acoustic Feature Analysis')
st.markdown("""
<div class="disclaimer">
⚠️ <strong>Disclaimer:</strong> This tool is for academic/research purposes only and is
<strong>NOT</strong> a clinical diagnostic tool. It should not replace professional
mental health assessment.
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.header('⚙️ Settings')
    models, scaler = load_models()

    if not models:
        st.error('No models found! Train models first and place .pkl files in models/ folder.')
        st.stop()

    selected_model = st.selectbox(
        'Select model',
        options=list(models.keys()),
        index=2  # default: Random Forest
    )
    st.success(f'✅ Model loaded: **{selected_model}**')

    st.divider()
    st.header('📊 About the Models')
    model_info = {
        'Logistic Regression': 'Linear baseline. Fast, interpretable. Good for linearly separable features.',
        'SVM (RBF)':           'Kernel trick maps features to higher dimensions. Excellent on small datasets.',
        'Random Forest':       'Ensemble of 200 trees. Robust, less prone to overfitting. Shows feature importance.',
        'XGBoost':             'Gradient boosting — typically best accuracy. Handles imbalanced data well.',
        'MLP':                 'Shallow neural network (128→64→32). Bridges classical and deep learning.',
    }
    st.info(model_info.get(selected_model, ''))

    st.divider()
    st.markdown('**Dataset:** DAIC-WOZ (AVEC 2017)')
    st.markdown('**Label:** PHQ-8 ≥ 10 → Depressed')
    st.markdown('**Features:** 194-dim acoustic vector')
    st.markdown('**Preprocessing:** StandardScaler + SMOTE')

# ── MAIN TABS ────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    '🎤 Predict', '📊 Model Comparison', '📈 Training Results', 'ℹ️ About'
])

# ════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════════
with tab1:
    st.subheader('Analyse a Speech Sample')
    input_method = st.radio(
        'Input method', ['🎤 Record voice', '📁 Upload audio file'],
        horizontal=True
    )

    audio_path = None

    if input_method == '🎤 Record voice':
        st.markdown("""
        **Instructions:** Click *Start Recording*, speak for at least 30–60 seconds,
        then click *Stop*. Answer questions like:
        - *"How have you been feeling lately?"*
        - *"Tell me about your sleep and energy levels."*
        - *"What activities do you enjoy?"*
        """)
        audio_data = st.audio_input('Record your voice')
        if audio_data is not None:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp.write(audio_data.getvalue())
                audio_path = tmp.name
            st.success('✅ Recording captured!')

    else:
        uploaded = st.file_uploader(
            'Upload audio file (WAV, MP3, FLAC, OGG)',
            type=['wav', 'mp3', 'flac', 'ogg', 'm4a']
        )
        if uploaded is not None:
            with tempfile.NamedTemporaryFile(suffix='.' + uploaded.name.split('.')[-1],
                                            delete=False) as tmp:
                tmp.write(uploaded.read())
                audio_path = tmp.name
            st.audio(uploaded)
            st.success(f'✅ Uploaded: {uploaded.name}')

    # ── ANALYSE BUTTON ───────────────────────────────────────
    if audio_path and st.button('🔍 Analyse for Depression Signs', type='primary'):
        with st.spinner('Extracting acoustic features...'):
            feat_vec, err = extract_features(audio_path)

        if err:
            st.error(f'Feature extraction failed: {err}')
        else:
            with st.spinner('Running model inference...'):
                pred, prob, feat_scaled = predict_on_features(
                    feat_vec, models, scaler, selected_model
                )

            dep_prob    = prob[1]
            label       = 'Depressed' if pred == 1 else 'Not Depressed'
            box_class   = 'depressed' if pred == 1 else 'not-depressed'
            icon        = '⚠️' if pred == 1 else '✅'

            # Result box
            st.markdown(f"""
            <div class="result-box {box_class}">
                {icon} Prediction: <strong>{label}</strong><br>
                <span style="font-size:1rem; font-weight:400;">
                    P(Depressed) = {dep_prob:.1%} &nbsp;|&nbsp;
                    P(Not Depressed) = {prob[0]:.1%}
                </span>
            </div>
            """, unsafe_allow_html=True)

            # Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric('P(Depressed)',     f'{dep_prob:.1%}')
            col2.metric('P(Not Depressed)', f'{prob[0]:.1%}')
            col3.metric('Model',            selected_model)

            st.divider()

            # Plots
            col_a, col_b = st.columns(2)

            with col_a:
                st.subheader('🎯 Probability gauge')
                fig_gauge = plot_probability_gauge(dep_prob)
                st.pyplot(fig_gauge, use_container_width=True)
                plt.close()

            with col_b:
                st.subheader('🔬 All models comparison')
                fig_cmp = plot_all_models_comparison(feat_vec, models, scaler)
                st.pyplot(fig_cmp, use_container_width=True)
                plt.close()

            st.subheader('🌊 Waveform & Mel Spectrogram')
            try:
                fig_wave = plot_waveform_and_spectrogram(audio_path)
                st.pyplot(fig_wave, use_container_width=True)
                plt.close()
            except Exception as e:
                st.warning(f'Could not plot waveform: {e}')

            st.subheader('🔥 MFCC Feature Heatmap')
            fig_heatmap = plot_mfcc_heatmap(feat_vec)
            st.pyplot(fig_heatmap, use_container_width=True)
            plt.close()

            st.subheader('📊 Top Feature Values (scaled)')
            fig_bar = plot_feature_bar(feat_scaled[0])
            st.pyplot(fig_bar, use_container_width=True)
            plt.close()

# ════════════════════════════════════════════════════════════
# TAB 2 — MODEL COMPARISON
# ════════════════════════════════════════════════════════════
with tab2:
    st.subheader('Model Comparison — Training Results')
    st.markdown("""
    Pre-generated charts.
    """)

    plot_files = {
        'Comparison Bar (CV)':    'comparison_bar.png',
        'ROC Curves':             'roc_curves.png',
        'Confusion Matrix':       'confusion_matrix.png',
        'Feature Importance':     'feature_importance.png',
        'MFCC Heatmap':           'mfcc_heatmap.png',
        'MLP Loss Curve':         'mlp_curves.png',
    }
    for title, fname in plot_files.items():
        path = os.path.join(PLOT_DIR, fname)
        if os.path.exists(path):
            st.image(path, caption=title, use_container_width=True)
        else:
            st.info(f'📂 Plot not found: {path}  — run training notebook first.')

# ════════════════════════════════════════════════════════════
# TAB 3 — TRAINING RESULTS TABLE
# ════════════════════════════════════════════════════════════
with tab3:
    st.subheader('📋 Results Summary Table')
    csv_path = os.path.join(PLOT_DIR, 'results_summary.csv')
    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path, index_col=0)
        st.dataframe(df, use_container_width=True)
        st.download_button('⬇️ Download CSV', df.to_csv(),
                           file_name='results_summary.csv', mime='text/csv')
    else:
        st.info('No results CSV found. Run training notebook and copy results_summary.csv to plots/ folder.')

    st.subheader('🏗️ Architecture Overview')
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Feature Vector (194 dims):**
        - MFCC 1–40 mean (40 dims)
        - MFCC 1–40 std (40 dims)
        - Delta-MFCC mean (40 dims)
        - Delta-MFCC std (40 dims)
        - Spectral centroid μ/σ
        - Spectral bandwidth μ/σ
        - Spectral rolloff μ/σ
        - ZCR μ/σ
        - RMS energy μ/σ
        - Chroma 1–12 mean + std (24 dims)
        """)
    with col2:
        st.markdown("""
        **Training Pipeline:**
        1. Load DAIC-WOZ audio (.wav)
        2. Extract 194-dim feature vector
        3. StandardScaler normalisation
        4. SMOTE oversampling (class balance)
        5. 5-Fold Stratified CV
        6. Final fit on full data
        7. Evaluation: Acc, F1, ROC-AUC
        8. Save `.pkl` model files
        """)

# ════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ════════════════════════════════════════════════════════════
with tab4:
    st.subheader('About This Project')
    st.markdown("""
    ### Automated Depression Detection from Speech

    **Dataset:** DAIC-WOZ (Distress Analysis Interview Corpus — Wizard of Oz)
    - Collected by USC Institute for Creative Technologies
    - Semi-structured clinical interviews in English
    - Ground truth: PHQ-8 depression scale
    - ~189 participants; label imbalanced (~33% depressed)

    **Why classical ML (not LSTM/Attention)?**
    - DAIC-WOZ has only ~189 samples — deep learning models need thousands
    - LSTM/Attention on ~130 training samples leads to severe overfitting
    - Classical models with engineered features generalise far better here
    - SMOTE handles the class imbalance without data augmentation risks

    **Models used and why:**

    | Model | Why chosen |
    |---|---|
    | Logistic Regression | Linear baseline, fast, interpretable |
    | SVM (RBF) | Excellent on small high-dim data; kernel trick |
    | Random Forest | Ensemble → less overfit; feature importance built-in |
    | XGBoost | Usually highest accuracy; handles imbalance with `scale_pos_weight` |
    | MLP (128→64→32) | Shallow NN; bridges classical and deep learning |

    **Acoustic features:**
    MFCCs (Mel-Frequency Cepstral Coefficients) are the most important features
    for depression detection. Research shows that depressed speech has lower energy,
    reduced pitch variability, slower speech rate, and altered spectral characteristics.

    **References:**
    - Gratch et al. (2014). "The Distress Analysis Interview Corpus of human and computer interviews."
    - DeVault et al. (2014). "SimSensei Kiosk: A virtual human interviewer for healthcare decision support."
    - Yang et al. (2022). "Automatic Depression Prediction using Internet of Things."
    """)