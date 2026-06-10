# 🧠 Automated Depression Detection from Speech and Text

> **Disclaimer:** This project is for academic, research, and placement demonstration purposes only. It is **not** a clinical diagnostic tool and must **not** replace professional mental health assessment.

A full-suite depression detection system trained on the **DAIC-WOZ (AVEC 2017)** dataset. The app combines **classical machine learning on acoustic features** with a **multimodal deep learning model (CNN-BiLSTM + Attention + NLP)** into a single Streamlit dashboard.

🔗 **Live App:** [https://depression-detection-appp.streamlit.app](https://depression-detection-appp.streamlit.app)

---

## 📌 Table of Contents

- [App Overview](#-app-overview)
- [Module 1 — Classical ML Models](#-module-1--classical-ml-models)
- [Module 2 — Multimodal Deep Learning](#-module-2--multimodal-deep-learning)
- [Why Two Approaches?](#-why-two-approaches)
- [Dataset](#-dataset)
- [Model Results](#-model-results)
- [Folder Structure](#-folder-structure)
- [Setup & Installation](#-setup--installation)
- [Training the Models](#-training-the-models)
- [Running the App](#-running-the-app)
- [Deployment](#-deployment)
- [Explainability](#-explainability)
- [Practical Limitations](#-practical-limitations)

---

## 🖥️ App Overview

The Streamlit app has **three modules** selectable from the sidebar:

| Module | Description |
|---|---|
| 🏠 **Home** | Project overview, comparison of both approaches, how-to-use guide |
| 🤖 **Classical ML Models** | Upload or record audio → predictions from 5 traditional ML models with comparison charts |
| 🔮 **Multimodal (LSTM + Attention)** | Upload or record audio + transcript → deep learning prediction with attention heatmap and Grad-CAM |

## 🤖 Module 1 — Classical ML Models

Compares **5 traditional ML models** trained on a **194-dimensional acoustic feature vector** extracted from each audio file.

### Features Extracted

| Feature Group | Count |
|---|---|
| MFCC mean values | 40 |
| MFCC standard deviations | 40 |
| Delta MFCC mean values | 40 |
| Delta MFCC standard deviations | 40 |
| Spectral centroid (mean + std) | 2 |
| Spectral bandwidth (mean + std) | 2 |
| Spectral rolloff (mean + std) | 2 |
| Zero crossing rate (mean + std) | 2 |
| RMS energy (mean + std) | 2 |
| Chroma mean values | 12 |
| Chroma standard deviations | 12 |
| **Total** | **194** |

### Preprocessing

1. Load audio with `librosa` at 16 kHz
2. Trim/pad and validate minimum length
3. Extract 194-dimensional feature vector
4. Standardise with `StandardScaler`
5. Apply **SMOTE** for class balancing during training

### Models

| Model | Description |
|---|---|
| **Logistic Regression** | Linear baseline; interpretable and fast |
| **SVM (RBF)** ⭐ best | Non-linear kernel SVM; best AUC on test set |
| **Random Forest** | 200-tree ensemble; robust, shows feature importance |
| **XGBoost** | Gradient boosting; strong on tabular/imbalanced data |
| **MLP** | Shallow neural net; bridges classical ML and deep learning |

### App Tabs (Classical ML Module)

- **Predict** — Record voice or upload audio file; see per-model probability and a cross-model comparison chart
- **Model Comparison** — Side-by-side CV and test metrics for all 5 models
- **Training Results** — Confusion matrices, ROC curves, and accuracy plots
- **About** — Feature description and dataset notes

---

## 🔮 Module 2 — Multimodal Deep Learning

A **PyTorch** model combining audio and transcript text for richer depression detection.

### Architecture

```
WAV file                          Transcript / Whisper ASR
     ↓                                       ↓
MFCC extraction              SentenceTransformer embeddings
     ↓                                       ↓
  Conv1D                         Text projection layer
  BiLSTM
Temporal Attention
     ↓                                       ↓
              Concatenation (audio + text fusion)
                           ↓
                  Dense layers + Dropout
                           ↓
                Sigmoid → Depression probability
```

### Key Features

- **Audio CNN-BiLSTM + Attention** — local pattern extraction, long-range temporal modelling, learned frame weighting
- **SentenceTransformer (NLP)** — semantic embeddings from interview transcript text
- **Whisper auto-transcription** — automatically transcribes uploaded audio if no manual transcript is provided
- **Live audio recording** — record directly in the browser
- **Attention weight visualisation** — shows which speech frames influenced the prediction
- **Grad-CAM time heatmap** — Conv1D layer importance over time frames

### App Tabs (Multimodal Module)

- **Prediction** — Upload/record audio + optional transcript input; Whisper fills transcript if blank
- **Multimodal Dashboard** — Attention heatmap, Grad-CAM plot, MFCC visualisation, probability breakdown
- **Dataset Explorer** — Browse DAIC-WOZ sample statistics and label distribution
- **Guide** — How to use the multimodal module effectively

---

## ⚖️ Why Two Approaches?

| | Classical ML | Multimodal Deep Learning |
|---|---|---|
| **Dataset size needed** | Small (100–200 samples) ✅ | Large (1000+ samples) ⚠️ |
| **Interpretability** | High (feature importance) ✅ | Low (black box) |
| **Input** | Audio only | Audio + transcript text |
| **Best model AUC** | 0.873 (SVM) | Depends on training |
| **Speed** | Fast inference | Slower (deep model) |
| **Overfitting risk** | Low | Higher on small data |

The classical ML module demonstrates that well-engineered acoustic features with appropriate models outperform deep learning on small datasets like DAIC-WOZ. The multimodal module shows the direction for future work with larger datasets.

---

## 📂 Dataset

The project uses the **DAIC-WOZ (AVEC 2017)** dataset. Place files in this structure:

```
DAIC_WOZ/
└── data/
    ├── audio/
    │   ├── 300.wav
    │   ├── 301.wav
    │   └── ...
    ├── transcripts/
    │   ├── 300_TRANSCRIPT.csv
    │   └── ...
    └── labels/
        ├── train_split_Depression_AVEC2017.csv
        ├── dev_split_Depression_AVEC2017.csv
        └── test_split_Depression_AVEC2017.csv
```

**Label definition:** PHQ-8 score ≥ 10 → `Depressed`

> The full dataset (~5.75 GB) is **not** included. Download from [Kaggle](https://www.kaggle.com/datasets/gunjangggupta/daic-woz) or use the provided script after setting up `kaggle.json`:
> ```bash
> python scripts/download_dataset.py
> ```

---

## 📊 Model Results

### Classical ML — Cross-validation and Test Metrics

| Model | CV Accuracy | CV F1 | CV AUC | Test Accuracy | Test F1 | Test AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.762 ± 0.055 | 0.790 ± 0.049 | 0.751 ± 0.080 | 0.775 | 0.791 | 0.780 |
| **SVM (RBF)** ⭐ | **0.839 ± 0.082** | **0.845 ± 0.075** | **0.932 ± 0.027** | **0.825** | **0.829** | **0.873** |
| Random Forest | 0.783 ± 0.051 | 0.795 ± 0.050 | 0.885 ± 0.063 | 0.775 | 0.757 | 0.820 |
| XGBoost | 0.773 ± 0.092 | 0.783 ± 0.081 | 0.876 ± 0.068 | 0.825 | 0.811 | 0.862 |
| MLP | 0.848 ± 0.057 | 0.857 ± 0.047 | 0.901 ± 0.080 | 0.700 | 0.700 | 0.793 |

Full metrics are in `plots/results_summary.csv`.

---

## 🗂️ Folder Structure

```
depression-detection/
├── .devcontainer/
│   └── devcontainer.json
├── comparison/                              # Classical ML module
│   ├── models/
│   │   ├── logistic_regression.pkl
│   │   ├── mlp.pkl
│   │   ├── random_forest.pkl
│   │   ├── scaler.pkl
│   │   ├── svm_(rbf).pkl
│   │   └── xgboost.pkl
│   ├── plots/
│   │   ├── comparison_bar.png
│   │   ├── confusion_matrix.png
│   │   ├── feature_importance.png
│   │   ├── mfcc_heatmap.png
│   │   ├── mlp_curves.png
│   │   ├── results_summary.csv
│   │   └── roc_curves.png
│   ├── app.py                               # Classical ML Streamlit app
│   └── requirements.txt
├── multimodal/                              # Multimodal deep learning module
│   └── Final Multimodal Depression Detection model/
│       └── automated_depression_detection_daic_woz/
│           ├── artifacts/
│           │   ├── models/
│           │   │   ├── best_attention_lstm.keras
│           │   │   ├── best_multimodal_attention_lstm.keras
│           │   │   ├── best_multimodal_pt.pth          # PyTorch model (primary)
│           │   │   ├── feature_scaler.joblib
│           │   │   └── preprocess_config.json
│           │   ├── plots/
│           │   │   ├── accuracy_curve.png
│           │   │   ├── auc_curve.png
│           │   │   ├── class_distribution_train.png
│           │   │   ├── confusion_matrix.png
│           │   │   ├── example_attention_weights.png
│           │   │   ├── example_gradcam_1d.png
│           │   │   ├── example_mfcc_heatmap.png
│           │   │   ├── loss_curve.png
│           │   │   ├── multimodal_class_distribution_train.png
│           │   │   ├── multimodal_confusion_matrix.png
│           │   │   ├── multimodal_example_attention_weights.png
│           │   │   ├── multimodal_example_gradcam_1d.png
│           │   │   ├── multimodal_example_mfcc_heatmap.png
│           │   │   ├── multimodal_prediction_probability_distribution.png
│           │   │   ├── multimodal_roc_curve.png
│           │   │   ├── prediction_probability_distribution.png
│           │   │   ├── roc_curve.png
│           │   │   └── training_history.csv
│           │   └── reports/
│           │       ├── dev_classification_report.txt
│           │       ├── dev_metrics.json
│           │       ├── dev_multimodal_classification_report.txt
│           │       ├── dev_multimodal_metrics.json
│           │       ├── dev_multimodal_predictions.csv
│           │       ├── dev_predictions.csv
│           │       └── training_summary.json
│           ├── configs/
│           │   └── config.yaml
│           ├── pytorch_src/                 # PyTorch multimodal pipeline
│           │   ├── __init__.py
│           │   ├── app_multimodal_pt.py
│           │   ├── inference_multimodal_pt.py
│           │   ├── multimodal_model_pt.py
│           │   └── train_multimodal_pt.py
│           ├── scripts/
│           │   ├── download_dataset.py
│           │   ├── quick_multimodal_check.py
│           │   ├── train_pipeline.py
│           │   └── verify_dataset.py
│           ├── src/                         # Keras/TF pipeline
│           │   ├── __init__.py
│           │   ├── audio_features.py
│           │   ├── config.py
│           │   ├── data_utils.py
│           │   ├── evaluate.py
│           │   ├── evaluate_multimodal.py
│           │   ├── explainability.py
│           │   ├── inference.py
│           │   ├── inference_multimodal.py
│           │   ├── inference_multimodal_pt.py
│           │   ├── model.py
│           │   ├── multimodal_data.py
│           │   ├── multimodal_explainability.py
│           │   ├── multimodal_model.py
│           │   ├── multimodal_model_pt.py
│           │   ├── text_utils.py
│           │   ├── train.py
│           │   ├── train_multimodal.py
│           │   ├── transcript_utils.py
│           │   └── visualization.py
│           ├── app.py                       # Audio-only Streamlit app
│           ├── app_multimodal.py            # Multimodal Streamlit app
│           ├── DEPLOYMENT.md
│           ├── multimodal_explaination.md
│           ├── MULTIMODAL_UPGRADE_README.md
│           ├── packages.txt
│           └── requirements.txt
├── .gitignore
├── .python-version
├── app_combined.py                          # ← Main entry point (full suite)
├── packages.txt
├── README.md
└── requirements.txt
```


## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/gunjanguptaa/DepressionDetection.git
cd DepressionDetection
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install PyTorch (CPU)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 5. Install multimodal-specific packages

```bash
pip install sentence-transformers transformers faster-whisper
```

### 6. Verify dataset

```bash
python scripts/verify_dataset.py
```

---

## 🏋️ Training the Models

### Classical ML models

```bash
python -m src.train
```

Saved to `models/` as `.pkl` files.

### Multimodal PyTorch model

```bash
python -m pytorch_src.train_multimodal_pt
```

Saved to `artifacts/models/best_multimodal_pt.pth`.

### One-command pipeline

```bash
python scripts/train_pipeline.py
```

Runs dataset verification → training → evaluation → plot generation in sequence.

---

## 🖥️ Running the App

### Full suite (recommended)

```bash
streamlit run app_combined.py
```

### Individual modules

```bash
streamlit run app_multimodal.py   # Multimodal only
streamlit run app.py              # Audio/Classical ML only
```

---

## 🚀 Deployment

The app is live on **Streamlit Community Cloud:**
🔗 [https://depression-detection-appp.streamlit.app](https://depression-detection-appp.streamlit.app)

### To redeploy / fork

1. Push this repository to GitHub with trained artifacts in `models/` and `artifacts/models/`.
2. **Do not push the 5.75 GB DAIC-WOZ dataset.** Train locally or in Colab/Kaggle, then commit only the model files.
3. On [Streamlit Cloud](https://streamlit.io/cloud), set the main file as `app_combined.py`.

**Required model artifacts:**

```
models/logistic_regression.pkl
models/svm_(rbf).pkl
models/random_forest.pkl
models/xgboost.pkl
models/mlp.pkl
models/scaler.pkl
artifacts/models/best_multimodal_pt.pth
artifacts/models/best_attention_lstm.keras
artifacts/models/feature_scaler.joblib
artifacts/models/preprocess_config.json
```

> If `.pth` or `.keras` files exceed GitHub's 100 MB limit, use [Git LFS](https://git-lfs.github.com/) or host on HuggingFace Hub and download at startup.

---

## 🔍 Explainability

| Method | Module | Description |
|---|---|---|
| Cross-model probability comparison | Classical ML | Bar chart comparing all 5 model outputs for a given audio input |
| Feature importance | Classical ML (RF/XGBoost) | Top acoustic features contributing to the prediction |
| Temporal Attention Weights | Multimodal | Which speech frames the model focused on |
| Grad-CAM (Conv1D) | Multimodal | Time-frame importance heatmap over the Conv1D layer |
| Input Gradient Saliency | Multimodal | MFCC dimensions most influential for the prediction |
| MFCC / Log-Mel Heatmaps | Both | Raw feature visualisations for each audio sample |
| Waveform + Mel Spectrogram | Classical ML | Visual diagnostic of the uploaded audio |

---

## ⚠️ Practical Limitations

- Classical ML models were trained on a relatively small cohort; generalisation to other populations is not guaranteed.
- Depression is a complex condition that cannot be reliably detected from voice or text alone.
- The multimodal deep learning model is best suited for larger datasets; on DAIC-WOZ it may overfit.
- Grad-CAM is adapted for 1D convolutions over time frames — it is an interpretability visualisation, not a clinical explanation.
- No clinical validation has been performed. This system must not be used for any real-world diagnosis.



## 📄 License

For academic and research use only. The DAIC-WOZ dataset is subject to its own usage agreement — review it at [dcapswoz.ict.usc.edu](https://dcapswoz.ict.usc.edu) before using or distributing this project.
