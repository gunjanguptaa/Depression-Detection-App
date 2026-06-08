# Automated Depression Detection from Speech using Attention Mechanism + LSTM

This project is a complete machine-learning pipeline for **binary depression detection from speech signals** using the **DAIC-WOZ** dataset structure shown below.

> Important: This project is for academic/research/placement demonstration only. It is **not** a medical diagnostic tool and must not be used to make clinical decisions.

## Dataset Expected

The code expects the Kaggle dataset to be placed in this exact structure:

```text
DAIC_WOZ/
└── data/
    ├── audio/
    │   ├── 300.wav
    │   ├── 301.wav
    │   └── ...
    ├── transcripts/
    │   ├── 300_TRANSCRIPT.csv
    │   ├── 301_TRANSCRIPT.csv
    │   └── ...
    └── labels/
        ├── train_split_Depression_AVEC2017.csv
        ├── dev_split_Depression_AVEC2017.csv
        └── test_split_Depression_AVEC2017.csv
```

The included ZIP contains the complete code and the required dataset folders. The actual 5.75 GB WAV/CSV dataset is not bundled in this generated package because only a screenshot was provided in chat, not the full Kaggle dataset files.

## What This Project Includes

- Audio feature extraction from speech signals using MFCC, delta, and delta-delta features.
- CNN + BiLSTM + temporal attention model.
- Class imbalance handling using class weights.
- Training curves:
  - Loss curve
  - Accuracy curve
  - AUC curve
- Evaluation visualizations:
  - ROC curve
  - Confusion matrix
  - Prediction probability distribution
  - Class distribution
- Explainability:
  - Attention-weight curve over speech frames
  - Grad-CAM style heatmap over Conv1D time frames
  - Input gradient saliency
  - MFCC/log-mel heatmaps
- Streamlit app with:
  - Upload audio option
  - Record audio option
  - Depressed and non-depressed probability percentages
  - Visual dashboard for all generated plots and reports

## Quick Start

### 1. Create environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add dataset

Option A: Manually download the Kaggle dataset and place files in `DAIC_WOZ/data/` exactly as shown above.

Option B: Use Kaggle API after setting up `kaggle.json`:

```bash
python scripts/download_dataset.py
```

### 4. Verify dataset

```bash
python scripts/verify_dataset.py
```

### 5. Train model

```bash
python -m src.train
```

The best model will be saved at:

```text
artifacts/models/best_attention_lstm.keras
```

### 6. Evaluate model and generate graphs

```bash
python -m src.evaluate
```

Generated graphs will be saved in:

```text
artifacts/plots/
```

Generated metrics/reports will be saved in:

```text
artifacts/reports/
```

### 7. Run Streamlit app

```bash
streamlit run app.py
```

## One-command Training Pipeline

After placing the dataset, run:

```bash
python scripts/train_pipeline.py
```

This will verify the dataset, train the model, evaluate the model, and generate all dashboard plots.

## Model Architecture

```text
Input MFCC sequence
        ↓
Masking layer
        ↓
Conv1D feature extractor
        ↓
Batch Normalization + Spatial Dropout
        ↓
Bidirectional LSTM
        ↓
Temporal Attention Mechanism
        ↓
Dense + Dropout
        ↓
Sigmoid output
        ↓
Depression probability
```

## Why Attention Mechanism Is Used

Speech interviews are long, and not every time frame is equally important. The attention mechanism learns which parts of the speech sequence are more useful for predicting depression. This helps the model focus on important voice patterns instead of treating the entire audio equally.

## Folder Structure

```text
automated_depression_detection_daic_woz/
├── app.py
├── README.md
├── requirements.txt
├── configs/
│   └── config.yaml
├── DAIC_WOZ/
│   └── data/
│       ├── audio/
│       ├── transcripts/
│       └── labels/
├── src/
│   ├── audio_features.py
│   ├── config.py
│   ├── data_utils.py
│   ├── evaluate.py
│   ├── explainability.py
│   ├── inference.py
│   ├── model.py
│   ├── train.py
│   └── visualization.py
├── scripts/
│   ├── download_dataset.py
│   ├── train_pipeline.py
│   └── verify_dataset.py
└── artifacts/
    ├── cache/
    ├── models/
    ├── plots/
    └── reports/
```

## Notes for Placement/Interview Explanation

You can explain the project like this:

> I built an automated depression detection system from speech signals using the DAIC-WOZ dataset. First, audio files are converted into MFCC-based time-series features. Then a CNN layer extracts local speech patterns, BiLSTM captures long-term temporal dependencies, and an attention mechanism highlights important time frames. The final sigmoid layer outputs the probability of depression. I also deployed the model with Streamlit, where users can upload or record audio and see depressed/non-depressed probabilities along with ROC curve, confusion matrix, training curves, attention heatmaps, and Grad-CAM style explanations.

## Practical Limitations

- The model depends heavily on the quality and distribution of the DAIC-WOZ dataset.
- Depression is complex and cannot be diagnosed from voice alone.
- Grad-CAM is adapted here for Conv1D audio features; it is a time-frame importance visualization, not a clinical explanation.
- For a real product, the system should be validated by clinicians and tested on diverse populations.
