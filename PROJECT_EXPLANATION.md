# Placement Explanation Script

## 1. Problem Statement

Depression is a serious mental-health condition, and early screening can help people seek support faster. The goal of this project is to detect depression automatically from speech signals using deep learning. The system takes a user's audio as input and returns the probability of depressed and non-depressed classes.

## 2. Dataset

The project uses DAIC-WOZ-style data with three main folders:

- `audio/`: interview audio files such as `300.wav`, `301.wav`
- `labels/`: CSV files containing depression labels such as `PHQ8_Binary`
- `transcripts/`: transcript CSV files, kept for future multimodal extension

In this version, the model is trained using speech/audio signals only.

## 3. Workflow

1. Load audio file.
2. Resample it to 16 kHz and convert it to mono.
3. Split long interviews into fixed-length segments.
4. Extract MFCC, delta, and delta-delta features.
5. Normalize features using `StandardScaler`.
6. Pass feature sequence into CNN + BiLSTM + Attention model.
7. Generate depressed/non-depressed probabilities.
8. Display explanations using attention weights, Grad-CAM style heatmap, saliency, spectrogram, ROC curve, and confusion matrix.

## 4. Model

The architecture is:

```text
MFCC Features → Conv1D → BatchNorm → BiLSTM → Attention → Dense → Sigmoid
```

- **Conv1D** captures local acoustic patterns.
- **BiLSTM** understands temporal patterns in speech.
- **Attention** focuses on important speech frames.
- **Sigmoid** gives the final depression probability.

## 5. Streamlit App

The Streamlit app has two prediction options:

1. Upload an audio file.
2. Record audio using the microphone.

After prediction, the app displays:

- Final class
- Depressed percentage
- Non-depressed percentage
- Waveform
- Log-mel spectrogram
- MFCC heatmap
- Attention weights
- Grad-CAM style heatmap
- Saliency curve

## 6. Evaluation

The evaluation pipeline generates:

- Accuracy curve
- Loss curve
- AUC curve
- ROC curve
- Confusion matrix
- Prediction probability distribution
- Classification report

## 7. Honest Limitations

This is not a medical diagnosis system. Depression cannot be confirmed only from speech. This project is an AI-based screening prototype and would need clinical validation before real-world use.
