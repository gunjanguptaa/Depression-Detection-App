"""
Depression Detection - Classical ML Models Training Script
===========================================================
Run this script ONCE locally with the exact environment in requirements.txt
to regenerate all .pkl model files compatible with Streamlit Cloud.

Usage:
    python train_classical_models.py --data_dir ./data/DAIC_WOZ
    python train_classical_models.py --use_synthetic   # for testing without dataset

Requirements (must match requirements.txt EXACTLY):
    numpy==1.26.4
    scikit-learn>=1.3
    xgboost>=2.0
    joblib>=1.3
    pandas>=2.0
    librosa==0.10.2
    imbalanced-learn>=0.11
"""

import os
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import librosa
from pathlib import Path

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SAVE_DIR = Path("comparison/models")
RESULTS_DIR = Path("comparison/results")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5   # Cross-validation folds

# PHQ-8 threshold for depression label  (score >= 10 = depressed)
PHQ_THRESHOLD = 10

# Audio feature extraction settings
SR = 16000
N_MFCC = 40
HOP_LENGTH = 512
N_FFT = 2048


# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────

def extract_features_from_audio(audio_path: str) -> np.ndarray:
    """Extract MFCCs + spectral features from a single audio file."""
    try:
        y, sr = librosa.load(audio_path, sr=SR, mono=True, duration=300)
    except Exception as e:
        print(f"  ⚠ Could not load {audio_path}: {e}")
        return None

    features = []

    # 1. MFCCs (mean + std)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                  n_fft=N_FFT, hop_length=HOP_LENGTH)
    features.extend(np.mean(mfccs, axis=1).tolist())
    features.extend(np.std(mfccs, axis=1).tolist())

    # 2. Delta MFCCs
    delta_mfccs = librosa.feature.delta(mfccs)
    features.extend(np.mean(delta_mfccs, axis=1).tolist())
    features.extend(np.std(delta_mfccs, axis=1).tolist())

    # 3. Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH)
    features.extend(np.mean(chroma, axis=1).tolist())
    features.extend(np.std(chroma, axis=1).tolist())

    # 4. Mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features.append(float(np.mean(mel_db)))
    features.append(float(np.std(mel_db)))

    # 5. Spectral features
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    for feat in [spec_centroid, spec_bandwidth, spec_rolloff, zcr, rms]:
        features.append(float(np.mean(feat)))
        features.append(float(np.std(feat)))

    features.extend(np.mean(spec_contrast, axis=1).tolist())
    features.extend(np.std(spec_contrast, axis=1).tolist())

    # 6. Pitch (fundamental frequency via pyin)
    try:
        f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'),
                                          fmax=librosa.note_to_hz('C7'))
        f0_valid = f0[voiced_flag] if voiced_flag is not None and voiced_flag.any() else np.array([0.0])
        features.append(float(np.mean(f0_valid)))
        features.append(float(np.std(f0_valid)))
        features.append(float(np.mean(voiced_flag.astype(float))) if voiced_flag is not None else 0.0)
    except Exception:
        features.extend([0.0, 0.0, 0.0])

    return np.array(features, dtype=np.float32)


def load_daic_woz_dataset(data_dir: str):
    """
    Load features and labels from the DAIC-WOZ dataset directory.

    Expected structure:
        data_dir/
          train_split_Depression_AVEC2017.csv   (or labels.csv)
          300_P/   300_AUDIO.wav
          301_P/   301_AUDIO.wav
          ...

    The CSV must have columns: Participant_ID, PHQ8_Score (or PHQ8_Binary)
    """
    data_dir = Path(data_dir)
    label_files = list(data_dir.glob("*train_split*.csv")) + list(data_dir.glob("labels*.csv"))

    if not label_files:
        raise FileNotFoundError(
            f"No label CSV found in {data_dir}. "
            "Expected file matching *train_split*.csv or labels*.csv"
        )

    df = pd.read_csv(label_files[0])
    print(f"✓ Loaded labels from: {label_files[0].name} ({len(df)} participants)")

    # Determine label column
    if "PHQ8_Binary" in df.columns:
        label_col = "PHQ8_Binary"
    elif "PHQ8_Score" in df.columns:
        label_col = None  # will threshold below
    else:
        raise ValueError("CSV must have 'PHQ8_Binary' or 'PHQ8_Score' column.")

    X, y = [], []
    skipped = 0

    for _, row in df.iterrows():
        pid = int(row["Participant_ID"])
        label = int(row[label_col]) if label_col else int(row["PHQ8_Score"] >= PHQ_THRESHOLD)

        # Try common audio path patterns
        audio_candidates = [
            data_dir / f"{pid}_P" / f"{pid}_AUDIO.wav",
            data_dir / f"{pid}" / f"{pid}_AUDIO.wav",
            data_dir / f"{pid}_AUDIO.wav",
        ]

        audio_path = None
        for cand in audio_candidates:
            if cand.exists():
                audio_path = str(cand)
                break

        if audio_path is None:
            skipped += 1
            continue

        feats = extract_features_from_audio(audio_path)
        if feats is not None:
            X.append(feats)
            y.append(label)
            print(f"  [{len(X):>3}] PID {pid} → label={label}  features={feats.shape[0]}")

    print(f"\n✓ Extracted features: {len(X)} samples  (skipped {skipped})")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def make_synthetic_dataset(n_samples=200, n_features=198):
    """
    Fallback: generate a reproducible synthetic dataset that matches the
    real feature dimensionality.  Use ONLY for testing the pipeline.
    """
    print("⚠  Using SYNTHETIC data — replace with real DAIC-WOZ for production!")
    rng = np.random.RandomState(RANDOM_STATE)

    X = rng.randn(n_samples, n_features).astype(np.float32)
    # Slightly separate the two classes so models learn something
    y = (rng.rand(n_samples) > 0.60).astype(np.int32)  # ~40 % depressed
    X[y == 1] += 0.8  # push depressed class away from origin
    return X, y


# ─────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────

def get_models():
    return {
        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "Logistic Regression": LogisticRegression(
            C=0.1,
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=RANDOM_STATE
        ),
    }


# ─────────────────────────────────────────────
# TRAINING + EVALUATION
# ─────────────────────────────────────────────

def train_and_evaluate(X, y):
    """
    For each model:
      1. Apply SMOTE to handle class imbalance
      2. Train on full dataset (after SMOTE)
      3. Get cross-validated predictions for honest metrics
      4. Save model + scaler as .pkl
      5. Save metrics as JSON
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler
    scaler_path = SAVE_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path, protocol=4)   # protocol=4 → Python 3.8+ safe
    print(f"\n✓ Saved scaler → {scaler_path}")

    models = get_models()
    all_results = {}
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for model_name, model in models.items():
        print(f"\n{'─'*55}")
        print(f"  Training: {model_name}")
        print(f"{'─'*55}")

        # ── Cross-validated predictions (no data leakage) ─────────────
        y_pred_cv = cross_val_predict(model, X_scaled, y, cv=skf, method="predict")
        try:
            y_prob_cv = cross_val_predict(model, X_scaled, y, cv=skf, method="predict_proba")[:, 1]
            auc = float(roc_auc_score(y, y_prob_cv))
        except Exception:
            auc = float(roc_auc_score(y, y_pred_cv))

        acc  = float(accuracy_score(y, y_pred_cv))
        f1   = float(f1_score(y, y_pred_cv, average="weighted"))
        prec = float(precision_score(y, y_pred_cv, average="weighted", zero_division=0))
        rec  = float(recall_score(y, y_pred_cv, average="weighted", zero_division=0))
        cm   = confusion_matrix(y, y_pred_cv).tolist()
        report = classification_report(y, y_pred_cv,
                                        target_names=["Non-depressed", "Depressed"],
                                        output_dict=True)

        print(f"  Accuracy : {acc:.4f}")
        print(f"  F1 Score : {f1:.4f}")
        print(f"  AUC-ROC  : {auc:.4f}")

        # ── Fit on FULL dataset with SMOTE ─────────────────────────────
        try:
            smote = SMOTE(random_state=RANDOM_STATE)
            X_res, y_res = smote.fit_resample(X_scaled, y)
            print(f"  SMOTE: {len(y)} → {len(y_res)} samples")
        except Exception as e:
            print(f"  SMOTE skipped ({e}) — using original data")
            X_res, y_res = X_scaled, y

        model.fit(X_res, y_res)

        # ── Save model ─────────────────────────────────────────────────
        safe_name = model_name.replace(" ", "_").lower()
        model_path = SAVE_DIR / f"{safe_name}.pkl"
        joblib.dump(model, model_path, protocol=4)   # ← protocol=4 is key!
        print(f"  ✓ Saved → {model_path}")

        # ── Save per-fold metrics ──────────────────────────────────────
        fold_metrics = []
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
            X_tr, X_val = X_scaled[tr_idx], X_scaled[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            try:
                sm = SMOTE(random_state=RANDOM_STATE)
                X_tr_r, y_tr_r = sm.fit_resample(X_tr, y_tr)
            except Exception:
                X_tr_r, y_tr_r = X_tr, y_tr
            fold_model = type(model)(**model.get_params())
            fold_model.fit(X_tr_r, y_tr_r)
            yp = fold_model.predict(X_val)
            fold_metrics.append({
                "fold": fold + 1,
                "accuracy": float(accuracy_score(y_val, yp)),
                "f1": float(f1_score(y_val, yp, average="weighted", zero_division=0)),
            })

        all_results[model_name] = {
            "accuracy": acc,
            "f1_score": f1,
            "precision": prec,
            "recall": rec,
            "roc_auc": auc,
            "confusion_matrix": cm,
            "classification_report": report,
            "fold_metrics": fold_metrics,
            "model_path": str(model_path),
            "n_samples": int(len(y)),
            "n_features": int(X.shape[1]),
            "class_distribution": {
                "non_depressed": int(np.sum(y == 0)),
                "depressed": int(np.sum(y == 1)),
            }
        }

    # ── Save combined results JSON ─────────────────────────────────────
    results_path = RESULTS_DIR / "classical_ml_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Saved all results → {results_path}")

    return all_results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Classical ML models for depression detection")
    parser.add_argument("--data_dir", type=str, default="./data/DAIC_WOZ",
                        help="Path to DAIC-WOZ dataset directory")
    parser.add_argument("--use_synthetic", action="store_true",
                        help="Use synthetic data instead of real dataset (for testing)")
    args = parser.parse_args()

    print("="*55)
    print("  Depression Detection — Classical ML Training")
    print("="*55)
    print(f"  NumPy version  : {np.__version__}")
    print(f"  Output dir     : {SAVE_DIR.resolve()}")
    print(f"  Results dir    : {RESULTS_DIR.resolve()}")
    print()

    if args.use_synthetic:
        X, y = make_synthetic_dataset()
    else:
        if not os.path.isdir(args.data_dir):
            print(f"❌  data_dir not found: {args.data_dir}")
            print("   Run with --use_synthetic to test the pipeline first.")
            return
        X, y = load_daic_woz_dataset(args.data_dir)

    print(f"\nDataset shape : X={X.shape}  y={y.shape}")
    print(f"Class balance : Non-depressed={np.sum(y==0)}  Depressed={np.sum(y==1)}")

    results = train_and_evaluate(X, y)

    print("\n" + "="*55)
    print("  SUMMARY")
    print("="*55)
    for name, m in results.items():
        print(f"  {name:<22}  Acc={m['accuracy']:.3f}  F1={m['f1_score']:.3f}  AUC={m['roc_auc']:.3f}")

    print("\n✅  Training complete!  Next steps:")
    print("   1. Copy  comparison/models/   into your GitHub repo")
    print("   2. Copy  comparison/results/  into your GitHub repo")
    print("   3. git add comparison/models/* comparison/results/*")
    print("   4. git commit -m 'retrain: numpy 1.26.4 compatible models'")
    print("   5. git push  → Streamlit Cloud will redeploy automatically")


if __name__ == "__main__":
    main()
