from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix, roc_auc_score, roc_curve


def _save_fig(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_class_distribution(y: np.ndarray, output_path: str | Path) -> None:
    labels, counts = np.unique(y.astype(int), return_counts=True)
    names = ["Non-depressed" if int(i) == 0 else "Depressed" for i in labels]
    plt.figure(figsize=(6, 4))
    plt.bar(names, counts)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Segments")
    for idx, value in enumerate(counts):
        plt.text(idx, value, str(value), ha="center", va="bottom")
    _save_fig(output_path)


def plot_training_history(history, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    hist = pd.DataFrame(history.history)
    hist.to_csv(output_dir / "training_history.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.plot(hist.index + 1, hist["loss"], label="Train Loss")
    if "val_loss" in hist:
        plt.plot(hist.index + 1, hist["val_loss"], label="Validation Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.legend()
    _save_fig(output_dir / "loss_curve.png")

    plt.figure(figsize=(7, 4))
    if "accuracy" in hist:
        plt.plot(hist.index + 1, hist["accuracy"], label="Train Accuracy")
    if "val_accuracy" in hist:
        plt.plot(hist.index + 1, hist["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    _save_fig(output_dir / "accuracy_curve.png")

    if "auc" in hist:
        plt.figure(figsize=(7, 4))
        plt.plot(hist.index + 1, hist["auc"], label="Train AUC")
        if "val_auc" in hist:
            plt.plot(hist.index + 1, hist["val_auc"], label="Validation AUC")
        plt.title("AUC Curve")
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        plt.legend()
        _save_fig(output_dir / "auc_curve.png")


def plot_confusion_matrix(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, output_path: str | Path) -> None:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true.astype(int), y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-depressed", "Depressed"])
    disp.plot(values_format="d")
    plt.title("Confusion Matrix")
    _save_fig(output_path)


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: str | Path) -> float:
    auc_value = float(roc_auc_score(y_true.astype(int), y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    fpr, tpr, _ = roc_curve(y_true.astype(int), y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc_value:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    _save_fig(output_path)
    return auc_value


def plot_probability_distribution(y_prob: np.ndarray, output_path: str | Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.hist(y_prob, bins=20)
    plt.title("Predicted Depression Probability Distribution")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    _save_fig(output_path)


def plot_feature_heatmap(features: np.ndarray, output_path: str | Path, title: str = "MFCC Feature Heatmap") -> None:
    # features: (time, feature_dim)
    plt.figure(figsize=(10, 4))
    plt.imshow(features.T, aspect="auto", origin="lower")
    plt.title(title)
    plt.xlabel("Time Frames")
    plt.ylabel("MFCC / Delta Feature Index")
    plt.colorbar(label="Normalized Feature Value")
    _save_fig(output_path)


def plot_attention_weights(attention: np.ndarray, output_path: str | Path) -> None:
    attention = np.asarray(attention).reshape(-1)
    plt.figure(figsize=(10, 3))
    plt.plot(attention)
    plt.title("Attention Weights Over Time")
    plt.xlabel("Time Frames")
    plt.ylabel("Attention Weight")
    _save_fig(output_path)


def plot_gradcam_1d(heatmap: np.ndarray, output_path: str | Path) -> None:
    heatmap = np.asarray(heatmap).reshape(-1)
    plt.figure(figsize=(10, 3))
    plt.plot(heatmap)
    plt.title("Grad-CAM Style Heatmap Over Speech Time Frames")
    plt.xlabel("Time Frames")
    plt.ylabel("Importance")
    _save_fig(output_path)
