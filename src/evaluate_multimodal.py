from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, roc_auc_score

from .config import ensure_dirs, load_config
from .data_utils import load_scaler, transform_with_scaler
from .multimodal_data import build_multimodal_examples_from_split
from .multimodal_explainability import get_multimodal_attention_weights, multimodal_gradcam_1d
from .multimodal_model import load_multimodal_model
from .visualization import (
    plot_attention_weights,
    plot_confusion_matrix,
    plot_feature_heatmap,
    plot_gradcam_1d,
    plot_probability_distribution,
    plot_roc_curve,
)


def main(config_path: str | None = None, split: str = "dev", overwrite_cache: bool = False) -> None:
    cfg = load_config(config_path) if config_path else load_config()
    ensure_dirs(cfg)

    cache_dir = Path(cfg["paths"]["cache_dir"])
    plots_dir = Path(cfg["paths"]["plots_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    model_path = Path(cfg["paths"]["model_dir"]) / "best_multimodal_attention_lstm.keras"

    if split == "test" and Path(cfg["paths"].get("test_labels", "")).exists():
        split_csv = cfg["paths"]["test_labels"]
        split_name = "test"
    else:
        split_csv = cfg["paths"]["dev_labels"]
        split_name = "dev"

    X_audio, X_text, y, metadata = build_multimodal_examples_from_split(
        split_csv,
        cfg["paths"]["audio_dir"],
        cfg["paths"]["transcript_dir"],
        cfg["features"],
        cache_dir=cache_dir,
        split_name=split_name,
        overwrite_cache=overwrite_cache,
    )
    scaler = load_scaler(cfg["paths"]["scaler"])
    X_audio_scaled = transform_with_scaler(X_audio, scaler)

    model = load_multimodal_model(str(model_path))
    y_prob = model.predict([X_audio_scaled, X_text], batch_size=int(cfg["training"].get("batch_size", 8)), verbose=1).reshape(-1)
    threshold = float(cfg["training"].get("threshold", 0.65))
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "split": split_name,
        "threshold": threshold,
        "accuracy": float(accuracy_score(y.astype(int), y_pred)),
    }
    if len(np.unique(y)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y.astype(int), y_prob))

    precision, recall, f1, support = precision_recall_fscore_support(y.astype(int), y_pred, labels=[0, 1], zero_division=0)
    metrics.update(
        {
            "non_depressed_precision": float(precision[0]),
            "non_depressed_recall": float(recall[0]),
            "non_depressed_f1": float(f1[0]),
            "depressed_precision": float(precision[1]),
            "depressed_recall": float(recall[1]),
            "depressed_f1": float(f1[1]),
        }
    )

    reports_dir.mkdir(parents=True, exist_ok=True)
    with (reports_dir / f"{split_name}_multimodal_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    report = classification_report(y.astype(int), y_pred, target_names=["Non-depressed", "Depressed"], zero_division=0)
    (reports_dir / f"{split_name}_multimodal_classification_report.txt").write_text(report, encoding="utf-8")

    predictions = metadata.copy()
    predictions["y_true"] = y.astype(int)
    predictions["depressed_probability"] = y_prob
    predictions["predicted_label"] = y_pred
    predictions.to_csv(reports_dir / f"{split_name}_multimodal_predictions.csv", index=False)

    plot_confusion_matrix(y, y_prob, threshold, plots_dir / "multimodal_confusion_matrix.png")
    if len(np.unique(y)) > 1:
        plot_roc_curve(y, y_prob, plots_dir / "multimodal_roc_curve.png")
    plot_probability_distribution(y_prob, plots_dir / "multimodal_prediction_probability_distribution.png")

    idx = int(np.argmax(y_prob)) if len(y_prob) else 0
    plot_feature_heatmap(X_audio_scaled[idx], plots_dir / "multimodal_example_mfcc_heatmap.png")
    att = get_multimodal_attention_weights(model, X_audio_scaled[idx : idx + 1], X_text[idx : idx + 1])[0]
    plot_attention_weights(att, plots_dir / "multimodal_example_attention_weights.png")
    cam = multimodal_gradcam_1d(model, X_audio_scaled[idx : idx + 1], X_text[idx : idx + 1])[0]
    plot_gradcam_1d(cam, plots_dir / "multimodal_example_gradcam_1d.png")

    print(json.dumps(metrics, indent=2))
    print(f"Multimodal evaluation reports saved in: {reports_dir}")


if __name__ == "__main__":
    main()
