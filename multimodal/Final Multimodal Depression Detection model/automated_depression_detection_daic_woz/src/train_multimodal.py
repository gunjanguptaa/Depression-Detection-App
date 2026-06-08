from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from .config import ensure_dirs, load_config, save_preprocess_config, set_global_seed
from .data_utils import fit_scaler, transform_with_scaler
from .multimodal_data import build_multimodal_examples_from_split
from .multimodal_model import build_multimodal_attention_lstm
from .text_utils import TEXT_EMBEDDING_DIM
from .visualization import plot_class_distribution, plot_training_history


def main(config_path: str | None = None, overwrite_cache: bool = False) -> None:
    cfg = load_config(config_path) if config_path else load_config()
    ensure_dirs(cfg)
    set_global_seed(int(cfg["training"].get("random_seed", 42)))

    cache_dir = Path(cfg["paths"]["cache_dir"])
    plots_dir = Path(cfg["paths"]["plots_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    model_dir = Path(cfg["paths"]["model_dir"])
    multimodal_model_path = str(model_dir / "best_multimodal_attention_lstm.keras")

    print("Preparing multimodal training features: audio + transcript embeddings...")
    X_audio_train, X_text_train, y_train, meta_train = build_multimodal_examples_from_split(
        cfg["paths"]["train_labels"],
        cfg["paths"]["audio_dir"],
        cfg["paths"]["transcript_dir"],
        cfg["features"],
        cache_dir=cache_dir,
        split_name="train",
        overwrite_cache=overwrite_cache,
    )
    print(f"Train audio: {X_audio_train.shape}, text: {X_text_train.shape}, labels: {np.bincount(y_train.astype(int))}")

    print("Preparing multimodal validation/dev features...")
    X_audio_val, X_text_val, y_val, meta_val = build_multimodal_examples_from_split(
        cfg["paths"]["dev_labels"],
        cfg["paths"]["audio_dir"],
        cfg["paths"]["transcript_dir"],
        cfg["features"],
        cache_dir=cache_dir,
        split_name="dev",
        overwrite_cache=overwrite_cache,
    )
    print(f"Dev audio: {X_audio_val.shape}, text: {X_text_val.shape}, labels: {np.bincount(y_val.astype(int))}")

    scaler = fit_scaler(X_audio_train, cfg["paths"]["scaler"])
    X_audio_train = transform_with_scaler(X_audio_train, scaler)
    X_audio_val = transform_with_scaler(X_audio_val, scaler)

    plot_class_distribution(y_train, plots_dir / "multimodal_class_distribution_train.png")

    model = build_multimodal_attention_lstm(
        audio_input_shape=X_audio_train.shape[1:],
        text_embedding_dim=TEXT_EMBEDDING_DIM,
        cfg=cfg,
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            multimodal_model_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=int(cfg["training"].get("patience", 7)),
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    class_weight = None
    if bool(cfg["training"].get("use_class_weights", True)) and len(np.unique(y_train)) > 1:
        weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_train.astype(int))
        class_weight = {0: float(weights[0]), 1: float(weights[1])}
        print(f"Class weights: {class_weight}")

    history = model.fit(
        [X_audio_train, X_text_train],
        y_train,
        validation_data=([X_audio_val, X_text_val], y_val),
        epochs=int(cfg["training"].get("epochs", 30)),
        batch_size=int(cfg["training"].get("batch_size", 8)),
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(multimodal_model_path)
    plot_training_history(history, plots_dir / "multimodal")
    save_preprocess_config(cfg)

    summary = {
        "train_audio_shape": list(X_audio_train.shape),
        "train_text_shape": list(X_text_train.shape),
        "val_audio_shape": list(X_audio_val.shape),
        "val_text_shape": list(X_text_val.shape),
        "best_multimodal_model": multimodal_model_path,
        "scaler": cfg["paths"]["scaler"],
        "note": "Multimodal model uses audio MFCC features + transcript sentence embeddings.",
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    with (reports_dir / "multimodal_training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Multimodal training complete. Run: python -m src.evaluate_multimodal")


if __name__ == "__main__":
    main()
