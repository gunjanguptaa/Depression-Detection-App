from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from .config import ensure_dirs, load_config, save_preprocess_config, set_global_seed
from .data_utils import build_examples_from_split, fit_scaler, transform_with_scaler
from .model import build_attention_lstm
from .visualization import plot_class_distribution, plot_training_history


def main(config_path: str | None = None, overwrite_cache: bool = False) -> None:
    cfg = load_config(config_path) if config_path else load_config()
    ensure_dirs(cfg)
    set_global_seed(int(cfg["training"].get("random_seed", 42)))

    cache_dir = Path(cfg["paths"]["cache_dir"])
    plots_dir = Path(cfg["paths"]["plots_dir"])

    print("Preparing training features...")
    X_train, y_train, meta_train = build_examples_from_split(
        cfg["paths"]["train_labels"],
        cfg["paths"]["audio_dir"],
        cfg["features"],
        cache_path=cache_dir / "train_features.npz",
        overwrite_cache=overwrite_cache,
    )
    print(f"Train segments: {X_train.shape}, labels: {np.bincount(y_train.astype(int))}")

    dev_csv = Path(cfg["paths"].get("dev_labels", ""))
    if dev_csv.exists():
        print("Preparing validation/dev features...")
        X_val, y_val, meta_val = build_examples_from_split(
            cfg["paths"]["dev_labels"],
            cfg["paths"]["audio_dir"],
            cfg["features"],
            cache_path=cache_dir / "dev_features.npz",
            overwrite_cache=overwrite_cache,
        )
    else:
        print("Dev split CSV not found. Using validation_split_fallback from train split.")
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=float(cfg["training"].get("validation_split_fallback", 0.2)),
            stratify=y_train if len(np.unique(y_train)) > 1 else None,
            random_state=int(cfg["training"].get("random_seed", 42)),
        )

    scaler = fit_scaler(X_train, cfg["paths"]["scaler"])
    X_train = transform_with_scaler(X_train, scaler)
    X_val = transform_with_scaler(X_val, scaler)

    plot_class_distribution(y_train, plots_dir / "class_distribution_train.png")

    model = build_attention_lstm(input_shape=X_train.shape[1:], cfg=cfg)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            cfg["paths"]["best_model"],
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
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=int(cfg["training"].get("epochs", 30)),
        batch_size=int(cfg["training"].get("batch_size", 8)),
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(cfg["paths"]["best_model"])
    plot_training_history(history, plots_dir)
    save_preprocess_config(cfg)

    summary = {
        "train_shape": list(X_train.shape),
        "val_shape": list(X_val.shape),
        "best_model": cfg["paths"]["best_model"],
        "scaler": cfg["paths"]["scaler"],
    }
    with Path(cfg["paths"]["reports_dir"]).joinpath("training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Training complete. Run: python -m src.evaluate")


if __name__ == "__main__":
    main()
