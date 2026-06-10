"""
PyTorch training script for the multimodal depression detection model.
Mirrors train_multimodal.py logic exactly.

Run on Kaggle:
    python train_multimodal_pt.py

Outputs:
    artifacts/models/best_multimodal_pt.pth   ← trained weights
    artifacts/models/feature_scaler.joblib    ← same scaler format
"""
from __future__ import annotations

import json
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
import joblib

# Re-use existing data pipeline — no changes needed
from src.config import load_config, ensure_dirs, set_global_seed, save_preprocess_config
from src.data_utils import fit_scaler, transform_with_scaler
from src.multimodal_data import build_multimodal_examples_from_split
from src.visualization import plot_class_distribution

from pytorch_src.multimodal_model_pt import build_model, save_model


def train_epoch(model, loader, optimizer, criterion, device, class_weights_tensor=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x_audio, x_text, y in loader:
        x_audio, x_text, y = x_audio.to(device), x_text.to(device), y.to(device)
        optimizer.zero_grad()
        prob, _ = model(x_audio, x_text)
        if class_weights_tensor is not None:
            weight = torch.where(y == 1, class_weights_tensor[1], class_weights_tensor[0])
            loss = nn.BCELoss(weight=weight)(prob, y)
        else:
            loss = criterion(prob, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        pred = (prob >= 0.5).float()
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x_audio, x_text, y in loader:
            x_audio, x_text, y = x_audio.to(device), x_text.to(device), y.to(device)
            prob, _ = model(x_audio, x_text)
            loss = criterion(prob, y)
            total_loss += loss.item() * y.size(0)
            pred = (prob >= 0.5).float()
            correct += (pred == y).sum().item()
            total += y.size(0)
            all_probs.extend(prob.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Compute AUC
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0
    return total_loss / total, correct / total, auc


def main(config_path=None, overwrite_cache=False):
    cfg = load_config(config_path) if config_path else load_config()
    ensure_dirs(cfg)
    set_global_seed(int(cfg["training"].get("random_seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cache_dir   = Path(cfg["paths"]["cache_dir"])
    plots_dir   = Path(cfg["paths"]["plots_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    model_dir   = Path(cfg["paths"]["model_dir"])
    model_path  = str(model_dir / "best_multimodal_pt.pth")

    # ── Load data (reuses existing pipeline) ─────────────────
    print("Loading training data...")
    X_audio_train, X_text_train, y_train, _ = build_multimodal_examples_from_split(
        cfg["paths"]["train_labels"],
        cfg["paths"]["audio_dir"],
        cfg["paths"]["transcript_dir"],
        cfg["features"],
        cache_dir=cache_dir, split_name="train",
        overwrite_cache=overwrite_cache,
    )
    print("Loading validation data...")
    X_audio_val, X_text_val, y_val, _ = build_multimodal_examples_from_split(
        cfg["paths"]["dev_labels"],
        cfg["paths"]["audio_dir"],
        cfg["paths"]["transcript_dir"],
        cfg["features"],
        cache_dir=cache_dir, split_name="dev",
        overwrite_cache=overwrite_cache,
    )

    # ── Scale audio features ──────────────────────────────────
    scaler = fit_scaler(X_audio_train, cfg["paths"]["scaler"])
    X_audio_train = transform_with_scaler(X_audio_train, scaler)
    X_audio_val   = transform_with_scaler(X_audio_val,   scaler)

    print(f"Train: audio={X_audio_train.shape}, text={X_text_train.shape}, labels={np.bincount(y_train.astype(int))}")
    print(f"Val  : audio={X_audio_val.shape},   text={X_text_val.shape},   labels={np.bincount(y_val.astype(int))}")

    plot_class_distribution(y_train, plots_dir / "multimodal_class_distribution_train.png")

    # ── Build model ───────────────────────────────────────────
    audio_feature_dim = X_audio_train.shape[2]  # (samples, time, features)
    model = build_model(cfg, audio_feature_dim=audio_feature_dim).to(device)
    print(model)

    # ── DataLoaders ───────────────────────────────────────────
    batch_size = int(cfg["training"].get("batch_size", 8))

    def make_loader(xa, xt, y, shuffle):
        ds = TensorDataset(
            torch.tensor(xa, dtype=torch.float32),
            torch.tensor(xt, dtype=torch.float32),
            torch.tensor(y,  dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(X_audio_train, X_text_train, y_train, shuffle=True)
    val_loader   = make_loader(X_audio_val,   X_text_val,   y_val,   shuffle=False)

    # ── Class weights ─────────────────────────────────────────
    class_weights_tensor = None
    if cfg["training"].get("use_class_weights", True) and len(np.unique(y_train)) > 1:
        weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train.astype(int))
        class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
        print(f"Class weights: {dict(enumerate(weights))}")

    # ── Optimizer & scheduler ─────────────────────────────────
    lr = float(cfg["training"].get("learning_rate", 3e-4))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6, verbose=True
    )
    criterion = nn.BCELoss()

    # ── Training loop ─────────────────────────────────────────
    epochs  = int(cfg["training"].get("epochs", 30))
    patience = int(cfg["training"].get("patience", 7))

    best_auc   = 0.0
    best_state = None
    no_improve = 0

    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device, class_weights_tensor)
        val_loss, val_acc, val_auc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        print(f"Epoch {epoch:02d}/{epochs} | "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.3f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.3f} auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc   = val_auc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, model_path)
            print(f"  ✓ Saved best model (val_auc={best_auc:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Save training summary ─────────────────────────────────
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "best_val_auc": best_auc,
        "model_path": model_path,
        "audio_feature_dim": audio_feature_dim,
        "train_shape": list(X_audio_train.shape),
        "val_shape": list(X_audio_val.shape),
        "history": history,
    }
    with (reports_dir / "multimodal_pt_training_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    save_preprocess_config(cfg)
    print(f"\nDone. Best val AUC: {best_auc:.4f}. Model saved to: {model_path}")


if __name__ == "__main__":
    main()
