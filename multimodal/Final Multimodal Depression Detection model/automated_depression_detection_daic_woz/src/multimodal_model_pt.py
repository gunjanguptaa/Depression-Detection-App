"""
PyTorch equivalent of the TF CNN-BiLSTM-Attention multimodal model.
Architecture mirrors multimodal_model.py exactly:
  Audio: Input → Masking → Conv1D → BN → SpatialDropout → BiLSTM → TemporalAttention → Dense
  Text:  Input → Dense(128) → BN → Dropout → Dense(64)
  Fusion: Concat → Dense(128) → Dropout → Dense(64) → Dropout → Dense(1, sigmoid)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TemporalAttention(nn.Module):
    """Additive (Bahdanau) attention over LSTM time steps — mirrors TF TemporalAttention."""

    def __init__(self, input_dim: int, attention_units: int = 64):
        super().__init__()
        self.score_dense = nn.Linear(input_dim, attention_units)
        self.score_vector = nn.Linear(attention_units, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        # x: (batch, time, features)
        scores = self.score_vector(torch.tanh(self.score_dense(x)))  # (batch, time, 1)
        scores = scores.squeeze(-1)  # (batch, time)
        if mask is not None:
            # mask: (batch, time) bool, True = valid
            scores = scores.masked_fill(~mask, -1e9)
        weights = F.softmax(scores, dim=1)  # (batch, time)
        context = (x * weights.unsqueeze(-1)).sum(dim=1)  # (batch, features)
        return context, weights


class MultimodalDepressionModel(nn.Module):
    """
    CNN-BiLSTM-Attention + Text fusion model for depression detection.

    Default hyperparameters match the trained Keras model:
      conv_filters=64, conv_kernel_size=5, lstm_units=96,
      dense_units=64, dropout=0.35, text_embedding_dim=384
    """

    def __init__(
        self,
        audio_feature_dim: int = 120,   # 40 MFCCs × 3 (delta, delta-delta)
        text_embedding_dim: int = 384,
        conv_filters: int = 64,
        conv_kernel_size: int = 5,
        lstm_units: int = 96,
        dense_units: int = 64,
        dropout: float = 0.35,
    ):
        super().__init__()
        self.dropout_rate = dropout

        # ── Audio branch ──────────────────────────────────────
        self.conv1d = nn.Conv1d(
            in_channels=audio_feature_dim,
            out_channels=conv_filters,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size // 2,  # "same" padding
        )
        self.conv_bn = nn.BatchNorm1d(conv_filters)
        self.spatial_dropout = nn.Dropout(dropout)  # applied on channel dim

        self.bilstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=lstm_units,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,  # handled externally
        )
        # BiLSTM output dim = lstm_units * 2
        self.attention = TemporalAttention(
            input_dim=lstm_units * 2,
            attention_units=dense_units,
        )
        self.audio_dense = nn.Linear(lstm_units * 2, dense_units)
        self.audio_bn_out = nn.BatchNorm1d(dense_units)
        self.audio_dropout = nn.Dropout(dropout)

        # ── Text branch ───────────────────────────────────────
        self.text_dense1 = nn.Linear(text_embedding_dim, 128)
        self.text_bn = nn.BatchNorm1d(128)
        self.text_dropout = nn.Dropout(dropout)
        self.text_dense2 = nn.Linear(128, dense_units)

        # ── Fusion ────────────────────────────────────────────
        self.fusion_dense1 = nn.Linear(dense_units * 2, 128)
        self.fusion_dropout1 = nn.Dropout(dropout)
        self.fusion_dense2 = nn.Linear(128, 64)
        self.fusion_dropout2 = nn.Dropout(dropout * 0.7)
        self.output_layer = nn.Linear(64, 1)

    def forward(
        self,
        audio: torch.Tensor,          # (batch, time, feature_dim)
        text: torch.Tensor,            # (batch, text_embedding_dim)
        audio_mask: torch.Tensor | None = None,  # (batch, time) bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            prob: (batch,) sigmoid probability of depression
            attn: (batch, time) attention weights
        """
        # ── Audio branch ──────────────────────────────────────
        # Conv1D expects (batch, channels, time)
        x = audio.permute(0, 2, 1)          # (batch, feature_dim, time)
        x = F.relu(self.conv1d(x))          # (batch, conv_filters, time)
        x = self.conv_bn(x)
        x = self.spatial_dropout(x)
        x = x.permute(0, 2, 1)             # (batch, time, conv_filters)

        lstm_out, _ = self.bilstm(x)        # (batch, time, lstm_units*2)
        context, attn = self.attention(lstm_out, mask=audio_mask)

        a = F.relu(self.audio_dense(context))
        a = self.audio_dropout(a)

        # ── Text branch ───────────────────────────────────────
        t = F.relu(self.text_dense1(text))
        t = self.text_bn(t)
        t = self.text_dropout(t)
        t = F.relu(self.text_dense2(t))

        # ── Fusion ────────────────────────────────────────────
        f = torch.cat([a, t], dim=1)
        f = F.relu(self.fusion_dense1(f))
        f = self.fusion_dropout1(f)
        f = F.relu(self.fusion_dense2(f))
        f = self.fusion_dropout2(f)
        logit = self.output_layer(f).squeeze(-1)
        prob = torch.sigmoid(logit)
        return prob, attn


def build_model(cfg: dict, audio_feature_dim: int = 120) -> MultimodalDepressionModel:
    """Build model from config dict (same structure as the YAML config)."""
    model_cfg = cfg.get("model", {})
    return MultimodalDepressionModel(
        audio_feature_dim=audio_feature_dim,
        text_embedding_dim=384,
        conv_filters=int(model_cfg.get("conv_filters", 64)),
        conv_kernel_size=int(model_cfg.get("conv_kernel_size", 5)),
        lstm_units=int(model_cfg.get("lstm_units", 96)),
        dense_units=int(model_cfg.get("dense_units", 64)),
        dropout=float(model_cfg.get("dropout", 0.35)),
    )


def save_model(model: MultimodalDepressionModel, path: str) -> None:
    torch.save(model.state_dict(), path)


def load_model(path: str, cfg: dict, audio_feature_dim: int = 120) -> MultimodalDepressionModel:
    model = build_model(cfg, audio_feature_dim)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model
