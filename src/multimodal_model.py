from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

from .model import TemporalAttention


def build_multimodal_attention_lstm(
    audio_input_shape: Tuple[int, int],
    text_embedding_dim: int,
    cfg: dict,
) -> tf.keras.Model:
    """Build audio CNN-BiLSTM-Attention + transcript embedding fusion model."""
    model_cfg = cfg.get("model", {})
    lr = float(cfg.get("training", {}).get("learning_rate", 3e-4))
    dropout = float(model_cfg.get("dropout", 0.35))
    dense_units = int(model_cfg.get("dense_units", 64))

    audio_input = layers.Input(shape=audio_input_shape, name="mfcc_input")
    x = layers.Masking(mask_value=0.0, name="padding_mask")(audio_input)
    x = layers.Conv1D(
        filters=int(model_cfg.get("conv_filters", 64)),
        kernel_size=int(model_cfg.get("conv_kernel_size", 5)),
        padding="same",
        activation="relu",
        name="conv1d_features",
    )(x)
    x = layers.BatchNormalization(name="conv_batch_norm")(x)
    x = layers.SpatialDropout1D(dropout, name="conv_dropout")(x)
    x = layers.Bidirectional(
        layers.LSTM(
            int(model_cfg.get("lstm_units", 96)),
            return_sequences=True,
            dropout=dropout,
            recurrent_dropout=0.0,
        ),
        name="bidirectional_lstm",
    )(x)
    audio_context, attention_weights = TemporalAttention(attention_units=dense_units, name="temporal_attention")(x)
    audio_vec = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="audio_embedding",
    )(audio_context)
    audio_vec = layers.Dropout(dropout, name="audio_dropout")(audio_vec)

    text_input = layers.Input(shape=(text_embedding_dim,), name="text_embedding_input")
    t = layers.Dense(128, activation="relu", name="text_dense_128")(text_input)
    t = layers.BatchNormalization(name="text_batch_norm")(t)
    t = layers.Dropout(dropout, name="text_dropout")(t)
    text_vec = layers.Dense(dense_units, activation="relu", name="text_embedding_dense")(t)

    fusion = layers.Concatenate(name="audio_text_fusion")([audio_vec, text_vec])
    fusion = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4), name="fusion_dense_128")(fusion)
    fusion = layers.Dropout(dropout, name="fusion_dropout_1")(fusion)
    fusion = layers.Dense(64, activation="relu", name="fusion_dense_64")(fusion)
    fusion = layers.Dropout(dropout * 0.7, name="fusion_dropout_2")(fusion)
    output = layers.Dense(1, activation="sigmoid", name="depression_probability")(fusion)

    model = models.Model(
        inputs=[audio_input, text_input],
        outputs=output,
        name="multimodal_cnn_bilstm_attention_text_fusion",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def load_multimodal_model(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path, custom_objects={"TemporalAttention": TemporalAttention})


def make_multimodal_attention_extractor(model: tf.keras.Model) -> tf.keras.Model:
    attention_layer = model.get_layer("temporal_attention")
    return tf.keras.Model(inputs=model.inputs, outputs=attention_layer.output[1])
