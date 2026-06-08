from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


class TemporalAttention(layers.Layer):
    """Additive attention over LSTM time steps."""

    def __init__(self, attention_units: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.attention_units = attention_units
        self.score_dense = layers.Dense(attention_units, activation="tanh")
        self.score_vector = layers.Dense(1, use_bias=False)

    def call(self, inputs, mask=None, training=None):
        scores = self.score_vector(self.score_dense(inputs))  # (batch, time, 1)
        scores = tf.squeeze(scores, axis=-1)  # (batch, time)
        if mask is not None:
            mask = tf.cast(mask, dtype=scores.dtype)
            scores += (1.0 - mask) * tf.constant(-1e9, dtype=scores.dtype)
        weights = tf.nn.softmax(scores, axis=1)
        context = tf.reduce_sum(inputs * tf.expand_dims(weights, axis=-1), axis=1)
        return context, weights

    def get_config(self):
        config = super().get_config()
        config.update({"attention_units": self.attention_units})
        return config


def build_attention_lstm(input_shape: Tuple[int, int], cfg: dict) -> tf.keras.Model:
    """Build CNN + BiLSTM + temporal attention model for depression classification."""
    model_cfg = cfg.get("model", {})
    lr = float(cfg.get("training", {}).get("learning_rate", 3e-4))

    inputs = layers.Input(shape=input_shape, name="mfcc_input")
    x = layers.Masking(mask_value=0.0, name="padding_mask")(inputs)
    x = layers.Conv1D(
        filters=int(model_cfg.get("conv_filters", 64)),
        kernel_size=int(model_cfg.get("conv_kernel_size", 5)),
        padding="same",
        activation="relu",
        name="conv1d_features",
    )(x)
    x = layers.BatchNormalization(name="conv_batch_norm")(x)
    x = layers.SpatialDropout1D(float(model_cfg.get("dropout", 0.35)), name="conv_dropout")(x)
    x = layers.Bidirectional(
        layers.LSTM(
            int(model_cfg.get("lstm_units", 96)),
            return_sequences=True,
            dropout=float(model_cfg.get("dropout", 0.35)),
            recurrent_dropout=0.0,
        ),
        name="bidirectional_lstm",
    )(x)
    context, attention_weights = TemporalAttention(
        attention_units=int(model_cfg.get("dense_units", 64)), name="temporal_attention"
    )(x)
    x = layers.Dense(
        int(model_cfg.get("dense_units", 64)),
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_features",
    )(context)
    x = layers.Dropout(float(model_cfg.get("dropout", 0.35)), name="dense_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="depression_probability")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="cnn_bilstm_attention_depression_detector")
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


def load_trained_model(model_path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(model_path, custom_objects={"TemporalAttention": TemporalAttention})


def make_attention_extractor(model: tf.keras.Model) -> tf.keras.Model:
    attention_layer = model.get_layer("temporal_attention")
    return tf.keras.Model(inputs=model.input, outputs=attention_layer.output[1])


def make_conv_extractor(model: tf.keras.Model) -> tf.keras.Model:
    return tf.keras.Model(inputs=model.input, outputs=model.get_layer("conv1d_features").output)
