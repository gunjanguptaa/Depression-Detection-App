from __future__ import annotations

import numpy as np
import tensorflow as tf

from .model import make_attention_extractor


def get_attention_weights(model: tf.keras.Model, x: np.ndarray) -> np.ndarray:
    """Extract temporal attention weights for one or more samples."""
    attention_model = make_attention_extractor(model)
    attention = attention_model.predict(x, verbose=0)
    return np.asarray(attention)


def gradcam_1d(model: tf.keras.Model, x: np.ndarray, conv_layer_name: str = "conv1d_features") -> np.ndarray:
    """Compute a Grad-CAM style explanation over the Conv1D time axis.

    This is Grad-CAM adapted for a Conv1D audio-feature model. It highlights time regions
    in the MFCC sequence that most influenced the depression probability.
    """
    conv_layer = model.get_layer(conv_layer_name)
    grad_model = tf.keras.Model(inputs=model.inputs, outputs=[conv_layer.output, model.output])

    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x_tensor, training=False)
        target = predictions[:, 0]
    grads = tape.gradient(target, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=1)  # (batch, channels)
    weighted = conv_outputs * tf.expand_dims(pooled_grads, axis=1)
    heatmap = tf.reduce_sum(weighted, axis=-1)  # (batch, time)
    heatmap = tf.nn.relu(heatmap)
    heatmap_np = heatmap.numpy()
    for i in range(heatmap_np.shape[0]):
        max_value = np.max(heatmap_np[i])
        if max_value > 0:
            heatmap_np[i] = heatmap_np[i] / max_value
    return heatmap_np


def saliency_map(model: tf.keras.Model, x: np.ndarray) -> np.ndarray:
    """Gradient saliency over the input MFCC matrix."""
    x_tensor = tf.Variable(x.astype(np.float32))
    with tf.GradientTape() as tape:
        preds = model(x_tensor, training=False)
        target = preds[:, 0]
    grads = tape.gradient(target, x_tensor)
    saliency = tf.reduce_max(tf.abs(grads), axis=-1).numpy()
    for i in range(saliency.shape[0]):
        m = np.max(saliency[i])
        if m > 0:
            saliency[i] /= m
    return saliency
