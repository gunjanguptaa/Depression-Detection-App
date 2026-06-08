from __future__ import annotations

import numpy as np
import tensorflow as tf

from .multimodal_model import make_multimodal_attention_extractor


def get_multimodal_attention_weights(model: tf.keras.Model, x_audio: np.ndarray, x_text: np.ndarray) -> np.ndarray:
    attention_model = make_multimodal_attention_extractor(model)
    return np.asarray(attention_model.predict([x_audio, x_text], verbose=0))


def multimodal_gradcam_1d(
    model: tf.keras.Model,
    x_audio: np.ndarray,
    x_text: np.ndarray,
    conv_layer_name: str = "conv1d_features",
) -> np.ndarray:
    conv_layer = model.get_layer(conv_layer_name)
    grad_model = tf.keras.Model(inputs=model.inputs, outputs=[conv_layer.output, model.output])
    x_audio_tensor = tf.convert_to_tensor(x_audio, dtype=tf.float32)
    x_text_tensor = tf.convert_to_tensor(x_text, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([x_audio_tensor, x_text_tensor], training=False)
        target = predictions[:, 0]
    grads = tape.gradient(target, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=1)
    weighted = conv_outputs * tf.expand_dims(pooled_grads, axis=1)
    heatmap = tf.reduce_sum(weighted, axis=-1)
    heatmap = tf.nn.relu(heatmap).numpy()
    for i in range(heatmap.shape[0]):
        m = np.max(heatmap[i])
        if m > 0:
            heatmap[i] /= m
    return heatmap


def multimodal_saliency_map(model: tf.keras.Model, x_audio: np.ndarray, x_text: np.ndarray) -> np.ndarray:
    x_audio_var = tf.Variable(x_audio.astype(np.float32))
    x_text_tensor = tf.convert_to_tensor(x_text.astype(np.float32))
    with tf.GradientTape() as tape:
        preds = model([x_audio_var, x_text_tensor], training=False)
        target = preds[:, 0]
    grads = tape.gradient(target, x_audio_var)
    saliency = tf.reduce_max(tf.abs(grads), axis=-1).numpy()
    for i in range(saliency.shape[0]):
        m = np.max(saliency[i])
        if m > 0:
            saliency[i] /= m
    return saliency
