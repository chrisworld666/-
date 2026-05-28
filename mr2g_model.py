# -*- coding: utf-8 -*-
"""
MR2G model definition and artifact loader for local/edge-side inference.

This file recreates the MR2G_SINGLE architecture used in the training notebook
so that mr2g_final.weights.h5 can be loaded outside the notebook.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM, GRU, Input, Concatenate, Lambda
from tensorflow.keras.optimizers import Adam


def build_conv_lstm_gru_backbone(x):
    x = Conv1D(filters=32, kernel_size=3, activation="relu")(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=64, kernel_size=3, activation="relu")(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = GRU(64, return_sequences=False)(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    return x


def build_mr2g_backbone(feature_dim: int):
    inp = Input(shape=(feature_dim,), name="mr2g_backbone_input")
    x = Lambda(lambda t: tf.expand_dims(t, axis=-1), name="mr2g_expand_dims")(inp)
    x = build_conv_lstm_gru_backbone(x)
    return Model(inp, x, name="mr2g_cnn_lstm_gru_backbone")


class MR2GRobustModel(tf.keras.Model):
    """MR2G: Mask-guided Reliability Residual Reconstruction Gate."""

    def __init__(
        self,
        feature_dim: int,
        bottleneck_dim: int = 64,
        lambda_recon: float = 0.20,
        lambda_consistency: float = 0.25,
        lambda_sparse: float = 0.01,
        learning_rate: float = 1e-3,
    ):
        super().__init__(name="MR2G_Robust_Model")
        self.feature_dim = int(feature_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.lambda_recon = float(lambda_recon)
        self.lambda_consistency = float(lambda_consistency)
        self.lambda_sparse = float(lambda_sparse)

        self.concat = Concatenate(name="mr2g_reliability_concat")
        self.hidden = Dense(max(128, self.feature_dim), activation="relu", name="mr2g_hidden")
        self.hidden_dropout = Dropout(0.20, name="mr2g_hidden_dropout")
        self.reliability_gate = Dense(self.feature_dim, activation="sigmoid", name="mr2g_reliability_gate")

        self.delta_low_rank = Dense(self.bottleneck_dim, activation="relu", name="mr2g_delta_low_rank")
        self.delta_out = Dense(self.feature_dim, activation="linear", name="mr2g_delta_out")

        self.backbone = build_mr2g_backbone(self.feature_dim)
        self.classifier = Dense(1, activation="sigmoid", name="mr2g_output")

        self.compile(optimizer=Adam(learning_rate=learning_rate), run_eagerly=False)

    def _forward_with_aux(self, feat, mask, training=False):
        feat = tf.cast(feat, tf.float32)
        mask = tf.cast(mask, tf.float32)
        abs_feat = tf.abs(feat)

        h = self.concat([feat, mask, abs_feat])
        h = self.hidden(h)
        h = self.hidden_dropout(h, training=training)

        reliability = self.reliability_gate(h)
        delta = self.delta_out(self.delta_low_rank(h))
        recon = feat + delta
        robust_feat = reliability * feat + (1.0 - reliability) * recon
        emb = self.backbone(robust_feat, training=training)
        prob = self.classifier(emb, training=training)
        return prob, recon, reliability, robust_feat

    def call(self, inputs, training=False, return_aux=False):
        if isinstance(inputs, dict):
            feat = inputs.get("joint", None)
            if feat is None:
                feat = inputs.get("feat", None)
            mask = inputs.get("mask", None)
        elif isinstance(inputs, (list, tuple)):
            feat, mask = inputs[0], inputs[1]
        else:
            feat = inputs
            mask = tf.ones_like(feat)

        if mask is None:
            mask = tf.ones_like(feat)

        prob, recon, reliability, robust_feat = self._forward_with_aux(feat, mask, training=training)
        if return_aux:
            aux = {
                "reliability": reliability,
                "recon": recon,
                "robust_feat": robust_feat,
                "reconstruction_abs_mean": tf.reduce_mean(tf.abs(recon - tf.cast(feat, tf.float32)), axis=1, keepdims=True),
                "robust_shift_abs_mean": tf.reduce_mean(tf.abs(robust_feat - tf.cast(feat, tf.float32)), axis=1, keepdims=True),
            }
            return prob, aux
        return prob


def load_mr2g_model(
    model_dir: str | Path,
    weights_path: Optional[str | Path] = None,
    config_path: Optional[str | Path] = None,
):
    """
    Load MR2G from a deployment directory.

    Expected files in model_dir:
    - mr2g_final.weights.h5
    - deploy_config.json
    """
    model_dir = Path(model_dir)
    config_path = Path(config_path) if config_path else model_dir / "deploy_config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = {}

    feature_dim = int(cfg.get("feature_dim", 93))
    weights_path = Path(weights_path) if weights_path else model_dir / cfg.get("weights_file", "mr2g_final.weights.h5")
    if not weights_path.exists():
        raise FileNotFoundError(f"Cannot find MR2G weights: {weights_path}")

    model = MR2GRobustModel(
        feature_dim=feature_dim,
        bottleneck_dim=int(cfg.get("bottleneck_dim", 64)),
        lambda_recon=float(cfg.get("recon_lambda", 0.20)),
        lambda_consistency=float(cfg.get("consistency_lambda", 0.25)),
        lambda_sparse=float(cfg.get("sparse_lambda", 0.01)),
        learning_rate=float(cfg.get("learning_rate", 1e-3)),
    )

    dummy_x = tf.zeros((1, feature_dim), dtype=tf.float32)
    dummy_m = tf.ones((1, feature_dim), dtype=tf.float32)
    _ = model([dummy_x, dummy_m], training=False)
    model.load_weights(str(weights_path))
    return model, cfg


def predict_with_aux(model, x_2d: np.ndarray, mask_2d: Optional[np.ndarray] = None):
    """Return probability and MR2G auxiliary outputs for local inference."""
    x_2d = np.asarray(x_2d, dtype=np.float32)
    if mask_2d is None:
        mask_2d = np.ones_like(x_2d, dtype=np.float32)
    else:
        mask_2d = np.asarray(mask_2d, dtype=np.float32)

    prob, aux = model(
        [tf.convert_to_tensor(x_2d), tf.convert_to_tensor(mask_2d)],
        training=False,
        return_aux=True,
    )
    aux_np = {k: v.numpy() for k, v in aux.items()}
    return prob.numpy().reshape(-1), aux_np
