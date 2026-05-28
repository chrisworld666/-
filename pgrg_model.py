# -*- coding: utf-8 -*-
"""
PGRG+ model definition and artifact loader.

This file is generated from the user's training notebook. It recreates the
PGRG_PLUS_SINGLE architecture so that pgrg_final.weights.h5 can be loaded
for local/edge-side inference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM, GRU,
    Input, Concatenate, Lambda
)
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

def build_pgrg_backbone(feature_dim):
    """
    与 baseline / 实验七模块保持 CNN-LSTM-GRU 主干一致，保证比较公平。
    """
    inp = Input(shape=(feature_dim,), name="pgrg_backbone_input")
    x = Lambda(lambda t: tf.expand_dims(t, axis=-1), name="pgrg_expand_dims")(inp)
    x = build_conv_lstm_gru_backbone(x)
    return Model(inp, x, name="pgrg_cnn_lstm_gru_backbone")

class PGRGRobustModel(tf.keras.Model):
    """
    PGRG+: Prototype-Guided Reliability Gating with Multi-Prototypes and Local Abnormality Summary

    给定受扰输入 x 与观测掩码 m：
        d_N, d_A = distance(x, prototype_N / prototype_A)
        r = sigmoid(MLP([x, m, d_N_feat, d_A_feat, d_A_feat - d_N_feat]))
        x_gate = r * m * x
        h = CNN-LSTM-GRU(x_gate)
        y_hat = Classifier([h, distance_summary])

    关键区别：
        PGRG 不生成 x_rec / x_star，不声称恢复原始真实特征；
        它只利用类别原型判断当前输入的可靠性和类别偏离，并在表示空间中抑制不可信维度。
    """
    def __init__(
        self,
        feature_dim,
        normal_proto,
        attack_proto,
        feature_scale,
        lambda_proto=0.15,
        lambda_consistency=0.25,
        lambda_gate=0.01,
        margin=0.50,
        gate_hidden_dim=256,
        proto_context_dim=32,
        enable_global_feature_gate=True,
        local_topk_ratio=0.20,
        learning_rate=1e-3
    ):
        super().__init__(name="PGRG_Robust_Model")
        self.feature_dim = int(feature_dim)
        self.lambda_proto = float(lambda_proto)
        self.lambda_consistency = float(lambda_consistency)
        self.lambda_gate = float(lambda_gate)
        self.margin = float(margin)
        self.enable_global_feature_gate = bool(enable_global_feature_gate)
        self.local_topk_ratio = float(local_topk_ratio)

        normal_proto = np.asarray(normal_proto, dtype=np.float32).reshape(-1, feature_dim)
        attack_proto = np.asarray(attack_proto, dtype=np.float32).reshape(-1, feature_dim)
        feature_scale = np.asarray(feature_scale, dtype=np.float32).reshape(1, 1, -1)

        self.normal_proto = tf.constant(normal_proto, dtype=tf.float32)
        self.attack_proto = tf.constant(attack_proto, dtype=tf.float32)
        self.feature_scale = tf.constant(feature_scale, dtype=tf.float32)

        # 全局可学习特征门控先验：对应 Learnable Feature Gating 思想。
        # 它学习哪些特征在总体上更稳定/更有判别力，再与样本级 reliability gate 相乘。
        self.global_feature_gate_logit = self.add_weight(
            name="pgrg_global_feature_gate_logit",
            shape=(1, self.feature_dim),
            initializer="zeros",
            trainable=True
        )

        self.gate_concat = Concatenate(name="pgrg_gate_concat")
        self.gate_hidden = Dense(gate_hidden_dim, activation="relu", name="pgrg_gate_hidden")
        self.gate_dropout = Dropout(0.20, name="pgrg_gate_dropout")
        self.reliability_gate = Dense(feature_dim, activation="sigmoid", name="pgrg_reliability_gate")

        self.backbone = build_pgrg_backbone(feature_dim)

        # backbone 输出为 128 维。这里设置两个可学习的 embedding 原型，用于端到端原型间隔约束。
        self.embedding_dim = 128
        self.emb_proto_normal = self.add_weight(
            name="pgrg_emb_proto_normal",
            shape=(1, self.embedding_dim),
            initializer="zeros",
            trainable=True
        )
        self.emb_proto_attack = self.add_weight(
            name="pgrg_emb_proto_attack",
            shape=(1, self.embedding_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.05),
            trainable=True
        )

        self.proto_context_dense = Dense(proto_context_dim, activation="relu", name="pgrg_proto_context")
        self.fusion_concat = Concatenate(name="pgrg_fusion_concat")
        self.fusion_dense = Dense(64, activation="relu", name="pgrg_fusion_dense")
        self.fusion_dropout = Dropout(0.20, name="pgrg_fusion_dropout")
        self.classifier = Dense(1, activation="sigmoid", name="pgrg_output")

        self.loss_fn = tf.keras.losses.BinaryCrossentropy()

        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.cls_loss_tracker = tf.keras.metrics.Mean(name="cls_loss")
        self.proto_loss_tracker = tf.keras.metrics.Mean(name="proto_loss")
        self.cons_loss_tracker = tf.keras.metrics.Mean(name="consistency_loss")
        self.gate_loss_tracker = tf.keras.metrics.Mean(name="gate_loss")

        self.metric_acc = tf.keras.metrics.BinaryAccuracy(name="accuracy")
        self.metric_auc = tf.keras.metrics.AUC(name="auc")
        self.metric_precision = tf.keras.metrics.Precision(name="precision")
        self.metric_recall = tf.keras.metrics.Recall(name="recall")

        self.compile(optimizer=Adam(learning_rate=learning_rate), run_eagerly=False)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.cls_loss_tracker,
            self.proto_loss_tracker,
            self.cons_loss_tracker,
            self.gate_loss_tracker,
            self.metric_acc,
            self.metric_auc,
            self.metric_precision,
            self.metric_recall
        ]

    def _feature_proto_distance(self, feat, mask):
        feat = tf.cast(feat, tf.float32)
        mask = tf.cast(mask, tf.float32)

        # 多原型距离：feat [B,F], proto [K,F] -> diff [B,K,F]
        feat_exp = tf.expand_dims(feat, axis=1)
        mask_exp = tf.expand_dims(mask, axis=1)

        diff_n_all = tf.abs((feat_exp - tf.expand_dims(self.normal_proto, axis=0)) / self.feature_scale)
        diff_a_all = tf.abs((feat_exp - tf.expand_dims(self.attack_proto, axis=0)) / self.feature_scale)

        # 只对观测到的维度计算距离，避免缺失填充值扭曲原型距离。
        denom = tf.reduce_sum(mask_exp, axis=2) + 1e-6
        dist_n_all = tf.reduce_sum(diff_n_all * mask_exp, axis=2) / denom
        dist_a_all = tf.reduce_sum(diff_a_all * mask_exp, axis=2) / denom

        # soft-min 聚合：既保留“最近原型”思想，又比 hard min 更平滑。
        w_n = tf.nn.softmax(-dist_n_all, axis=1)
        w_a = tf.nn.softmax(-dist_a_all, axis=1)

        dist_n = tf.reduce_sum(w_n * dist_n_all, axis=1, keepdims=True)
        dist_a = tf.reduce_sum(w_a * dist_a_all, axis=1, keepdims=True)
        diff_n_feat = tf.reduce_sum(tf.expand_dims(w_n, axis=2) * diff_n_all, axis=1)
        diff_a_feat = tf.reduce_sum(tf.expand_dims(w_a, axis=2) * diff_a_all, axis=1)

        miss_ratio = 1.0 - tf.reduce_mean(mask, axis=1, keepdims=True)

        # 局部异常摘要：PLAG 等新工作强调特征级局部异常，而不是只看全局样本异常分数。
        local_gap_feat = tf.abs(diff_a_feat - diff_n_feat) * mask
        local_gap_mean = tf.reduce_sum(local_gap_feat, axis=1, keepdims=True) / (tf.reduce_sum(mask, axis=1, keepdims=True) + 1e-6)
        k = max(1, int(self.feature_dim * self.local_topk_ratio))
        local_topk_mean = tf.reduce_mean(tf.nn.top_k(local_gap_feat, k=k).values, axis=1, keepdims=True)

        proto_summary = tf.concat([dist_n, dist_a, dist_n - dist_a, miss_ratio, local_gap_mean, local_topk_mean], axis=1)
        return diff_n_feat, diff_a_feat, dist_n, dist_a, proto_summary

    def _embedding_proto_distance(self, emb):
        emb_dist_n = tf.reduce_mean(tf.square(emb - self.emb_proto_normal), axis=1, keepdims=True)
        emb_dist_a = tf.reduce_mean(tf.square(emb - self.emb_proto_attack), axis=1, keepdims=True)
        return emb_dist_n, emb_dist_a

    def _forward_with_aux(self, feat, mask, training=False):
        feat = tf.cast(feat, tf.float32)
        mask = tf.cast(mask, tf.float32)

        diff_n_feat, diff_a_feat, dist_n, dist_a, proto_summary = self._feature_proto_distance(feat, mask)
        proto_gap_feat = diff_a_feat - diff_n_feat

        gate_input = self.gate_concat([
            feat,
            mask,
            diff_n_feat * mask,
            diff_a_feat * mask,
            proto_gap_feat * mask
        ])
        g = self.gate_hidden(gate_input)
        g = self.gate_dropout(g, training=training)

        # reliability 只允许作用在观测到的维度；缺失维度不强行补值。
        sample_gate = self.reliability_gate(g)
        if self.enable_global_feature_gate:
            global_gate = tf.sigmoid(self.global_feature_gate_logit)
            reliability = sample_gate * global_gate * mask
        else:
            reliability = sample_gate * mask
        gated_feat = reliability * feat

        emb = self.backbone(gated_feat, training=training)
        emb_dist_n, emb_dist_a = self._embedding_proto_distance(emb)

        proto_context = self.proto_context_dense(
            tf.concat([proto_summary, emb_dist_n, emb_dist_a, emb_dist_n - emb_dist_a], axis=1),
            training=training
        )
        fused = self.fusion_concat([emb, proto_context])
        fused = self.fusion_dense(fused)
        fused = self.fusion_dropout(fused, training=training)
        prob = self.classifier(fused, training=training)

        aux = {
            "reliability": reliability,
            "gated_feat": gated_feat,
            "feature_dist_n": dist_n,
            "feature_dist_a": dist_a,
            "emb_dist_n": emb_dist_n,
            "emb_dist_a": emb_dist_a,
            "proto_summary": proto_summary
        }
        return prob, aux

    def call(self, inputs, training=False, return_aux=False):
        # 推理期支持：
        # 1) [X_joint, mask]
        # 2) {"joint": X_joint, "mask": mask}
        # 3) 单输入 X_joint，此时默认 mask 全 1
        if isinstance(inputs, dict):
            feat = inputs.get("joint", None)
            if feat is None:
                feat = inputs.get("feat", None)
            if feat is None:
                feat = inputs.get("clean", None)
            mask = inputs.get("mask", None)
        elif isinstance(inputs, (list, tuple)):
            feat = inputs[0]
            mask = inputs[1] if len(inputs) > 1 else None
        else:
            feat = inputs
            mask = None

        if mask is None:
            mask = tf.ones_like(feat)

        prob, aux = self._forward_with_aux(feat, mask, training=training)
        if return_aux:
            return prob, aux
        return prob

    def _unpack_train_inputs(self, x):
        if isinstance(x, dict):
            x_clean = tf.cast(x["clean"], tf.float32)
            x_joint = tf.cast(x["joint"], tf.float32)
            mask = tf.cast(x["mask"], tf.float32)
        elif isinstance(x, (list, tuple)) and len(x) >= 3:
            x_clean = tf.cast(x[0], tf.float32)
            x_joint = tf.cast(x[1], tf.float32)
            mask = tf.cast(x[2], tf.float32)
        else:
            raise ValueError("PGRG 训练输入必须包含 clean / joint / mask。")
        return x_clean, x_joint, mask

    def _proto_margin_loss(self, y, emb_dist_n, emb_dist_a):
        y = tf.cast(tf.reshape(y, (-1, 1)), tf.float32)
        same_dist = tf.where(y > 0.5, emb_dist_a, emb_dist_n)
        diff_dist = tf.where(y > 0.5, emb_dist_n, emb_dist_a)
        return tf.reduce_mean(tf.nn.relu(self.margin + same_dist - diff_dist))

    def train_step(self, data):
        x, y = data
        x_clean, x_joint, mask = self._unpack_train_inputs(x)
        y = tf.cast(tf.reshape(y, (-1, 1)), tf.float32)
        clean_mask = tf.ones_like(x_clean)

        with tf.GradientTape() as tape:
            clean_prob, clean_aux = self._forward_with_aux(x_clean, clean_mask, training=True)
            joint_prob, joint_aux = self._forward_with_aux(x_joint, mask, training=True)

            cls_loss = self.loss_fn(y, joint_prob)
            proto_loss = self._proto_margin_loss(y, joint_aux["emb_dist_n"], joint_aux["emb_dist_a"])
            cons_loss = tf.reduce_mean(tf.square(tf.stop_gradient(clean_prob) - joint_prob))

            # 让 gate 更有选择性，避免全部维度都卡在 0.5；不做“修复”，只做可靠性选择。
            r = joint_aux["reliability"]
            gate_entropy = tf.reduce_mean(r * (1.0 - r))
            gate_retention = tf.reduce_mean(1.0 - r) * 0.05
            gate_loss = gate_entropy + gate_retention

            total_loss = (
                cls_loss
                + self.lambda_proto * proto_loss
                + self.lambda_consistency * cons_loss
                + self.lambda_gate * gate_loss
            )

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.total_loss_tracker.update_state(total_loss)
        self.cls_loss_tracker.update_state(cls_loss)
        self.proto_loss_tracker.update_state(proto_loss)
        self.cons_loss_tracker.update_state(cons_loss)
        self.gate_loss_tracker.update_state(gate_loss)

        self.metric_acc.update_state(y, joint_prob)
        self.metric_auc.update_state(y, joint_prob)
        self.metric_precision.update_state(y, joint_prob)
        self.metric_recall.update_state(y, joint_prob)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        x_clean, x_joint, mask = self._unpack_train_inputs(x)
        y = tf.cast(tf.reshape(y, (-1, 1)), tf.float32)
        clean_mask = tf.ones_like(x_clean)

        clean_prob, clean_aux = self._forward_with_aux(x_clean, clean_mask, training=False)
        joint_prob, joint_aux = self._forward_with_aux(x_joint, mask, training=False)

        cls_loss = self.loss_fn(y, joint_prob)
        proto_loss = self._proto_margin_loss(y, joint_aux["emb_dist_n"], joint_aux["emb_dist_a"])
        cons_loss = tf.reduce_mean(tf.square(tf.stop_gradient(clean_prob) - joint_prob))

        r = joint_aux["reliability"]
        gate_loss = tf.reduce_mean(r * (1.0 - r)) + 0.05 * tf.reduce_mean(1.0 - r)

        total_loss = (
            cls_loss
            + self.lambda_proto * proto_loss
            + self.lambda_consistency * cons_loss
            + self.lambda_gate * gate_loss
        )

        self.total_loss_tracker.update_state(total_loss)
        self.cls_loss_tracker.update_state(cls_loss)
        self.proto_loss_tracker.update_state(proto_loss)
        self.cons_loss_tracker.update_state(cons_loss)
        self.gate_loss_tracker.update_state(gate_loss)

        self.metric_acc.update_state(y, joint_prob)
        self.metric_auc.update_state(y, joint_prob)
        self.metric_precision.update_state(y, joint_prob)
        self.metric_recall.update_state(y, joint_prob)
        return {m.name: m.result() for m in self.metrics}



def _read_pgrg_prototypes(prototype_csv: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read pgrg_feature_prototypes.csv generated by the notebook."""
    prototype_csv = Path(prototype_csv)
    if not prototype_csv.exists():
        raise FileNotFoundError(f"Cannot find prototype file: {prototype_csv}")

    proto_df = pd.read_csv(prototype_csv)
    required = {"Class", "Prototype_ID", "Feature_Index", "Prototype_Value", "Feature_Scale"}
    missing = required - set(proto_df.columns)
    if missing:
        raise ValueError(f"Prototype CSV missing columns: {sorted(missing)}")

    def pivot_class(class_name: str) -> np.ndarray:
        sub = proto_df[proto_df["Class"].astype(str).str.lower() == class_name.lower()].copy()
        if sub.empty:
            raise ValueError(f"No {class_name} prototypes found in {prototype_csv}")
        wide = sub.pivot(index="Prototype_ID", columns="Feature_Index", values="Prototype_Value")
        wide = wide.sort_index(axis=0).sort_index(axis=1)
        return wide.values.astype(np.float32)

    normal_proto = pivot_class("Normal")
    attack_proto = pivot_class("Attack")
    scale = (
        proto_df[["Feature_Index", "Feature_Scale"]]
        .drop_duplicates(subset=["Feature_Index"])
        .sort_values("Feature_Index")["Feature_Scale"]
        .values.astype(np.float32)
    )
    return normal_proto, attack_proto, scale


def load_pgrg_model(
    model_dir: str | Path,
    weights_path: Optional[str | Path] = None,
    prototype_csv: Optional[str | Path] = None,
    config_path: Optional[str | Path] = None,
):
    """
    Load PGRG+ from a deployment directory.

    Expected files in model_dir:
    - pgrg_final.weights.h5
    - pgrg_feature_prototypes.csv
    - deploy_config.json
    """
    model_dir = Path(model_dir)
    config_path = Path(config_path) if config_path else model_dir / "deploy_config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = {}

    weights_path = Path(weights_path) if weights_path else model_dir / cfg.get("weights_file", "pgrg_final.weights.h5")
    prototype_csv = Path(prototype_csv) if prototype_csv else model_dir / cfg.get("prototype_file", "pgrg_feature_prototypes.csv")

    if not weights_path.exists():
        raise FileNotFoundError(f"Cannot find PGRG+ weights: {weights_path}")

    normal_proto, attack_proto, feature_scale = _read_pgrg_prototypes(prototype_csv)
    feature_dim = int(cfg.get("feature_dim", normal_proto.shape[1]))

    model = PGRGRobustModel(
        feature_dim=feature_dim,
        normal_proto=normal_proto,
        attack_proto=attack_proto,
        feature_scale=feature_scale,
        lambda_proto=float(cfg.get("proto_lambda", 0.15)),
        lambda_consistency=float(cfg.get("consistency_lambda", 0.25)),
        lambda_gate=float(cfg.get("gate_lambda", 0.01)),
        margin=float(cfg.get("margin", 0.50)),
        gate_hidden_dim=int(cfg.get("gate_hidden_dim", 256)),
        proto_context_dim=int(cfg.get("proto_context_dim", 32)),
        enable_global_feature_gate=bool(cfg.get("enable_global_feature_gate", True)),
        local_topk_ratio=float(cfg.get("local_topk_ratio", 0.20)),
        learning_rate=float(cfg.get("learning_rate", 1e-3)),
    )

    # Build variables before loading weights.
    dummy_x = tf.zeros((1, feature_dim), dtype=tf.float32)
    dummy_m = tf.ones((1, feature_dim), dtype=tf.float32)
    _ = model([dummy_x, dummy_m], training=False)
    model.load_weights(str(weights_path))
    return model, cfg


def predict_with_aux(model, x_2d: np.ndarray, mask_2d: Optional[np.ndarray] = None):
    """Return probability and PGRG+ auxiliary outputs for local inference."""
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
