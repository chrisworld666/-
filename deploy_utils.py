# -*- coding: utf-8 -*-
"""Utilities for local/edge-side PGRG+ inference."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


NORMAL_KEYWORDS = {
    "normal", "benign", "bening", "benigntraffic",
    "non-attack", "non_attack", "safe", "0"
}


def preprocess_mixed_dataframe(x_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert mixed-type raw traffic columns to numeric columns.

    This mirrors the preprocessing logic in the training notebook:
    timestamp-like fields are converted to timestamps, numeric fields are kept,
    and string/category fields are encoded as category codes.
    """
    processed = pd.DataFrame(index=x_raw.index)
    for col in x_raw.columns:
        s = x_raw[col]
        if s.isnull().all():
            continue

        col_lower = str(col).lower()
        if any(k in col_lower for k in ["time", "date", "timestamp"]):
            try:
                dt = pd.to_datetime(s, errors="coerce")
                if dt.notna().sum() > 0:
                    ts = (dt.astype("int64") // 10**9).replace(-9223372037, 0)
                    processed[col] = ts.astype("float32")
                    continue
            except Exception:
                pass

        if pd.api.types.is_numeric_dtype(s):
            processed[col] = pd.to_numeric(s, errors="coerce").fillna(0).astype("float32")
            continue

        try:
            cat = s.astype(str).fillna("missing")
            processed[col] = pd.Categorical(cat).codes.astype("float32")
        except Exception:
            pass

    processed = processed.replace([np.inf, -np.inf], 0).fillna(0)
    return processed


def to_binary_labels(series: pd.Series) -> np.ndarray:
    """0 = Normal/Benign, 1 = Attack."""
    if pd.api.types.is_numeric_dtype(series):
        y = pd.to_numeric(series, errors="coerce").fillna(1)
        return (y != 0).astype("int32").values
    s = series.astype(str).str.strip().str.lower()
    return s.apply(lambda x: 0 if x in NORMAL_KEYWORDS else 1).astype("int32").values


def load_preprocessing_artifacts(model_dir: str | Path):
    """Load selected_features.json and scaler.pkl from a deployment directory."""
    model_dir = Path(model_dir)
    selected_path = model_dir / "selected_features.json"
    scaler_path = model_dir / "scaler.pkl"

    if not selected_path.exists():
        raise FileNotFoundError(f"Cannot find selected feature file: {selected_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Cannot find scaler file: {scaler_path}")

    with open(selected_path, "r", encoding="utf-8") as f:
        selected_features = json.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return selected_features, scaler


def build_raw_observed_mask(raw_df: pd.DataFrame, selected_features: list[str]) -> np.ndarray:
    """
    Build a feature observation mask from raw CSV columns.

    mask=1 means the selected feature is present and not NaN in the input file;
    mask=0 means the feature column is absent or the cell is missing.
    """
    mask = np.zeros((len(raw_df), len(selected_features)), dtype=np.float32)
    for j, feat in enumerate(selected_features):
        if feat in raw_df.columns:
            mask[:, j] = (~raw_df[feat].isna()).astype("float32").values
        else:
            mask[:, j] = 0.0
    return mask


def align_and_scale_features(
    raw_df: pd.DataFrame,
    selected_features: list[str],
    scaler,
    label_col: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Align a raw dataframe to the training feature order and apply the train-only scaler.

    Missing raw features are filled before scaling, then set to 0 in standardized space
    with mask=0, matching the notebook's missing-feature protocol.
    """
    drop_cols = []
    if label_col and label_col in raw_df.columns:
        drop_cols.append(label_col)
    for c in ["Attack_label", "Attack_type", "label", "Label"]:
        if c in raw_df.columns and c not in drop_cols:
            drop_cols.append(c)

    raw_feature_df = raw_df.drop(columns=drop_cols, errors="ignore")
    raw_mask = build_raw_observed_mask(raw_feature_df, selected_features)

    processed = preprocess_mixed_dataframe(raw_feature_df)
    aligned = pd.DataFrame(index=raw_df.index)
    for feat in selected_features:
        aligned[feat] = processed[feat] if feat in processed.columns else 0.0
    aligned = aligned[selected_features].replace([np.inf, -np.inf], 0).fillna(0)

    x_scaled = scaler.transform(aligned).astype("float32")
    x_scaled[raw_mask < 0.5] = 0.0
    return x_scaled, raw_mask.astype("float32"), aligned


def apply_joint_corruption_with_mask_2d(
    x_2d: np.ndarray,
    base_mask: Optional[np.ndarray] = None,
    missing_rate: float = 0.0,
    noise_std: float = 0.0,
    fill_value: float = 0.0,
    feature_fraction: float = 1.0,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optional demo-time perturbation for visualizing robustness.

    This is not required for real inference. It is included so the Streamlit UI can
    show how predictions change when missing rate/noise intensity is adjusted.
    """
    rng = np.random.default_rng(random_state)
    x_joint = np.asarray(x_2d, dtype=np.float32).copy()
    observed_mask = np.ones_like(x_joint, dtype=np.float32) if base_mask is None else np.asarray(base_mask, dtype=np.float32).copy()

    if missing_rate > 0:
        missing = rng.random(x_joint.shape) < float(missing_rate)
        x_joint[missing] = fill_value
        observed_mask[missing] = 0.0

    if noise_std > 0:
        if feature_fraction >= 1.0:
            noise_mask = np.ones_like(x_joint, dtype=bool)
        else:
            noise_mask = rng.random(x_joint.shape) < float(feature_fraction)
        noise_mask = noise_mask & (observed_mask > 0.5)
        noise = rng.normal(loc=0.0, scale=float(noise_std), size=x_joint.shape).astype(np.float32)
        x_joint[noise_mask] += noise[noise_mask]

    return x_joint.astype(np.float32), observed_mask.astype(np.float32)


def make_prediction_table(
    probabilities: np.ndarray,
    aux: dict,
    threshold: float = 0.5,
    latency_ms_per_sample: Optional[float] = None,
) -> pd.DataFrame:
    probabilities = np.asarray(probabilities).reshape(-1)
    pred = (probabilities >= threshold).astype(int)
    out = pd.DataFrame({
        "row_id": np.arange(len(probabilities)),
        "attack_probability": probabilities,
        "pred_label": pred,
        "pred_class": np.where(pred == 1, "Attack", "Normal"),
    })
    if "reliability" in aux:
        out["reliability_mean"] = aux["reliability"].mean(axis=1)
        out["reliability_min"] = aux["reliability"].min(axis=1)
        out["reliability_max"] = aux["reliability"].max(axis=1)
    if "feature_dist_n" in aux:
        out["dist_to_normal_proto"] = aux["feature_dist_n"].reshape(-1)
    if "feature_dist_a" in aux:
        out["dist_to_attack_proto"] = aux["feature_dist_a"].reshape(-1)
    if latency_ms_per_sample is not None:
        out["latency_ms_per_sample"] = float(latency_ms_per_sample)
    return out
