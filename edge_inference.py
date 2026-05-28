# -*- coding: utf-8 -*-
"""
Unified local/edge-side inference script for MR2G and PGRG/PGRG+.

Examples:
    python edge_inference.py --model both --input your_edge_flow.csv
    python edge_inference.py --model mr2g --input your_edge_flow.csv --output mr2g_predictions.csv
    python edge_inference.py --model pgrg --input your_edge_flow.csv --missing-rate 0.3 --noise-std 0.2
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score

from deploy_utils import (
    align_and_scale_features,
    apply_joint_corruption_with_mask_2d,
    load_preprocessing_artifacts,
    make_prediction_table,
    to_binary_labels,
)
from mr2g_model import load_mr2g_model, predict_with_aux as predict_mr2g_with_aux
from pgrg_model import load_pgrg_model, predict_with_aux as predict_pgrg_with_aux


LABEL_CANDIDATES = ["Attack_type", "Attack_label", "label", "Label", "attack", "Attack", "class", "Class"]
MODEL_ALIASES = {
    "mr2g": "MR2G",
    "pgrg": "PGRG",
}


def find_label_col(df: pd.DataFrame, explicit: str | None = None) -> str | None:
    if explicit and explicit in df.columns:
        return explicit
    for col in LABEL_CANDIDATES:
        if col in df.columns:
            return col
    return None


def _default_model_dir(root_dir: Path, model_key: str) -> Path:
    if model_key == "mr2g":
        return root_dir / "edge_deploy_mr2g"
    if model_key == "pgrg":
        # Prefer the PGRG directory. Keep edge_deploy_pgrg_plus as a backward-compatible alias.
        pgrg_dir = root_dir / "edge_deploy_pgrg"
        if pgrg_dir.exists():
            return pgrg_dir
        return root_dir / "edge_deploy_pgrg_plus"
    raise ValueError(f"Unsupported model key: {model_key}")


def _resolve_model_dir(args, model_key: str) -> Path:
    explicit = getattr(args, f"{model_key}_dir", None)
    if explicit:
        return Path(explicit)
    root_dir = Path(args.root_dir)
    return _default_model_dir(root_dir, model_key)


def _load_model_and_predict(
    model_key: str,
    model_dir: Path,
    x_model: np.ndarray,
    mask_model: np.ndarray,
    threshold: float,
    weights_path: str | None = None,
) -> Tuple[pd.DataFrame, dict, dict, float]:
    if model_key == "mr2g":
        model, cfg = load_mr2g_model(model_dir=model_dir, weights_path=weights_path or None)
        predict_fn = predict_mr2g_with_aux
    elif model_key == "pgrg":
        model, cfg = load_pgrg_model(model_dir=model_dir, weights_path=weights_path or None)
        predict_fn = predict_pgrg_with_aux
    else:
        raise ValueError(f"Unsupported model: {model_key}")

    threshold = float(threshold if threshold is not None else cfg.get("threshold", 0.5))
    start = time.perf_counter()
    probabilities, aux = predict_fn(model, x_model, mask_model)
    elapsed = time.perf_counter() - start
    latency_ms = elapsed / max(1, len(x_model)) * 1000.0

    pred_df = make_prediction_table(probabilities, aux, threshold=threshold, latency_ms_per_sample=latency_ms)
    pred_df.insert(0, "model", MODEL_ALIASES[model_key])

    # MR2G-specific auxiliary summaries.
    if "reconstruction_abs_mean" in aux:
        pred_df["reconstruction_abs_mean"] = aux["reconstruction_abs_mean"].reshape(-1)
    if "robust_shift_abs_mean" in aux:
        pred_df["robust_shift_abs_mean"] = aux["robust_shift_abs_mean"].reshape(-1)

    return pred_df, aux, cfg, latency_ms


def _metric_row(y_true: np.ndarray, pred_df: pd.DataFrame) -> dict:
    y_pred = pred_df["pred_label"].values
    return {
        "model": pred_df["model"].iloc[0],
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Attack_Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Mean_Attack_Probability": float(pred_df["attack_probability"].mean()),
        "Mean_Latency_ms_per_sample": float(pred_df["latency_ms_per_sample"].iloc[0]),
    }


def run_inference(args):
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    selected_features, scaler = load_preprocessing_artifacts(_resolve_model_dir(args, "pgrg") if args.model == "pgrg" else _resolve_model_dir(args, "mr2g"))

    raw_df = pd.read_csv(input_path, low_memory=False)
    if args.max_rows and args.max_rows > 0:
        raw_df = raw_df.head(args.max_rows).copy()

    label_col = find_label_col(raw_df, args.label_col)
    x_scaled, base_mask, _ = align_and_scale_features(raw_df, selected_features, scaler, label_col=label_col)
    x_model, mask_model = apply_joint_corruption_with_mask_2d(
        x_scaled,
        base_mask=base_mask,
        missing_rate=float(args.missing_rate),
        noise_std=float(args.noise_std),
        fill_value=0.0,
        feature_fraction=float(args.feature_fraction),
        random_state=int(args.random_state),
    )

    model_keys = ["mr2g", "pgrg"] if args.model == "both" else [args.model]
    outputs = []
    metric_rows = []

    for model_key in model_keys:
        model_dir = _resolve_model_dir(args, model_key)
        weights = args.mr2g_weights if model_key == "mr2g" else args.pgrg_weights
        print(f"Running {MODEL_ALIASES[model_key]} from {model_dir}")
        pred_df, aux, cfg, latency_ms = _load_model_and_predict(
            model_key=model_key,
            model_dir=model_dir,
            x_model=x_model,
            mask_model=mask_model,
            threshold=float(args.threshold),
            weights_path=weights,
        )
        outputs.append(pred_df)
        print(f"  Rows: {len(pred_df)}")
        print(f"  Predicted Attack: {int((pred_df['pred_label'] == 1).sum())}")
        print(f"  Mean latency: {latency_ms:.4f} ms/sample")
        if label_col is not None:
            metric_rows.append(_metric_row(to_binary_labels(raw_df[label_col]), pred_df))

    final_df = pd.concat(outputs, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved predictions to: {output_path}")

    if metric_rows:
        metric_df = pd.DataFrame(metric_rows)
        metric_path = output_path.with_name(output_path.stem + "_metrics.csv")
        metric_df.to_csv(metric_path, index=False, encoding="utf-8-sig")
        print(f"Saved metrics to: {metric_path}")
        print(metric_df.to_string(index=False))


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Unified MR2G / PGRG+ local edge-side inference")
    parser.add_argument("--model", choices=["mr2g", "pgrg", "both"], default="both", help="Which model to run")
    parser.add_argument("--root-dir", default="results_baseline_edge_stress_tests", help="Root directory containing edge_deploy_mr2g and edge_deploy_pgrg / edge_deploy_pgrg_plus")
    parser.add_argument("--mr2g-dir", default=None, help="Optional explicit MR2G deployment artifact directory")
    parser.add_argument("--pgrg-dir", default=None, help="Optional explicit PGRG+ deployment artifact directory")
    parser.add_argument("--mr2g-weights", default=None, help="Optional explicit mr2g_final.weights.h5 path")
    parser.add_argument("--pgrg-weights", default=None, help="Optional explicit pgrg_final.weights.h5 path")
    parser.add_argument("--input", required=True, help="Input traffic CSV path")
    parser.add_argument("--output", default="predictions_mr2g_pgrg.csv", help="Output predictions CSV path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--label-col", default=None, help="Optional label column for evaluation")
    parser.add_argument("--max-rows", type=int, default=5000, help="Limit rows for local demo; use 0 for all rows")
    parser.add_argument("--missing-rate", type=float, default=0.0, help="Optional demo missing-rate perturbation")
    parser.add_argument("--noise-std", type=float, default=0.0, help="Optional demo Gaussian-noise perturbation")
    parser.add_argument("--feature-fraction", type=float, default=1.0, help="Feature fraction affected by demo noise")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for demo perturbation")
    return parser


if __name__ == "__main__":
    run_inference(build_arg_parser().parse_args())
