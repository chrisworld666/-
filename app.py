# -*- coding: utf-8 -*-
"""Streamlit UI for the integrated MR2G + PGRG/PGRG+ edge-side intrusion detection demo."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score

from deploy_utils import (
    align_and_scale_features,
    apply_joint_corruption_with_mask_2d,
    load_preprocessing_artifacts,
    make_prediction_table,
    to_binary_labels,
)
from edge_inference import find_label_col
from mr2g_model import load_mr2g_model, predict_with_aux as predict_mr2g_with_aux
from pgrg_model import load_pgrg_model, predict_with_aux as predict_pgrg_with_aux


st.set_page_config(page_title="MR²G + PGRG/PGRG+ 端侧入侵检测", page_icon="🛡️", layout="wide")

MODEL_NAMES = {
    "mr2g": "MR²G",
    "pgrg": "PGRG/PGRG+",
}


@st.cache_resource(show_spinner=False)
def cached_load_mr2g(model_dir: str, weights_path: str | None):
    return load_mr2g_model(model_dir=model_dir, weights_path=weights_path or None)


@st.cache_resource(show_spinner=False)
def cached_load_pgrg(model_dir: str, weights_path: str | None):
    return load_pgrg_model(model_dir=model_dir, weights_path=weights_path or None)


@st.cache_data(show_spinner=False)
def load_compare_assets():
    base = Path(__file__).resolve().parent / "assets"
    out = {}
    for name in ["comparison_summary.csv", "resource_comparison.csv"]:
        path = base / name
        if path.exists():
            out[name] = pd.read_csv(path)
    return out


def run_one_model(model_key, model_dir, weights_path, x_model, mask_model, threshold):
    if model_key == "mr2g":
        model, cfg = cached_load_mr2g(model_dir, weights_path.strip() or None)
        predict_fn = predict_mr2g_with_aux
    elif model_key == "pgrg":
        model, cfg = cached_load_pgrg(model_dir, weights_path.strip() or None)
        predict_fn = predict_pgrg_with_aux
    else:
        raise ValueError(model_key)

    start = time.perf_counter()
    prob, aux = predict_fn(model, x_model, mask_model)
    elapsed = time.perf_counter() - start
    latency_ms = elapsed / max(1, len(x_model)) * 1000.0
    pred_df = make_prediction_table(prob, aux, threshold=threshold, latency_ms_per_sample=latency_ms)
    pred_df.insert(0, "model", MODEL_NAMES[model_key])
    if "reconstruction_abs_mean" in aux:
        pred_df["reconstruction_abs_mean"] = aux["reconstruction_abs_mean"].reshape(-1)
    if "robust_shift_abs_mean" in aux:
        pred_df["robust_shift_abs_mean"] = aux["robust_shift_abs_mean"].reshape(-1)
    return pred_df


def render_prediction_charts(pred_df: pd.DataFrame):
    st.subheader("攻击概率变化")
    prob_chart = pred_df.pivot(index="row_id", columns="model", values="attack_probability")
    st.line_chart(prob_chart)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Normal / Attack 预测数量")
        count_df = pred_df.groupby(["model", "pred_class"]).size().reset_index(name="数量")
        count_df["模型-类别"] = count_df["model"] + "-" + count_df["pred_class"]
        st.bar_chart(count_df.set_index("模型-类别")[["数量"]])
    with c2:
        st.subheader("平均推理耗时")
        latency_df = pred_df.groupby("model")["latency_ms_per_sample"].mean().reset_index()
        st.bar_chart(latency_df.set_index("model"))

    if "reliability_mean" in pred_df.columns:
        st.subheader("可靠性门控均值")
        gate_chart = pred_df.pivot(index="row_id", columns="model", values="reliability_mean")
        st.line_chart(gate_chart)

    pgrg_part = pred_df[pred_df["model"].eq("PGRG/PGRG+")]
    if not pgrg_part.empty and {"dist_to_normal_proto", "dist_to_attack_proto"}.issubset(pgrg_part.columns):
        st.subheader("PGRG/PGRG+ 原型距离对比")
        dist_df = pgrg_part[["row_id", "dist_to_normal_proto", "dist_to_attack_proto"]].set_index("row_id")
        st.line_chart(dist_df)

    mr2g_part = pred_df[pred_df["model"].eq("MR²G")]
    if not mr2g_part.empty and "reconstruction_abs_mean" in mr2g_part.columns:
        st.subheader("MR²G 重构/鲁棒特征变化幅度")
        cols = ["row_id", "reconstruction_abs_mean"]
        if "robust_shift_abs_mean" in mr2g_part.columns:
            cols.append("robust_shift_abs_mean")
        st.line_chart(mr2g_part[cols].set_index("row_id"))


def render_prediction_tab(config):
    uploaded = st.file_uploader("上传 IoT 流量 CSV 文件", type=["csv"])
    if uploaded is None:
        st.info("请上传包含 93 个训练特征或原始 Edge-IIoTset 字段的 CSV 文件。")
        return

    try:
        raw_df = pd.read_csv(uploaded, low_memory=False)
    except Exception as e:
        st.error(f"CSV 读取失败：{e}")
        return

    raw_df = raw_df.head(int(config["max_rows"])).copy()
    st.write(f"已读取 {len(raw_df)} 条样本，{raw_df.shape[1]} 个字段。")
    with st.expander("查看原始数据预览"):
        st.dataframe(raw_df.head(20), use_container_width=True)

    if not st.button("开始端侧推理", type="primary"):
        return

    try:
        # 两个模型共用同一份 selected_features/scaler。优先读取 MR2G 目录；如果不存在，则读取 PGRG/PGRG+ 目录。
        artifact_dir_for_preprocess = config["mr2g_dir"] if Path(config["mr2g_dir"]).exists() else config["pgrg_dir"]
        selected_features, scaler = load_preprocessing_artifacts(artifact_dir_for_preprocess)
        label_col = find_label_col(raw_df, None)
        x_scaled, base_mask, _ = align_and_scale_features(raw_df, selected_features, scaler, label_col=label_col)
        x_model, mask_model = apply_joint_corruption_with_mask_2d(
            x_scaled,
            base_mask=base_mask,
            missing_rate=float(config["missing_rate"]),
            noise_std=float(config["noise_std"]),
            fill_value=0.0,
            feature_fraction=1.0,
            random_state=int(config["random_state"]),
        )

        outputs = []
        with st.spinner("正在本地推理..."):
            if config["model_choice"] in ["MR²G", "两个都跑"]:
                outputs.append(run_one_model("mr2g", config["mr2g_dir"], config["mr2g_weights"], x_model, mask_model, config["threshold"]))
            if config["model_choice"] in ["PGRG/PGRG+", "两个都跑"]:
                outputs.append(run_one_model("pgrg", config["pgrg_dir"], config["pgrg_weights"], x_model, mask_model, config["threshold"]))

        pred_df = pd.concat(outputs, ignore_index=True)
        summary = pred_df.groupby("model").agg(
            样本数=("row_id", "count"),
            预测Attack=("pred_label", "sum"),
            平均攻击概率=("attack_probability", "mean"),
            平均可靠性门控=("reliability_mean", "mean"),
            平均推理耗时ms=("latency_ms_per_sample", "mean"),
        ).reset_index()

        st.subheader("推理汇总")
        st.dataframe(summary, use_container_width=True)

        if label_col is not None:
            y_true = to_binary_labels(raw_df[label_col])
            rows = []
            for model_name, sub in pred_df.groupby("model"):
                y_pred = sub["pred_label"].values
                rows.append({
                    "model": model_name,
                    "Accuracy": accuracy_score(y_true, y_pred),
                    "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
                    "Precision": precision_score(y_true, y_pred, zero_division=0),
                    "Attack Recall": recall_score(y_true, y_pred, zero_division=0),
                    "F1": f1_score(y_true, y_pred, zero_division=0),
                })
            st.subheader("带标签数据上的临时评估")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        render_prediction_charts(pred_df)
        st.subheader("预测结果明细")
        st.dataframe(pred_df.head(300), use_container_width=True)
        csv_bytes = pred_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("下载预测结果 CSV", data=csv_bytes, file_name="mr2g_pgrg_predictions.csv", mime="text/csv")
    except Exception as e:
        st.error("推理失败。请确认两个模型目录中的权重、配置、selected_features.json 和 scaler.pkl 已正确导出。")
        st.exception(e)


def render_compare_assets():
    assets = load_compare_assets()
    st.subheader("MR²G / PGRG/PGRG+ / 四模块软投票论文结果对比")
    st.caption("这里直接使用你前面训练后的汇总表，用于说明单模块部署候选模型和四模块集成模型的性能/开销差异。")

    if "comparison_summary.csv" in assets:
        df = assets["comparison_summary.csv"].copy()
        alias_col = "Model_Alias" if "Model_Alias" in df.columns else "Model"
        metric_cols = [c for c in ["Accuracy", "Balanced_Accuracy", "Recall", "Precision", "F1", "ROC_AUC"] if c in df.columns]
        if metric_cols:
            st.markdown("**综合性能均值**")
            st.bar_chart(df.set_index(alias_col)[metric_cols])
        st.dataframe(df, use_container_width=True)

    if "resource_comparison.csv" in assets:
        res = assets["resource_comparison.csv"].copy()
        alias_col = "Model_Alias" if "Model_Alias" in res.columns else "Model"
        res_cols = [c for c in ["Module_Count", "Parameter_Count", "Mean_Test_Time(s)"] if c in res.columns]
        if res_cols:
            st.markdown("**资源与推理开销**")
            st.bar_chart(res.set_index(alias_col)[res_cols])
        st.dataframe(res, use_container_width=True)


def render_help():
    st.markdown(
        """
### 目录要求
默认情况下，本系统读取：

```text
results_baseline_edge_stress_tests/
├── edge_deploy_mr2g/
│   ├── mr2g_final.weights.h5
│   ├── selected_features.json
│   ├── scaler.pkl
│   └── deploy_config.json
└── edge_deploy_pgrg/
    ├── pgrg_final.weights.h5
    ├── pgrg_feature_prototypes.csv
    ├── selected_features.json
    ├── scaler.pkl
    └── deploy_config.json
```

### 推荐论文表述
本系统是“面向端侧部署的 MR²G 与 PGRG/PGRG+ 可视化原型系统”。它用于展示云端训练后的模型可以迁移到本地端侧推理流程中，并能直观比较两个单模块候选模型的检测结果、攻击概率、可靠性门控和推理耗时。

不要写成“已经完成真实工业端部署上线”，建议写“端侧迁移可行性验证”或“本地轻量推理原型”。

### 启动方式
```bash
streamlit run app.py
```
        """
    )


def main():
    st.title("MR²G + PGRG/PGRG+ 端侧入侵检测可视化原型")
    st.markdown("本系统整合两个单模块候选模型，用于展示 **云端训练、本地端侧推理、可视化对比** 的毕业设计工程环节。")

    with st.sidebar:
        st.header("模型选择")
        model_choice = st.selectbox("运行模型", ["两个都跑", "MR²G", "PGRG/PGRG+"])

        st.header("部署目录")
        root = st.text_input("结果根目录", "results_baseline_edge_stress_tests")
        mr2g_dir = st.text_input("MR²G 部署目录", str(Path(root) / "edge_deploy_mr2g"))
        pgrg_dir = st.text_input("PGRG/PGRG+ 部署目录", str(Path(root) / "edge_deploy_pgrg"))
        mr2g_weights = st.text_input("MR²G 权重路径，可留空", "")
        pgrg_weights = st.text_input("PGRG/PGRG+ 权重路径，可留空", "")

        st.header("推理配置")
        threshold = st.slider("分类阈值", 0.0, 1.0, 0.5, 0.01)
        max_rows = st.number_input("最多读取样本数", min_value=1, max_value=20000, value=1000, step=100)

        st.header("扰动模拟，仅用于可视化")
        missing_rate = st.slider("模拟特征缺失率", 0.0, 0.7, 0.0, 0.05)
        noise_std = st.slider("模拟噪声标准差", 0.0, 0.7, 0.0, 0.05)
        random_state = st.number_input("随机种子", min_value=1, max_value=999999, value=42, step=1)

    config = dict(
        model_choice=model_choice,
        mr2g_dir=mr2g_dir,
        pgrg_dir=pgrg_dir,
        mr2g_weights=mr2g_weights,
        pgrg_weights=pgrg_weights,
        threshold=threshold,
        max_rows=max_rows,
        missing_rate=missing_rate,
        noise_std=noise_std,
        random_state=random_state,
    )

    tab_predict, tab_compare, tab_help = st.tabs(["样本预测", "论文结果对比", "使用说明"])
    with tab_predict:
        render_prediction_tab(config)
    with tab_compare:
        render_compare_assets()
    with tab_help:
        render_help()


if __name__ == "__main__":
    main()
