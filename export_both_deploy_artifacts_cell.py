# =========================
# 端侧部署导出单元：MR²G + PGRG/PGRG+ 两个模型统一导出（稳健版）
# 作用：把训练好的权重、特征顺序、标准化器和配置文件导出到端侧部署目录。
# 说明：本单元可单独运行；如果 RESULT_DIR / edge_data 等变量尚未定义，会自动使用默认路径或给出明确提示。
# =========================
import os
import json
import pickle
import shutil
from pathlib import Path

# 0) 路径兜底：避免 NameError: RESULT_DIR is not defined
RESULT_DIR = globals().get("RESULT_DIR", "results_baseline_edge_stress_tests")
RESULT_DIR = str(RESULT_DIR)
os.makedirs(RESULT_DIR, exist_ok=True)

MR2G_RESULT_DIR = globals().get(
    "MR2G_RESULT_DIR",
    os.path.join(RESULT_DIR, "exp7_mr2g_single_module")
)
PGRG_RESULT_DIR = globals().get(
    "PGRG_RESULT_DIR",
    os.path.join(RESULT_DIR, "exp7_pgrg_single_module")
)

BASELINE_SELECTED_FEATURES_PATH = globals().get(
    "BASELINE_SELECTED_FEATURES_PATH",
    "baseline_edge_stress_selected_features.json"
)
BASELINE_SCALER_PATH = globals().get(
    "BASELINE_SCALER_PATH",
    "baseline_edge_stress_scaler.pkl"
)

EDGE_ROOT_DIR = os.path.join(RESULT_DIR, "edge_deploy_models")
EDGE_MR2G_DIR = os.path.join(RESULT_DIR, "edge_deploy_mr2g")
EDGE_PGRG_DIR = os.path.join(RESULT_DIR, "edge_deploy_pgrg")
EDGE_PGRG_PLUS_ALIAS_DIR = os.path.join(RESULT_DIR, "edge_deploy_pgrg_plus")
os.makedirs(EDGE_ROOT_DIR, exist_ok=True)
os.makedirs(EDGE_MR2G_DIR, exist_ok=True)
os.makedirs(EDGE_PGRG_DIR, exist_ok=True)
os.makedirs(EDGE_PGRG_PLUS_ALIAS_DIR, exist_ok=True)


def _first_existing_path(candidates):
    """返回候选路径中第一个存在的路径。"""
    for p in candidates:
        if p is None:
            continue
        p = str(p)
        if os.path.exists(p):
            return p
    return None


def _get_edge_data_value(key):
    """从 notebook 的 edge_data 字典中安全读取值。"""
    data = globals().get("edge_data", None)
    if isinstance(data, dict) and key in data:
        return data[key]
    return None


def _copy_or_dump_preprocess(dst_dir):
    """复制或导出 selected_features.json 与 scaler.pkl。"""
    os.makedirs(dst_dir, exist_ok=True)

    selected_candidates = [
        globals().get("BASELINE_SELECTED_FEATURES_PATH", None),
        BASELINE_SELECTED_FEATURES_PATH,
        os.path.join(RESULT_DIR, "baseline_edge_stress_selected_features.json"),
        os.path.join(RESULT_DIR, "selected_features.json"),
        "baseline_edge_stress_selected_features.json",
        "selected_features.json",
    ]
    scaler_candidates = [
        globals().get("BASELINE_SCALER_PATH", None),
        BASELINE_SCALER_PATH,
        os.path.join(RESULT_DIR, "baseline_edge_stress_scaler.pkl"),
        os.path.join(RESULT_DIR, "scaler.pkl"),
        "baseline_edge_stress_scaler.pkl",
        "scaler.pkl",
    ]

    selected_src = _first_existing_path(selected_candidates)
    scaler_src = _first_existing_path(scaler_candidates)

    selected_dst = os.path.join(dst_dir, "selected_features.json")
    scaler_dst = os.path.join(dst_dir, "scaler.pkl")

    if selected_src is not None:
        shutil.copy2(selected_src, selected_dst)
    else:
        selected_features = _get_edge_data_value("selected_features")
        if selected_features is None:
            raise FileNotFoundError(
                "未找到特征顺序文件 selected_features.json，也没有在当前 notebook 内存中找到 edge_data['selected_features']。\n"
                "请先运行前面的数据预处理单元，或确认 baseline_edge_stress_selected_features.json 已存在。"
            )
        with open(selected_dst, "w", encoding="utf-8") as f:
            json.dump(list(selected_features), f, ensure_ascii=False, indent=2)

    if scaler_src is not None:
        shutil.copy2(scaler_src, scaler_dst)
    else:
        scaler = _get_edge_data_value("scaler")
        if scaler is None:
            raise FileNotFoundError(
                "未找到标准化器 scaler.pkl，也没有在当前 notebook 内存中找到 edge_data['scaler']。\n"
                "请先运行前面的数据预处理单元，或确认 baseline_edge_stress_scaler.pkl 已存在。"
            )
        with open(scaler_dst, "wb") as f:
            pickle.dump(scaler, f)

    with open(selected_dst, "r", encoding="utf-8") as f:
        selected_features = json.load(f)
    return int(len(selected_features))


def _copy_if_exists(src, dst, required=True, label="file"):
    if src is None or not os.path.exists(str(src)):
        msg = f"未找到 {label}：{src}"
        if required:
            raise FileNotFoundError(msg)
        print("[跳过]", msg)
        return False
    shutil.copy2(str(src), str(dst))
    return True


def _resolve_weights(model_key):
    """寻找权重路径；若内存中有模型对象但还未保存，则现场保存。"""
    if model_key == "mr2g":
        candidates = [
            globals().get("mr2g_path", None),
            os.path.join(MR2G_RESULT_DIR, "mr2g_final.weights.h5"),
            os.path.join(RESULT_DIR, "exp7_mr2g_single_module", "mr2g_final.weights.h5"),
            "mr2g_final.weights.h5",
        ]
        found = _first_existing_path(candidates)
        if found is not None:
            return found
        model_obj = globals().get("mr2g_model", None)
        if model_obj is not None:
            tmp_path = os.path.join(MR2G_RESULT_DIR, "mr2g_final.weights.h5")
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            model_obj.save_weights(tmp_path)
            return tmp_path
        return None

    if model_key == "pgrg":
        candidates = [
            globals().get("pgrg_path", None),
            os.path.join(PGRG_RESULT_DIR, "pgrg_final.weights.h5"),
            os.path.join(RESULT_DIR, "exp7_pgrg_single_module", "pgrg_final.weights.h5"),
            "pgrg_final.weights.h5",
        ]
        found = _first_existing_path(candidates)
        if found is not None:
            return found
        model_obj = globals().get("pgrg_model", None)
        if model_obj is not None:
            tmp_path = os.path.join(PGRG_RESULT_DIR, "pgrg_final.weights.h5")
            os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
            model_obj.save_weights(tmp_path)
            return tmp_path
        return None

    raise ValueError(f"未知模型：{model_key}")


def _resolve_or_write_pgrg_proto():
    """寻找或导出 PGRG+ 原型 CSV。"""
    candidates = [
        os.path.join(PGRG_RESULT_DIR, "pgrg_feature_prototypes.csv"),
        os.path.join(RESULT_DIR, "exp7_pgrg_single_module", "pgrg_feature_prototypes.csv"),
        "pgrg_feature_prototypes.csv",
    ]
    found = _first_existing_path(candidates)
    if found is not None:
        return found

    proto_df = globals().get("pgrg_proto_df", None)
    if proto_df is not None:
        tmp_path = os.path.join(PGRG_RESULT_DIR, "pgrg_feature_prototypes.csv")
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        proto_df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
        return tmp_path
    return None


def _safe_int(value, default):
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value, default):
    try:
        return float(value)
    except Exception:
        return float(default)


_export_report = []

# 1) 导出 MR²G
try:
    feature_dim = _copy_or_dump_preprocess(EDGE_MR2G_DIR)
    _mr2g_weights = _resolve_weights("mr2g")
    _copy_if_exists(
        _mr2g_weights,
        os.path.join(EDGE_MR2G_DIR, "mr2g_final.weights.h5"),
        required=True,
        label="MR²G 权重"
    )

    _mr2g_config = {
        "model_name": globals().get("MR2G_MODEL_NAME", "MR2G_SINGLE"),
        "feature_dim": feature_dim,
        "threshold": _safe_float(globals().get("MR2G_THRESHOLD", globals().get("DUAL_THRESHOLD", 0.50)), 0.50),
        "weights_file": "mr2g_final.weights.h5",
        "selected_features_file": "selected_features.json",
        "scaler_file": "scaler.pkl",
        "input_protocol": "raw_csv -> selected_features -> StandardScaler -> missing_mask -> MR²G([x_scaled, mask])",
        "task": "binary_iot_intrusion_detection",
        "label_mapping": {"Normal": 0, "Attack": 1},
        "bottleneck_dim": _safe_int(globals().get("MR2G_BOTTLENECK_DIM", 64), 64),
        "recon_lambda": _safe_float(globals().get("MR2G_RECON_LAMBDA", 0.20), 0.20),
        "consistency_lambda": _safe_float(globals().get("MR2G_CONS_LAMBDA", 0.25), 0.25),
        "sparse_lambda": _safe_float(globals().get("MR2G_SPARSE_LAMBDA", 0.01), 0.01),
        "learning_rate": _safe_float(globals().get("MR2G_LR", globals().get("DUAL_LR", 1e-3)), 1e-3),
        "note": "云端训练、本地端侧推理；端侧不执行训练。"
    }
    with open(os.path.join(EDGE_MR2G_DIR, "deploy_config.json"), "w", encoding="utf-8") as f:
        json.dump(_mr2g_config, f, ensure_ascii=False, indent=2)
    _export_report.append({"model": "MR²G", "status": "ok", "dir": EDGE_MR2G_DIR})
except Exception as e:
    _export_report.append({"model": "MR²G", "status": "failed", "error": str(e), "dir": EDGE_MR2G_DIR})
    print("[MR²G 导出失败]", e)
    print("提示：如果 MR²G 还没有训练，请把 RUN_MR2G_EXPERIMENT=True 后运行 MR²G 单元，再重新运行本导出单元。")

# 2) 导出 PGRG+
try:
    feature_dim = _copy_or_dump_preprocess(EDGE_PGRG_DIR)
    _pgrg_weights = _resolve_weights("pgrg")
    _copy_if_exists(
        _pgrg_weights,
        os.path.join(EDGE_PGRG_DIR, "pgrg_final.weights.h5"),
        required=True,
        label="PGRG/PGRG+ 权重"
    )

    _pgrg_proto = _resolve_or_write_pgrg_proto()
    _copy_if_exists(
        _pgrg_proto,
        os.path.join(EDGE_PGRG_DIR, "pgrg_feature_prototypes.csv"),
        required=True,
        label="PGRG/PGRG+ 原型文件"
    )

    _pgrg_config = {
        "model_name": globals().get("PGRG_MODEL_NAME", "PGRG_SINGLE"),
        "feature_dim": feature_dim,
        "threshold": _safe_float(globals().get("PGRG_THRESHOLD", globals().get("DUAL_THRESHOLD", 0.50)), 0.50),
        "weights_file": "pgrg_final.weights.h5",
        "prototype_file": "pgrg_feature_prototypes.csv",
        "selected_features_file": "selected_features.json",
        "scaler_file": "scaler.pkl",
        "input_protocol": "raw_csv -> selected_features -> StandardScaler -> missing_mask -> PGRG/PGRG+([x_scaled, mask])",
        "task": "binary_iot_intrusion_detection",
        "label_mapping": {"Normal": 0, "Attack": 1},
        "proto_lambda": _safe_float(globals().get("PGRG_PROTO_LAMBDA", 0.15), 0.15),
        "consistency_lambda": _safe_float(globals().get("PGRG_CONS_LAMBDA", 0.25), 0.25),
        "gate_lambda": _safe_float(globals().get("PGRG_GATE_LAMBDA", 0.01), 0.01),
        "margin": _safe_float(globals().get("PGRG_MARGIN", 0.50), 0.50),
        "gate_hidden_dim": _safe_int(globals().get("PGRG_GATE_HIDDEN_DIM", 256), 256),
        "proto_context_dim": _safe_int(globals().get("PGRG_PROTO_CONTEXT_DIM", 32), 32),
        "num_prototypes_per_class": _safe_int(globals().get("PGRG_NUM_PROTOTYPES_PER_CLASS", 3), 3),
        "enable_global_feature_gate": bool(globals().get("PGRG_ENABLE_GLOBAL_FEATURE_GATE", True)),
        "local_topk_ratio": _safe_float(globals().get("PGRG_LOCAL_TOPK_RATIO", 0.20), 0.20),
        "learning_rate": _safe_float(globals().get("PGRG_LR", globals().get("DUAL_LR", 1e-3)), 1e-3),
        "note": "云端训练、本地端侧推理；端侧不执行训练。"
    }
    with open(os.path.join(EDGE_PGRG_DIR, "deploy_config.json"), "w", encoding="utf-8") as f:
        json.dump(_pgrg_config, f, ensure_ascii=False, indent=2)
    # 同步一份旧目录别名，兼容之前的 app.py / edge_inference.py
    try:
        for name in ["pgrg_final.weights.h5", "pgrg_feature_prototypes.csv", "selected_features.json", "scaler.pkl", "deploy_config.json"]:
            src_path = os.path.join(EDGE_PGRG_DIR, name)
            if os.path.exists(src_path):
                shutil.copy2(src_path, os.path.join(EDGE_PGRG_PLUS_ALIAS_DIR, name))
    except Exception as alias_e:
        print("[提示] PGRG+ 兼容目录同步失败，不影响 edge_deploy_pgrg 主目录使用：", alias_e)
    _export_report.append({"model": globals().get("PGRG_MODEL_NAME", "PGRG"), "status": "ok", "dir": EDGE_PGRG_DIR})
except Exception as e:
    _export_report.append({"model": "PGRG/PGRG+", "status": "failed", "error": str(e), "dir": EDGE_PGRG_DIR})
    print("[PGRG+ 导出失败]", e)

# 3) 生成一个总索引文件，便于记录
with open(os.path.join(EDGE_ROOT_DIR, "deploy_index.json"), "w", encoding="utf-8") as f:
    json.dump(_export_report, f, ensure_ascii=False, indent=2)

print("\n端侧部署导出完成，报告如下：")
for item in _export_report:
    print(item)

print("\n整合部署包默认读取以下目录：")
print("  RESULT_DIR:", RESULT_DIR)
print("  MR²G :", EDGE_MR2G_DIR)
print("  PGRG/PGRG+:", EDGE_PGRG_DIR)
print("\n启动方式：")
print("  python edge_inference.py --model both --root-dir", RESULT_DIR, "--input your_edge_flow.csv --output predictions_mr2g_pgrg.csv")
print("  streamlit run app.py")
