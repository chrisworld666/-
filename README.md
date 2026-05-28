# MR²G + PGRG/PGRG+ 端侧推理与可视化整合包

本包把你的两个单模块候选模型整合到同一套端侧部署原型中：

- **MR²G**：掩码引导的可靠性残差重构门控模块
- **PGRG+**：原型引导可靠性门控模块

适合毕业设计中写成：**“面向端侧部署的 MR²G 与 PGRG+ 本地推理及可视化原型系统”**。

## 1. 先在 notebook 里导出两个模型 artifact

我已经提供了 `export_both_deploy_artifacts_cell.py`，并把同样代码加入修改版 notebook 末尾。

运行后默认生成：

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

如果 MR²G 没有导出成功，通常是因为 notebook 里 `RUN_MR2G_EXPERIMENT=False`，需要改成 `True` 并运行 MR²G 训练单元后再导出。

## 2. 安装环境

建议 Python 3.9、3.10 或 3.11。

```bash
pip install -r requirements.txt
```

## 3. 命令行端侧推理

同时运行两个模型：

```bash
python edge_inference.py \
  --model both \
  --root-dir results_baseline_edge_stress_tests \
  --input your_edge_flow.csv \
  --output predictions_mr2g_pgrg.csv
```

只运行 MR²G：

```bash
python edge_inference.py --model mr2g --input your_edge_flow.csv --output predictions_mr2g.csv
```

只运行 PGRG+：

```bash
python edge_inference.py --model pgrg --input your_edge_flow.csv --output predictions_pgrg.csv
```

可视化模拟扰动：

```bash
python edge_inference.py \
  --model both \
  --input your_edge_flow.csv \
  --missing-rate 0.3 \
  --noise-std 0.2 \
  --output predictions_corrupted.csv
```

## 4. 启动可视化界面

```bash
streamlit run app.py
```

界面功能：

1. 上传 CSV 流量数据；
2. 选择运行 MR²G、PGRG+ 或两个都跑；
3. 展示 Normal / Attack 预测结果；
4. 展示攻击概率、可靠性门控、推理耗时；
5. 对 MR²G 展示重构变化幅度；
6. 对 PGRG+ 展示 Normal / Attack 原型距离；
7. 支持噪声强度和特征缺失率滑块，用于端侧鲁棒性可视化演示；
8. 展示 MR²G / PGRG+ / 四模块软投票的论文结果对比。

## 5. 论文建议写法

推荐写：

> 本文进一步设计了面向端侧部署的 MR²G 与 PGRG+ 可视化原型系统。系统采用“云端训练、本地推理、可视化展示”的方式，将训练后的模型权重、标准化参数、特征顺序和分类阈值导出到本地端侧环境，并通过统一推理脚本和 Streamlit 界面完成样本检测、攻击概率展示、可靠性门控分析和推理耗时统计。该系统并不等同于完整工业设备上线部署，而是作为端侧迁移前的轻量验证环节，为后续迁移到树莓派、Jetson 或工业边缘网关提供基础。

不要写成：

- 已完成真实工业端部署上线；
- 已在物联网设备中正式运行；
- 已完成边缘网关生产环境部署。

更稳妥的表述是：

- 端侧迁移可行性验证；
- 本地轻量推理原型；
- 面向端侧部署的可视化原型系统。

## 6. 常见报错：`NameError: RESULT_DIR is not defined`

如果你把导出单元作为 notebook 的第一个单元运行，旧版代码会因为前面定义 `RESULT_DIR` 的单元尚未执行而报错。新版导出单元已经加入路径兜底：

```python
RESULT_DIR = globals().get("RESULT_DIR", "results_baseline_edge_stress_tests")
```

因此可以单独运行。若后续提示找不到 `selected_features.json`、`scaler.pkl` 或模型权重，则说明还没有运行数据预处理或对应模型训练单元，需要先运行前面的预处理、PGRG+ 训练，以及可选的 MR²G 训练。


## TensorFlow 安装报错时怎么办

如果出现 `Could not find a version that satisfies the requirement tensorflow`，通常是 Python 版本或设备架构不兼容。建议：

1. 在电脑端或模拟端侧环境使用 Python 3.10 或 3.11；
2. 使用 `pip install -r requirements_pc.txt`，其中依赖为 `tensorflow-cpu`，不是完整 GPU 版 TensorFlow；
3. 如果真实边缘设备无法安装 TensorFlow，则论文中可以说明当前完成的是“本地端侧推理原型”，后续再将模型转换为 TensorFlow Lite。

本包的 `edge_inference.py` 和 `app.py` 默认加载 Keras 权重，因此仍需要 TensorFlow/Keras 运行时。
