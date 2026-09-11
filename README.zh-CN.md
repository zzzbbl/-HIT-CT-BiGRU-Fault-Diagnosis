# HIT CT-BiGRU Fault Diagnosis

[English](README.md) | [中文](README.zh-CN.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)

这是一个用于航发中介轴承振动信号故障诊断的
1DCNN-Transformer-BiGRU 代码仓库。

本仓库为已发表于 *Scientific Reports* 的相关论文提供配套实现：

> Wang, Y., Zhang, B. A fault diagnosis method for aero-engine inter-shaft
> bearings based on 1DCNN-Transformer-BiGRU. *Scientific Reports* (2026).
> https://doi.org/10.1038/s41598-026-61710-4

本项目提供数据集预处理、模型训练、评估、噪声鲁棒性测试、模块消融和特征可视化的可复现代码。仓库不包含原始数据集、处理后的张量数据、训练好的模型权重、实验结果表格或生成的图片。

## 项目特点

- CT-BiGRU 结构：结合大卷积核 1DCNN、Transformer 编码器和双向 GRU。
- HIT 全序列数据处理：按原始样本级别划分训练集、验证集和测试集。
- 提供基线模型和消融模型，便于对比不同模块的作用。
- 使用 YAML 配置文件管理训练和评估参数。
- 支持不同信噪比条件下的噪声鲁棒性评估。
- 提供基于 t-SNE 的特征可视化脚本。

## 目录结构

```text
configs/                  YAML 实验配置
data/                     数据放置与准备说明
docs/                     复现、数据政策、引用等文档
notebooks/                可选演示 notebook 目录
src/hit_ct_bigru/         可复用 Python 包
src/train.py              训练 CT-BiGRU 或基线模型
src/evaluate.py           评估已保存的 checkpoint
src/noise_robustness.py   测试噪声注入条件下的鲁棒性
src/ablation.py           运行模块消融实验
src/prepare_hit_dataset.py  准备 HIT 处理后数据划分
src/visualize_features.py   生成 t-SNE 特征可视化
```

## 安装

创建虚拟环境并安装依赖：

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

如果希望以可编辑模式安装本项目：

```bash
pip install -e .
```

如果使用 GPU 训练，请根据 CUDA 版本从 PyTorch 官方安装页面安装对应的 PyTorch 版本。

## 数据准备
HIT 航发中介轴承数据集可从以下地址获取：

- 数据集页面：<https://github.com/HouLeiHIT/HIT-dataset>

请遵守数据集提供方的许可协议和使用条款。本仓库不重新分发原始数据或处理后的张量文件。
请自行按照数据集提供方的要求准备原始数组文件，期望文件名如下：

```text
data1.npy
data2.npy
data3.npy
data4.npy
data5.npy
```

然后运行：

```bash
python src/prepare_hit_dataset.py --raw-dir <path-to-hit-raw-arrays> --out-dir data/hit_full
```

生成的数据目录应为：

```text
data/hit_full/
├── train_xdata
├── train_ylabel
├── val_xdata
├── val_ylabel
├── test_xdata
├── test_ylabel
└── metadata.json
```

更多说明见 [data/README.md](data/README.md) 和 [docs/data_policy.md](docs/data_policy.md)。

## 训练

训练默认 CT-BiGRU 模型：

```bash
python src/train.py --config configs/hit_ct_bigru.yaml
```

训练生成的 checkpoint、指标和图片会保存在 `outputs/`，该目录默认被 Git 忽略。

## 评估

评估训练好的模型：

```bash
python src/evaluate.py \
  --config configs/hit_ct_bigru.yaml \
  --checkpoint outputs/hit_ct_bigru/best_model.pth
```

## 噪声鲁棒性测试

在不同信噪比的高斯噪声注入条件下评估模型：

```bash
python src/noise_robustness.py \
  --config configs/hit_noise.yaml \
  --checkpoint outputs/hit_ct_bigru/best_model.pth
```

## 消融实验

运行模块消融实验：

```bash
python src/ablation.py --config configs/hit_ablation.yaml
```

该脚本会训练以下模型变体：

- `1dcnn_transformer`
- `1dcnn_bigru`
- `transformer_bigru`
- `ct_bigru`

## 默认模型配置

[configs/hit_ct_bigru.yaml](configs/hit_ct_bigru.yaml) 中的默认设置包括：

- 输入通道数：`4`
- 序列长度：`20480`
- 1DCNN 卷积核大小：`(32, 16)`
- 1DCNN 步长：`(8, 4)`
- Transformer 编码器层数：`2`
- 注意力头数：`8`
- 嵌入维度：`128`
- 前馈网络维度：`512`
- BiGRU 隐藏单元数：`128`
- 卷积前端 Dropout：`0.4`
- Batch size：`16`
- Epochs：`50`

## 复现说明

代码中设置了随机种子，并通过配置文件固定主要实验参数。但由于硬件、PyTorch 版本、CUDA/cuDNN 版本、浮点计算和数据预处理差异，具体数值结果可能有所波动。

推荐记录随机种子、PyTorch 版本、硬件、数据划分方式和配置文件。更多建议见 [docs/reproducibility.md](docs/reproducibility.md)。

## 引用

如果本仓库对你的研究有帮助，请引用以下论文：

```bibtex
@article{wang2026fault,
  title = {A fault diagnosis method for aero-engine inter-shaft bearings based on 1DCNN-Transformer-BiGRU},
  author = {Wang, Yang and Zhang, Boliang},
  journal = {Scientific Reports},
  year = {2026},
  doi = {10.1038/s41598-026-61710-4},
  url = {https://doi.org/10.1038/s41598-026-61710-4}
}
```

机器可读引用信息见 [CITATION.cff](CITATION.cff)。

## 许可证

本项目采用 [MIT License](LICENSE) 开源。该许可证适用于本仓库中的源代码和文档。外部数据集、第三方论文以及用户自行生成的实验输出遵循其各自的许可和使用条款。
