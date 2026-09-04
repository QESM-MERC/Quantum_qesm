# QESM

<p align="right">
  <a href="README.md">English</a> | <strong>简体中文</strong>
</p>

**面向多模态对话情感识别的动态融合与纠缠测量量子启发网络**

[![任务](https://img.shields.io/badge/task-multimodal%20emotion%20recognition-5c6ac4)](https://github.com/QESM-MERC/Quantum_qesm)
[![框架](https://img.shields.io/badge/framework-PyTorch-ee4c2c)](https://pytorch.org/)
[![测试](https://github.com/QESM-MERC/Quantum_qesm/actions/workflows/ci.yml/badge.svg)](https://github.com/QESM-MERC/Quantum_qesm/actions/workflows/ci.yml)

> [!IMPORTANT]
> 当前先公开项目说明。源代码、冻结实验配置、原始特征检查点、论文元数据和开源许可证
> 正在整理，将在后续版本中发布。

QESM 是受量子数学结构启发、运行于经典硬件上的 PyTorch 模型，不需要量子计算机。它将文本、音频和视觉话语特征映射为复值状态，沿对话上下文演化这些状态，建模跨模态扰动，并进行联合情感测量。

![QESM 架构](overview.png)

## 模型结构

实验采用的处理顺序为：

```text
量子态制备 -> UTE_1 -> OCS -> UTE_2 -> 密度混合融合 -> EBM
```

- **幺正时间演化（UTE）**通过保持范数的相位演化传播说话人感知上下文。
- **OTOC 跨模态扰乱（OCS）**通过受非时序关联函数启发的交互建模动态跨模态互补性。
- **纠缠 Born 测量（EBM）**结合跨模态相位相干性与三模态类别一致性完成情感分类。

## 实验结果

QESM 在 IEMOCAP 和 MELD 上采用加权 F1（WF1）评估。

| 数据集 | 话语数 | 类别数 | 代表性 WF1 | 五次运行 WF1 |
|---|---:|---:|---:|---:|
| IEMOCAP | 7,433 | 6 | **73.23** | **72.82 ± 0.23** |
| MELD | 13,708 | 7 | **67.31** | **67.17 ± 0.11** |

五次运行统计使用随机种子 `{0, 1, 2, 3, 6}`，误差项为样本标准差。正式发布前将补充精确配置快照。

## 仓库结构

```text
qotoc/          QESM 复值核心模块与模型
configs/        纳入版本控制的训练 CLI JSON 示例
dataset.py      IEMOCAP/MELD 特征加载与数据划分
train.py        训练和评估入口
evaluate.py     CFN-ESA 原始特征检查点推理入口
tests/          数值正确性与模型集成测试
```

内部 Python 包保留历史名称 `qotoc`。检查点仍须与保存时的架构、预处理协议和特征维度
一致；推理入口会严格验证这些条件。

## 环境安装

建议使用 Python 3.10 或更高版本。

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 数据准备

默认路径直接读取 CFN-ESA 提供的单个合并特征文件，不需要转换后的 M3Net 文件或替换特征：

```text
data/cfn_esa/
├── iemocap_multimodal_features.pkl
└── meld_multimodal_features.pkl
```

加载器支持 CFN-ESA 的 IEMOCAP 12 字段格式，以及 MELD 13/14 字段格式。模型使用四层
1024 维 RoBERTa 特征、原始音频特征（IEMOCAP 为 1582 维，MELD 为 300 维）和原始
342 维视觉特征。该模式会拒绝任何外部文本、音频或视觉替换特征。本仓库不重新分发
数据集或缓存特征。Python pickle 文件在加载时可能执行代码，请只使用可信来源提供的文件。

## 训练

```bash
python train.py --dataset iemocap --cfn-pkl data/cfn_esa/iemocap_multimodal_features.pkl --epochs 80
python train.py --dataset meld --cfn-pkl data/cfn_esa/meld_multimodal_features.pkl --epochs 40 --batch-size 32
python train.py --config configs/iemocap_example.json --seed 1
```

命令行中显式给出的选项会覆盖 `--config` 中的值。仓库内 JSON 文件目前是可运行示例，并非结果表所用配置的最终冻结版本。

代码也支持 IEMOCAP 按会话的 LOSO 评估，例如：

```bash
python train.py --dataset iemocap --data-dir data/m3net \
  --loso-test-session 5 --loso-internal-valid-frac 0.1
```

运行输出、检查点、报告和指标写入 `results/<run-name>/`。

在 CFN-ESA 原始特征模式下，`--merge-valid` 会使用合并 pickle 中的完整训练池。

## 检查点推理

对于使用 CFN-ESA 原始特征维度训练的检查点，只需合并 `.pkl`、检查点和训练配置
（训练 JSON 或 `metrics.json`）即可评估。仓库内的 `*_cfnesa_raw.json` 保存了两套已经
验证过的原始特征检查点架构：

```bash
python evaluate.py \
  --checkpoint results/iemocap_raw/best_test.pt \
  --config configs/iemocap_cfnesa_raw.json \
  --cfn-pkl data/cfn_esa/iemocap_multimodal_features.pkl
```

推理程序会严格加载权重，并在前向计算前核对每个模态的输入维度。使用增强或替换特征
训练的检查点会被明确拒绝，不会通过补零、截断或静默投影来强行适配。程序输出测试
话语数、加权 F1、宏 F1 和准确率。

## 测试

```bash
python -m pytest -q
```

测试覆盖 OTOC 闭式核、幺正相位旋转、复值状态归一化、Born 概率、填充不变性、反向
梯度有限性、CFN-ESA 两类数据格式、原始特征约束、检查点兼容性和端到端推理。

## 引用

论文目前处于审稿阶段，正式发表信息和 BibTeX 将在可用后补充。在此之前，请引用仓库链接：

```text
https://github.com/QESM-MERC/Quantum_qesm
```

## 许可证

许可证尚待确认。在加入 `LICENSE` 文件前，所有权利仍归作者所有，本仓库尚不构成开源发布。
