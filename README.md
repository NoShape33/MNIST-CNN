# MNIST 手写数字识别 — 基于卷积神经网络

这是一个电子信息菜鸡的人工智能结课论文项目，基于PyTorch做了一个识别手写数字的项目

项目代码用 PyTorch 实现，包含一个基础 CNN 和一个 Mini-ResNet 作为对照实验，所有训练日志、图表均可在本地完整复现。

## 项目结构

```
├── src/
│   ├── model.py          # CNN 模型定义（BasicCNN + MiniResNet）
│   ├── train.py          # 训练主脚本（三组对照实验）
│   ├── data_loader.py    # 数据加载与预处理
│   ├── evaluate.py       # 评估与可视化图表生成
│   ├── utils.py          # 辅助函数（计时、绘图等）
│   └── test_omml.py      # OMML 公式验证脚本
├── figures/              # 所有论文用图（运行 evaluate.py 后生成）
├── outputs/              # 训练日志 JSON 文件
├── paper/                # 课程论文 Word 文档
├── requirements.txt      # Python 依赖清单
├── .gitignore
└── README.md
```

## 环境要求

- Python 3.10+
- PyTorch 2.0+（CUDA 可选，CPU 也能跑）

## 安装与运行

```bash
# 1. 克隆仓库
git clone https://github.com/NoShape33/-.git
cd -

# 2. 安装依赖
pip install -r requirements.txt

# 3. 训练模型（三组对照实验，约 10-15 分钟）
python src/train.py

# 4. 生成评估图表
python src/evaluate.py
```

训练完成后：
- 模型权重与训练日志保存在 `outputs/` 目录
- 所有论文用图（loss 曲线、混淆矩阵、特征图等）生成到 `figures/` 目录

MNIST 数据集会在首次运行时自动下载到 `data/` 目录。

## 实验设置

| 参数 | 值 |
|------|-----|
| Epochs | 20（早停 patience=5） |
| Batch Size | 64 |
| 优化器 | Adam (lr=0.001) / SGD (lr=0.01, momentum=0.9) |
| 学习率调度 | ReduceLROnPlateau (factor=0.5, patience=3) |
| 数据增强 | 随机旋转 ±10° |

## 对照实验

| 实验组 | 模型 | 优化器 | 测试准确率 |
|--------|------|--------|-----------|
| 1 | BasicCNN | Adam | 99.47% |
| 2 | BasicCNN | SGD | 99.44% |
| 3 | MiniResNet | Adam | 99.52% |

## 模型架构

### BasicCNN

```
Conv2d(1, 32, 3) → BN → ReLU → MaxPool(2)
Conv2d(32, 64, 3) → BN → ReLU → MaxPool(2)
Conv2d(64, 128, 3) → BN → ReLU → MaxPool(2)
FC(1152, 256) → ReLU → Dropout(0.5)
FC(256, 10) → LogSoftmax
```

### MiniResNet（对照实验）

```
Conv2d(1, 32, 3) → BN → ReLU
ResidualBlock(32→32) ×2
ResidualBlock(32→64, stride=2) ×2
GlobalAvgPool → FC(64, 10) → LogSoftmax
```
