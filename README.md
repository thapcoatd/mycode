# ST-GAT-LSTM / ASTGCN 交通流量预测模型

端到端的时空图神经网络预测框架，当前提供 **GAT-LSTM**（串行空间→时间）、**ASTGCN**（时空注意力 + Chebyshev 图卷积）和 **Graph WaveNet**（扩散图卷积 + 空洞时间卷积 + 自适应邻接）三套可切换模型。支持完整的训练、评估、可视化、Checkpoint 管理与断点续训，附带中文文档与可视化脚本。

## 功能亮点
- ✅ 双模型可选：`gat_lstm`（默认）与 `astgcn`，命令行一键切换
- ✅ 完整训练/测试脚本：早停、学习率调度、梯度裁剪、断点续训、Checkpoint 清理
- ✅ 评估体系：分时段指标（3/6/12 步），支持保存预测与可视化
- ✅ 可视化：TensorBoard 实时监控 + Matplotlib 自动生成 loss/指标/LR/仪表盘
- ✅ 数据管线：Z-score 标准化、三分数据加载器、邻接矩阵预处理/可阈值/自环
- ✅ 文档完善：使用指南、可视化教程、重构说明与 FAQ

## 目录结构
```
DCRNN/
├── train.py                  # 训练脚本（支持 --model_type）
├── test.py                   # 测试/评估脚本（支持 --model_type）
├── models/
│   ├── gat_lstm_model.py     # GAT-LSTM 模型
│   └── astgcn_model.py       # ASTGCN 模型（新增）
├── lib/
│   ├── data_loader.py        # 数据加载/标准化/邻接处理
│   ├── metrics.py            # 训练与评估指标
│   ├── trainer_utils.py      # Checkpoint、早停、调度、日志等
│   └── visualization.py      # TensorBoard & Matplotlib 可视化
├── configs/train_config.yaml # 配置文件（可添加 model_type）
├── scripts/                  # 数据/基线/邻接生成脚本
├── runs/                     # 日志、Checkpoint、测试结果、图表
└── data/                     # METR-LA 样例数据与邻接矩阵
```

## 快速开始
### 1) 安装依赖
```bash
pip install torch numpy scipy pandas pyyaml tables tensorboard matplotlib
```

### 2) 训练
```bash
# 默认 GAT-LSTM
python train.py --config configs/train_config.yaml
# 使用 ASTGCN
python train.py --config configs/train_config.yaml --model_type astgcn
# 断点续训
python train.py --config configs/train_config.yaml --resume runs/checkpoints/last_model.pt --model_type astgcn

# 使用 Graph WaveNet
python train.py --config configs/train_config.yaml --model_type gwnet

# 同一模型多次实验，使用 exp_name 区分日志目录
python train.py --config configs/train_config.yaml --model_type gat_lstm --exp_name exp1
python train.py --config configs/train_config.yaml --model_type gat_lstm --exp_name exp2
```

### 3) 实时可视化
```bash
# GAT-LSTM 默认实验（未指定 exp_name）
tensorboard --logdir=runs/logs/gat_lstm/tensorboard

# ASTGCN 默认实验
tensorboard --logdir=runs/logs/astgcn/tensorboard

# Graph WaveNet 默认实验
tensorboard --logdir=runs/logs/gwnet/tensorboard

# 指定了 exp_name 的实验，例如：
#   --model_type gat_lstm --exp_name exp1
tensorboard --logdir=runs/logs/gat_lstm/exp1/tensorboard
#   --model_type gat_lstm --exp_name exp2
tensorboard --logdir=runs/logs/gat_lstm/exp2/tensorboard
#   --model_type gwnet --exp_name exp1
tensorboard --logdir=runs/logs/gwnet/exp1/tensorboard

# 浏览器访问 http://localhost:6006
```

### 4) 测试/评估
```bash
# 基础评估
python test.py --config configs/train_config.yaml --checkpoint runs/checkpoints/best_model.pt
# 评估 ASTGCN 并保存预测
python test.py --config configs/train_config.yaml --checkpoint runs/checkpoints/best_model.pt \
  --model_type astgcn --save_predictions
```

## 配置与模型
在 `configs/train_config.yaml` 的 `model` 段可指定模型相关参数；命令行的 `--model_type` 优先级最高。
```yaml
model:
  model_type: gat_lstm   # 可选 gat_lstm / astgcn（命令行覆盖）
  num_nodes: 207
  input_dim: 2
  output_horizon: 12
  # GAT-LSTM 相关
  gat_hidden_dim: 32
  gat_num_heads: 4
  gat_num_layers: 2
  gat_dropout_feat: 0.1
  gat_dropout_attn: 0.1
  gat_activation: elu
  gat_use_residual: true
  lstm_hidden_dim: 64
  lstm_num_layers: 1
  lstm_dropout: 0.1
  # ASTGCN 相关（仅 astgcn 使用）
  astgcn_blocks: 2
  astgcn_cheb_k: 3
  astgcn_channels: 32
  astgcn_heads: 2
  astgcn_dropout: 0.1
```

## 模型概览
- **GAT-LSTM（串行空间→时间）**  
  - 每个时间步共享多层 GAT 提取空间特征 → reshape 后送 LSTM 编码 → MLP 一次性解码多步预测。  
  - 优点：结构简洁，参数可控，训练稳定。

- **ASTGCN（时空注意力 + Chebyshev GCN）**  
  - 时间注意力（多头）→ 空间自适应注意力 → Chebyshev 图卷积（可混合静态邻接）→ 时间卷积 + 残差。  
  - 优点：显式学习时空权重，Cheb 多项式支持高阶邻域。

## 训练流程要点
- 数据：`lib/data_loader.create_dataloaders` 自动计算训练集均值/方差并标准化；`prepare_adjacency_matrix` 支持阈值与自环。
- 损失：训练用 masked MAE 主损失，日志附带 RMSE；评估用 `MetricsCalculator` 反标准化后计算 MAE/RMSE/MAPE（含 3/6/12 步）。
- 优化与调度：支持 Adam/AdamW；Step/MultiStep/Cosine/Plateau 调度；梯度裁剪；早停。
- Checkpoint：最佳/最新/按 epoch 保存，保留最近 N 个；包含 scaler、优化器、调度器状态，便于续训。

## 可视化
- **TensorBoard（推荐）**：`runs/logs/tensorboard`，记录 Loss、指标、学习率、Train/Val 对比。
- **Matplotlib 自动图表**：训练结束生成到 `runs/logs/figures/`  
  - `loss_curves.png`、`metrics_curves.png`、`learning_rate.png`、`training_dashboard.png`
- 手动重绘：`python generate_figures.py` 或从 `lib.visualization.TrainingVisualizer` 调用。

## 输出与文件
- 日志：`runs/logs/train.log`
- 训练历史：`runs/logs/training_history.json`
- Checkpoints：`runs/checkpoints/best_model.pt`、`last_model.pt`、`checkpoint_epoch_xxxx.pt`
- 测试结果：`runs/test_results/test_metrics.json`（可选 `predictions.npz` + 可视化）

## 常用命令速查
```bash
# 训练 & TensorBoard
python train.py --config configs/train_config.yaml --model_type gat_lstm
tensorboard --logdir=runs/logs/tensorboard

# 评估并保存预测
python test.py --config configs/train_config.yaml --checkpoint runs/checkpoints/best_model.pt \
  --model_type astgcn --save_predictions

# 模块自测
python models/gat_lstm_model.py
python models/astgcn_model.py
python lib/data_loader.py
python lib/metrics.py
python lib/visualization.py
```

## 常见问题（FAQ）
- **CUDA OOM**：调小 `data.batch_size`，或降低 `gat_hidden_dim`/`astgcn_channels`。
- **训练不收敛**：降低 `train.base_lr`，增加 `train.grad_clip`，检查数据标准化。
- **过拟合**：增加 dropout/weight_decay，或减少模型层数/通道。
- **Windows 加速 DataLoader**：`data.num_workers` 设为 0；GPU 训练可保留 `pin_memory: true`。

## 重构要点（摘自 REFACTORING_SUMMARY）
- 新增完整训练/测试脚本、Checkpoint 管理、早停、调度、训练历史保存。
- 重构数据加载/标准化、评估指标（Torch + NumPy 双实现）、可视化（TensorBoard + Matplotlib）。
- 模型模块化：GAT-LSTM + ASTGCN，可通过配置/命令行切换。
- 文档与注释全面，命令和配置默认值合理，可直接复现。

## 许可证
MIT License
