"""
模型测试脚本 (Testing/Evaluation Script)
用于评估已训练的时空图神经网络模型（ST-GAT-LSTM 或 ASTGCN）

功能特性：
    1. 加载训练好的模型checkpoint
    2. 在测试集上进行完整评估
    3. 分时段评估：短期/中期/长期预测精度
    4. 保存预测结果和评估指标
    5. 可视化预测结果（可选）

使用方法：
    # 评估最佳模型（自动从checkpoint推断模型类型）
    python test.py --config configs/train_config.yaml --checkpoint runs/checkpoints/best_model.pt

    # 指定模型类型并保存预测结果
    python test.py --config configs/train_config.yaml --checkpoint runs/checkpoints/best_model.pt \
      --model_type astgcn --save_predictions

作者：重构版本
日期：2025-10
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Optional

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

# 导入自定义模块
from models.gat_lstm_model import create_model_from_config
from models.astgcn_model import create_astgcn_from_config
from models.gwnet_model import create_gwnet_from_config
from lib.data_loader import create_dataloaders, prepare_adjacency_matrix, StandardScaler
from lib.metrics import MetricsCalculator, print_evaluation_results
from lib.trainer_utils import setup_logging


@torch.no_grad()
def test_model(
    model: nn.Module,
    dataloader: DataLoader,
    adj: torch.Tensor,
    scaler: StandardScaler,
    device: torch.device,
    null_val: float = 0.0,
    save_predictions: bool = False,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    测试模型并计算评估指标

    参数：
        model: 训练好的模型
        dataloader: 测试数据加载器
        adj: 邻接矩阵
        scaler: 数据标准化器
        device: 设备
        null_val: 缺失值标记
        save_predictions: 是否保存预测结果
        output_dir: 输出目录

    返回：
        评估结果字典
    """
    model.eval()

    # 用于累积所有预测结果
    all_predictions = []
    all_targets = []

    # 指标计算器
    metrics_calc = MetricsCalculator(null_val=null_val)

    logging.info("开始测试...")
    logging.info(f"测试批次数: {len(dataloader)}")

    for batch_idx, batch in enumerate(dataloader):
        x = batch['x'].to(device)  # (B, T, N, F)
        y = batch['y'].to(device)  # (B, H, N, 1)

        # 前向传播
        y_pred = model(x, adj)  # (B, H, N, 1)

        # 反标准化到真实尺度
        y_true_np = scaler.inverse_transform(y.cpu().numpy())
        y_pred_np = scaler.inverse_transform(y_pred.cpu().numpy())

        # 更新指标计算器
        metrics_calc.update(
            torch.from_numpy(y_pred_np),
            torch.from_numpy(y_true_np)
        )

        # 如果需要保存预测，累积结果
        if save_predictions:
            all_predictions.append(y_pred_np)
            all_targets.append(y_true_np)

        # 打印进度
        if (batch_idx + 1) % 10 == 0:
            logging.info(f"  已处理 {batch_idx + 1}/{len(dataloader)} 批次")

    # 计算评估指标
    logging.info("\n计算评估指标...")
    results = metrics_calc.compute(horizons=[3, 6, 12])

    # 保存预测结果
    if save_predictions and output_dir:
        logging.info("\n保存预测结果...")
        output_dir.mkdir(parents=True, exist_ok=True)

        predictions = np.concatenate(all_predictions, axis=0)  # (N, H, num_nodes, 1)
        targets = np.concatenate(all_targets, axis=0)

        pred_path = output_dir / 'predictions.npz'
        np.savez(
            pred_path,
            predictions=predictions,
            targets=targets,
        )
        logging.info(f"✓ 预测结果已保存: {pred_path}")
        logging.info(f"  - 预测形状: {predictions.shape}")
        logging.info(f"  - 目标形状: {targets.shape}")

    return results


def visualize_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_dir: Path,
    num_samples: int = 5,
    num_nodes: int = 3,
):
    """
    可视化预测结果（需要matplotlib）

    参数：
        predictions: 预测值 (N, H, num_nodes, 1)
        targets: 真实值
        output_dir: 输出目录
        num_samples: 可视化的样本数
        num_nodes: 可视化的节点数
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("未安装matplotlib，跳过可视化")
        return

    logging.info("\n生成可视化...")

    N, H, total_nodes, _ = predictions.shape

    # 随机选择样本和节点
    sample_indices = np.random.choice(N, size=min(num_samples, N), replace=False)
    node_indices = np.random.choice(total_nodes, size=min(num_nodes, total_nodes), replace=False)

    fig, axes = plt.subplots(num_samples, num_nodes, figsize=(15, 3 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    if num_nodes == 1:
        axes = axes.reshape(-1, 1)

    for i, sample_idx in enumerate(sample_indices):
        for j, node_idx in enumerate(node_indices):
            ax = axes[i, j]

            pred = predictions[sample_idx, :, node_idx, 0]
            true = targets[sample_idx, :, node_idx, 0]

            ax.plot(true, 'o-', label='真实值', linewidth=2)
            ax.plot(pred, 's--', label='预测值', linewidth=2)
            ax.set_xlabel('时间步')
            ax.set_ylabel('流量')
            ax.set_title(f'样本{sample_idx}, 节点{node_idx}')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = output_dir / 'predictions_visualization.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"✓ 可视化已保存: {fig_path}")


def main(
    config_path: str,
    checkpoint_path: str,
    save_predictions: bool = False,
    model_type: str = None,
):
    """
    主测试函数

    参数：
        config_path: 配置文件路径
        checkpoint_path: 模型checkpoint路径
        save_predictions: 是否保存预测结果
        model_type: 模型类型（gat_lstm / astgcn），优先使用命令行，
                    若未指定则尝试从checkpoint和配置中推断
    """
    # ==================== 加载配置 ====================
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    data_cfg = config['data']
    model_cfg = config['model']
    train_cfg = config['train']

    # ==================== 设置日志 ====================
    output_dir = Path('runs/test_results')
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(log_dir=str(output_dir), log_file='test.log', level=logging.INFO)

    logging.info("=" * 70)
    logging.info("模型测试与评估")
    logging.info("=" * 70)
    logging.info(f"配置文件: {config_path}")
    logging.info(f"Checkpoint: {checkpoint_path}")
    logging.info(f"输出目录: {output_dir}")

    # ==================== 设置设备 ====================
    device_name = train_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_name)
    logging.info(f"使用设备: {device}")

    # ==================== 加载数据 ====================
    logging.info("\n" + "-" * 70)
    logging.info("加载数据...")
    logging.info("-" * 70)

    dataset_dir = data_cfg['dataset_dir']
    loaders, scaler = create_dataloaders(
        dataset_dir=dataset_dir,
        batch_size=data_cfg.get('batch_size', 64),
        num_workers=data_cfg.get('num_workers', 0),
        pin_memory=False,  # 测试时不需要pin_memory
        shuffle_train=False,
        target_channel=data_cfg.get('target_channel', 0),
        return_all_channels=data_cfg.get('return_all_channels', False),
    )

    logging.info(f"数据标准化器: {scaler}")

    # 加载邻接矩阵
    graph_pkl = data_cfg['graph_pkl_filename']
    adj = prepare_adjacency_matrix(
        pkl_path=graph_pkl,
        device=device,
        add_self_loops=model_cfg.get('add_self_loops', True),
        threshold=model_cfg.get('adj_threshold', 0.0),
    )

    # 加载checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint文件不存在: {checkpoint_path}")

    logging.info(f"正在加载checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 从checkpoint配置中解析模型类型（如果命令行未指定）
    if model_type is None:
        ckpt_cfg = checkpoint.get('config', {})
        ckpt_model_cfg = ckpt_cfg.get('model', {}) if isinstance(ckpt_cfg, dict) else {}
        ckpt_model_type = ckpt_model_cfg.get('model_type')

        if ckpt_model_type:
            model_type = str(ckpt_model_type)
        else:
            # 回退到配置文件中的 model_type 或默认值
            model_type = model_cfg.get('model_type', 'gat_lstm')

    model_type = model_type.lower()

    # ==================== 创建模型 ====================
    logging.info("\n" + "-" * 70)
    logging.info("加载模型...")
    logging.info("-" * 70)

    # 将选择的模型类型写回配置，保证与命令行/Checkpoint一致
    model_cfg['model_type'] = model_type

    if model_type == 'gat_lstm':
        model = create_model_from_config(config)
    elif model_type == 'astgcn':
        model = create_astgcn_from_config(config)
    elif model_type == 'gwnet':
        model = create_gwnet_from_config(config)
    else:
        raise ValueError(f"未知的模型类型: {model_type}，可选：gat_lstm / astgcn / gwnet")

    logging.info(f"模型类型: {model_type}")
    model = model.to(device)

    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info("✓ 模型权重已加载")

    # 打印checkpoint信息
    logging.info(f"Checkpoint信息:")
    logging.info(f"  - 训练轮次: {checkpoint.get('epoch', 'N/A')}")
    logging.info(f"  - 验证损失: {checkpoint.get('val_loss', 'N/A')}")
    logging.info(f"  - 保存时间: {checkpoint.get('timestamp', 'N/A')}")

    # 从checkpoint恢复scaler（如果有）
    if 'scaler' in checkpoint:
        scaler = StandardScaler.from_dict(checkpoint['scaler'])
        logging.info(f"已从checkpoint恢复标准化器: {scaler}")

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"模型总参数量: {total_params:,}")

    # ==================== 测试模型 ====================
    logging.info("\n" + "=" * 70)
    logging.info("在测试集上评估模型")
    logging.info("=" * 70)

    null_val = data_cfg.get('null_val', 0.0)

    # 执行测试
    test_results = test_model(
        model=model,
        dataloader=loaders['test'],
        adj=adj,
        scaler=scaler,
        device=device,
        null_val=null_val,
        save_predictions=save_predictions,
        output_dir=output_dir if save_predictions else None,
    )

    # ==================== 打印和保存结果 ====================
    # 打印详细结果
    print_evaluation_results(test_results, title="测试集评估结果")

    # 保存结果到JSON
    results_path = output_dir / 'test_metrics.json'

    # 转换NumPy类型为Python原生类型
    def convert_to_serializable(obj):
        """递归转换对象为JSON可序列化类型"""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    # 转换test_results
    serializable_results = convert_to_serializable(test_results)

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    logging.info(f"✓ 评估指标已保存: {results_path}")

    # ==================== 可视化（如果保存了预测） ====================
    if save_predictions:
        pred_file = output_dir / 'predictions.npz'
        if pred_file.exists():
            data = np.load(pred_file)
            predictions = data['predictions']
            targets = data['targets']

            visualize_predictions(
                predictions=predictions,
                targets=targets,
                output_dir=output_dir,
                num_samples=5,
                num_nodes=3,
            )

    # ==================== 完成 ====================
    logging.info("\n" + "=" * 70)
    logging.info("测试完成！")
    logging.info("=" * 70)

    # 打印总体指标摘要
    overall = test_results.get('overall', {})
    logging.info(f"\n总体指标:")
    logging.info(f"  MAE:  {overall.get('MAE', 0):.4f}")
    logging.info(f"  RMSE: {overall.get('RMSE', 0):.4f}")
    logging.info(f"  MAPE: {overall.get('MAPE', 0):.2f}%")

    logging.info(f"\n所有结果已保存到: {output_dir}")
    logging.info("✓ 测试流程完成！")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='测试时空图神经网络模型')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='配置文件路径 (YAML格式)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='模型checkpoint路径'
    )
    parser.add_argument(
        '--save_predictions',
        action='store_true',
        help='是否保存预测结果'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default=None,
        choices=['gat_lstm', 'astgcn', 'gwnet'],
        help='选择模型类型：gat_lstm（默认）/ astgcn / gwnet'
    )

    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.config):
        print(f"错误：配置文件不存在: {args.config}")
        sys.exit(1)

    if not os.path.exists(args.checkpoint):
        print(f"错误：Checkpoint文件不存在: {args.checkpoint}")
        sys.exit(1)

    # 开始测试
    main(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        save_predictions=args.save_predictions,
        model_type=args.model_type,
    )
