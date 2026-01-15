"""
模型训练脚本 (Training Script)
用于训练时空图神经网络交通流量预测模型（支持 ST-GAT-LSTM 与 ASTGCN）

功能特性：
    1. 完整的训练流程：训练、验证、早停
    2. Checkpoint管理：自动保存最佳模型和最新模型
    3. 训练监控：实时输出训练指标，保存训练历史
    4. 断点续训：支持从checkpoint恢复训练
    5. 灵活配置：通过YAML配置文件和命令行参数管理超参数与模型类型

使用方法：
    python train.py --config configs/train_config.yaml

    # 从checkpoint恢复训练
    python train.py --config configs/train_config.yaml --resume runs/checkpoints/last_model.pt

作者：重构版本
日期：2025-10
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Tuple

import yaml
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

# 导入自定义模块
from models.gat_lstm_model import SpatialTemporalGATLSTM, create_model_from_config
from models.astgcn_model import ASTGCN, create_astgcn_from_config
from models.gwnet_model import GraphWaveNet, create_gwnet_from_config
from lib.data_loader import create_dataloaders, prepare_adjacency_matrix, StandardScaler
from lib.metrics import masked_mae_loss, masked_rmse_loss, MetricsCalculator, print_evaluation_results
from lib.trainer_utils import (
    CheckpointManager, EarlyStopping, TrainingHistory,
    get_lr_scheduler, set_seed, get_current_lr, print_model_summary, setup_logging
)
from lib.visualization import TensorBoardLogger, TrainingVisualizer


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    adj: torch.Tensor,
    scaler: StandardScaler,
    device: torch.device,
    grad_clip: float = 5.0,
    null_val: float = 0.0,
) -> Tuple[float, float]:
    """
    训练一个epoch

    参数：
        model: 模型
        dataloader: 训练数据加载器
        optimizer: 优化器
        adj: 邻接矩阵
        scaler: 数据标准化器
        device: 设备
        grad_clip: 梯度裁剪阈值
        null_val: 缺失值标记

    返回：
        (平均MAE损失, 平均RMSE损失)
    """
    model.train()

    total_mae_loss = 0.0
    total_rmse_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # 数据移到设备
        x = batch['x'].to(device)  # (B, T, N, F)
        y = batch['y'].to(device)  # (B, H, N, 1)

        # 前向传播
        optimizer.zero_grad()
        y_pred = model(x, adj)  # (B, H, N, 1)

        # 计算损失（在标准化数据上计算，保留梯度）
        mae_loss = masked_mae_loss(y_pred, y, null_val)
        rmse_loss = masked_rmse_loss(y_pred, y, null_val)

        # 反向传播（使用MAE作为主要损失）
        loss = mae_loss
        loss.backward()

        # 梯度裁剪
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # 更新参数
        optimizer.step()

        # 累积损失
        total_mae_loss += mae_loss.item()
        total_rmse_loss += rmse_loss.item()
        num_batches += 1

        # 打印进度（每10个batch）
        if (batch_idx + 1) % 10 == 0:
            logging.debug(
                f"  Batch [{batch_idx + 1}/{len(dataloader)}] "
                f"MAE: {mae_loss.item():.4f}, RMSE: {rmse_loss.item():.4f}"
            )

    # 返回平均损失
    avg_mae = total_mae_loss / num_batches
    avg_rmse = total_rmse_loss / num_batches

    return avg_mae, avg_rmse


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    adj: torch.Tensor,
    scaler: StandardScaler,
    device: torch.device,
    null_val: float = 0.0,
) -> Dict[str, float]:
    """
    验证模型

    参数：
        model: 模型
        dataloader: 验证数据加载器
        adj: 邻接矩阵
        scaler: 数据标准化器
        device: 设备
        null_val: 缺失值标记

    返回：
        包含MAE和RMSE的字典
    """
    model.eval()

    # 使用MetricsCalculator累积所有预测
    metrics_calc = MetricsCalculator(null_val=null_val)

    for batch in dataloader:
        x = batch['x'].to(device)
        y = batch['y'].to(device)

        # 前向传播
        y_pred = model(x, adj)

        # 反标准化
        y_true = scaler.inverse_transform(y.cpu().numpy())
        y_pred_real = scaler.inverse_transform(y_pred.cpu().numpy())

        # 更新指标计算器
        metrics_calc.update(
            torch.from_numpy(y_pred_real),
            torch.from_numpy(y_true)
        )

    # 计算最终指标
    results = metrics_calc.compute()

    # 提取总体指标
    overall = results.get('overall', {})

    return {
        'MAE': overall.get('MAE', 0.0),
        'RMSE': overall.get('RMSE', 0.0),
        'MAPE': overall.get('MAPE', 0.0),
    }


def main(
    config_path: str,
    resume_path: str = None,
    model_type: str = None,
    experiment_name: str = None,
):
    """
    主训练函数

    参数：
        config_path: 配置文件路径
        resume_path: 恢复训练的checkpoint路径（可选）
        model_type: 模型类型（gat_lstm / astgcn），命令行优先，其次配置，最后默认gat_lstm
        experiment_name: 实验名，用于区分同一模型的多次运行；
                         命令行优先，其次配置 train.experiment_name；
                         若均为空，则不创建额外子目录
    """
    # ==================== 加载配置 ====================
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    data_cfg = config['data']
    model_cfg = config['model']
    train_cfg = config['train']

    # ==================== 解析模型类型 ====================
    # 优先级：命令行参数 > 配置文件 > 默认 gat_lstm
    if model_type is None:
        config_model_type = model_cfg.get('model_type', None)
        if config_model_type:
            model_type = str(config_model_type)
        else:
            model_type = 'gat_lstm'
    model_type = model_type.lower()

    # ==================== 设置随机种子 ====================
    seed = train_cfg.get('seed', 42)
    set_seed(seed)

    # ==================== 解析实验名 ====================
    # 优先级：命令行参数 > 配置文件 train.experiment_name > None
    if experiment_name is None:
        experiment_name = train_cfg.get('experiment_name', None)

    # ==================== 设置日志目录（按模型类型、实验分桶） ====================
    # log_root 作为基础目录，例如 runs/logs
    log_root = Path(train_cfg.get('log_dir', 'runs/logs'))

    # 先按模型类型组织目录
    if log_root.name in ('gat_lstm', 'astgcn'):
        model_log_root = log_root
    else:
        model_log_root = log_root / model_type

    # 再按实验名进行细分（如果提供），否则直接使用模型级目录
    if experiment_name:
        log_dir = model_log_root / experiment_name
    else:
        log_dir = model_log_root
    setup_logging(log_dir=str(log_dir), log_file='train.log', level=logging.INFO)

    logging.info("=" * 70)
    logging.info("开始训练时空图神经网络模型")
    logging.info("=" * 70)
    logging.info(f"配置文件: {config_path}")
    logging.info(f"日志目录: {log_dir}")

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
        pin_memory=data_cfg.get('pin_memory', True),
        shuffle_train=data_cfg.get('shuffle_train', True),
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

    # ==================== 创建模型 ====================
    logging.info("\n" + "-" * 70)
    logging.info("创建模型...")
    logging.info("-" * 70)

    # 将最终使用的模型类型写回配置，保证保存到checkpoint/日志的一致性
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

    # 打印模型摘要
    print_model_summary(model)

    # 打印详细参数统计
    model_size_info = model.get_model_size()
    logging.info("模型参数详情:")
    for key, value in model_size_info.items():
        logging.info(f"  {key}: {value:,}")

    # ==================== 创建优化器 ====================
    optimizer_type = train_cfg.get('optimizer', 'adam').lower()
    base_lr = train_cfg.get('base_lr', 1e-3)
    weight_decay = train_cfg.get('weight_decay', 0.0)

    if optimizer_type == 'adam':
        optimizer = Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"不支持的优化器: {optimizer_type}")

    logging.info(f"优化器: {optimizer_type.upper()}, 学习率: {base_lr}, 权重衰减: {weight_decay}")

    # ==================== 创建学习率调度器 ====================
    scheduler = None
    if 'lr_scheduler' in train_cfg:
        scheduler_cfg = train_cfg['lr_scheduler']
        scheduler = get_lr_scheduler(optimizer, **scheduler_cfg)
        logging.info(f"学习率调度器: {scheduler_cfg.get('scheduler_type', 'None')}")

    # ==================== 设置训练工具 ====================
    # Checkpoint管理器
    ckpt_dir = train_cfg.get('ckpt_dir', 'runs/checkpoints')
    ckpt_manager = CheckpointManager(save_dir=ckpt_dir, keep_last_n=3)

    # 早停机制
    patience = train_cfg.get('patience', 20)
    early_stopping = EarlyStopping(patience=patience, mode='min', verbose=True)

    # 训练历史记录
    history = TrainingHistory(save_dir=log_dir)

    # ==================== 可视化工具 ====================
    # TensorBoard实时监控（推荐）
    use_tensorboard = train_cfg.get('use_tensorboard', True)
    tb_logger = TensorBoardLogger(
        log_dir=str(log_dir / 'tensorboard'),
        enabled=use_tensorboard
    )

    # Matplotlib图表生成器
    visualizer = TrainingVisualizer(save_dir=str(log_dir / 'figures'))

    # ==================== 恢复训练（如果指定） ====================
    start_epoch = 0
    best_val_mae = float('inf')

    if resume_path and os.path.exists(resume_path):
        logging.info(f"\n从checkpoint恢复训练: {resume_path}")
        checkpoint = ckpt_manager.load_checkpoint(
            checkpoint_name=Path(resume_path).name,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_mae = checkpoint.get('best_val_mae', float('inf'))

        # 恢复scaler
        if 'scaler' in checkpoint:
            scaler = StandardScaler.from_dict(checkpoint['scaler'])
            logging.info(f"已恢复标准化器: {scaler}")

    # ==================== 训练循环 ====================
    num_epochs = train_cfg.get('epochs', 100)
    grad_clip = train_cfg.get('grad_clip', 5.0)
    null_val = data_cfg.get('null_val', 0.0)

    logging.info("\n" + "=" * 70)
    logging.info("开始训练")
    logging.info("=" * 70)
    logging.info(f"总轮次: {num_epochs}, 起始轮次: {start_epoch}")
    logging.info(f"早停耐心值: {patience}")
    logging.info(f"梯度裁剪: {grad_clip}")

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()

        # ========== 训练阶段 ==========
        train_mae, train_rmse = train_one_epoch(
            model=model,
            dataloader=loaders['train'],
            optimizer=optimizer,
            adj=adj,
            scaler=scaler,
            device=device,
            grad_clip=grad_clip,
            null_val=null_val,
        )

        # ========== 验证阶段 ==========
        val_metrics = validate(
            model=model,
            dataloader=loaders['val'],
            adj=adj,
            scaler=scaler,
            device=device,
            null_val=null_val,
        )

        val_mae = val_metrics['MAE']
        val_rmse = val_metrics['RMSE']
        val_mape = val_metrics['MAPE']

        # ========== 学习率调度 ==========
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_mae)
            else:
                scheduler.step()

        current_lr = get_current_lr(optimizer)

        # ========== 计算耗时 ==========
        epoch_time = time.time() - epoch_start_time

        # ========== 打印训练信息 ==========
        logging.info(
            f"Epoch [{epoch:03d}/{num_epochs}] | "
            f"Train MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f} | "
            f"Val MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, MAPE: {val_mape:.2f}% | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # ========== 更新训练历史 ==========
        history.update(
            epoch=epoch,
            train_loss=train_mae,
            val_loss=val_mae,
            metrics={'MAE': val_mae, 'RMSE': val_rmse, 'MAPE': val_mape},
            lr=current_lr
        )

        # ========== 记录到TensorBoard ==========
        tb_logger.log_scalar('Loss/train', train_mae, epoch)
        tb_logger.log_scalar('Loss/val', val_mae, epoch)
        tb_logger.log_train_val_comparison('MAE', train_mae, val_mae, epoch)

        # 记录验证指标
        tb_logger.log_scalar('Metrics/val_RMSE', val_rmse, epoch)
        tb_logger.log_scalar('Metrics/val_MAPE', val_mape, epoch)

        # 记录学习率
        tb_logger.log_scalar('LR', current_lr, epoch)

        # ========== 检查是否为最佳模型 ==========
        is_best = val_mae < best_val_mae
        if is_best:
            best_val_mae = val_mae

        # ========== 保存Checkpoint ==========
        ckpt_manager.save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            val_loss=val_mae,
            val_metrics=val_metrics,
            train_loss=train_mae,
            config=config,
            is_best=is_best,
            scaler=scaler.to_dict(),
            best_val_mae=best_val_mae,
        )

        # ========== 早停检查 ==========
        if early_stopping(val_mae):
            logging.info(f"\n⛔ 早停触发！最佳验证MAE: {best_val_mae:.4f}")
            break

    # ==================== 训练结束 ====================
    logging.info("\n" + "=" * 70)
    logging.info("训练完成！")
    logging.info("=" * 70)
    logging.info(f"最佳验证MAE: {best_val_mae:.4f}")

    # 关闭TensorBoard
    tb_logger.close()

    # 保存训练历史
    history.save('training_history.json')
    logging.info(f"训练历史已保存")

    # 生成训练可视化图表
    logging.info("\n生成训练可视化图表...")
    history_path = log_dir / 'training_history.json'
    try:
        visualizer.create_all_figures_from_history(str(history_path))
    except Exception as e:
        logging.warning(f"生成可视化图表失败: {e}")

    # ==================== 测试集评估 ====================
    logging.info("\n" + "-" * 70)
    logging.info("在测试集上评估最佳模型...")
    logging.info("-" * 70)

    # 加载最佳模型
    best_ckpt = ckpt_manager.load_checkpoint(
        checkpoint_name='best_model.pt',
        model=model,
        device=device
    )

    # 测试集评估
    test_metrics_calc = MetricsCalculator(null_val=null_val)

    model.eval()
    with torch.no_grad():
        for batch in loaders['test']:
            x = batch['x'].to(device)
            y = batch['y'].to(device)

            y_pred = model(x, adj)

            # 反标准化
            y_true = scaler.inverse_transform(y.cpu().numpy())
            y_pred_real = scaler.inverse_transform(y_pred.cpu().numpy())

            test_metrics_calc.update(
                torch.from_numpy(y_pred_real),
                torch.from_numpy(y_true)
            )

    # 计算并打印测试集指标
    test_results = test_metrics_calc.compute(horizons=[3, 6, 12])
    print_evaluation_results(test_results, title="测试集评估结果")

    # 保存测试结果（转换为 JSON 可序列化类型，避免 float32 报错）
    import json

    def convert_to_serializable(obj):
        if isinstance(obj, (float, int, str, bool)) or obj is None:
            return obj
        if isinstance(obj, (torch.Tensor,)):
            return convert_to_serializable(obj.detach().cpu().tolist())
        if isinstance(obj, (dict,)):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert_to_serializable(v) for v in obj]
        try:
            # 处理 numpy 标量，如 float32/float64
            import numpy as np
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except Exception:
            pass
        return obj

    serializable_results = convert_to_serializable(test_results)
    test_results_path = log_dir / 'test_results.json'
    with open(test_results_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    logging.info(f"测试结果已保存: {test_results_path}")

    logging.info("\n✓ 所有流程完成！")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练时空图神经网络模型')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='配置文件路径 (YAML格式)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='从checkpoint恢复训练的路径'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default=None,
        choices=['gat_lstm', 'astgcn', 'gwnet'],
        help='选择模型类型：gat_lstm（默认）/ astgcn / gwnet'
    )
    parser.add_argument(
        '--exp_name',
        type=str,
        default=None,
        help='实验名称（同一模型多次运行时用于区分日志目录）'
    )

    args = parser.parse_args()

    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"错误：配置文件不存在: {args.config}")
        sys.exit(1)

    # 开始训练
    main(
        config_path=args.config,
        resume_path=args.resume,
        model_type=args.model_type,
        experiment_name=args.exp_name,
    )
