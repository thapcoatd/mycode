"""
简化版可视化生成脚本
不依赖lib模块，直接生成训练可视化图表
"""

import json
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def load_training_history(history_path):
    """加载训练历史"""
    with open(history_path, 'r', encoding='utf-8') as f:
        history = json.load(f)
    return history


def plot_loss_curves(history, save_dir):
    """绘制loss曲线"""
    epochs = history['epoch']
    train_loss = history['train_loss']
    val_loss = history['val_loss']

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='训练Loss', color='#3498db', linewidth=2)
    plt.plot(epochs, val_loss, label='验证Loss', color='#e74c3c', linewidth=2)

    # 标注最佳epoch
    best_epoch_idx = val_loss.index(min(val_loss))
    best_epoch = epochs[best_epoch_idx]
    best_val_loss = val_loss[best_epoch_idx]

    plt.scatter([best_epoch], [best_val_loss], color='#2ecc71', s=100, zorder=5, label=f'最佳 (Epoch {best_epoch})')
    plt.axvline(x=best_epoch, color='#2ecc71', linestyle='--', alpha=0.5)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MAE Loss', fontsize=12)
    plt.title('训练/验证Loss曲线', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = Path(save_dir) / 'loss_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Loss曲线已保存: {save_path}")


def plot_metrics_curves(history, save_dir):
    """绘制评估指标曲线"""
    epochs = history['epoch']
    metrics_list = history['metrics']

    # 提取MAE, RMSE, MAPE
    mae = [m['MAE'] for m in metrics_list]
    rmse = [m['RMSE'] for m in metrics_list]
    mape = [m['MAPE'] for m in metrics_list]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # MAE
    axes[0].plot(epochs, mae, color='#3498db', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MAE')
    axes[0].set_title('验证集 MAE')
    axes[0].grid(True, alpha=0.3)

    # RMSE
    axes[1].plot(epochs, rmse, color='#9b59b6', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('验证集 RMSE')
    axes[1].grid(True, alpha=0.3)

    # MAPE
    axes[2].plot(epochs, mape, color='#e67e22', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('MAPE (%)')
    axes[2].set_title('验证集 MAPE')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = Path(save_dir) / 'metrics_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] 指标曲线已保存: {save_path}")


def plot_learning_rate(history, save_dir):
    """绘制学习率变化"""
    epochs = history['epoch']
    lr = history['learning_rate']

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, lr, color='#16a085', linewidth=2, marker='o', markersize=3)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('学习率', fontsize=12)
    plt.title('学习率变化曲线', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()

    save_path = Path(save_dir) / 'learning_rate.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] 学习率曲线已保存: {save_path}")


def plot_training_dashboard(history, save_dir):
    """生成综合训练仪表盘"""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    epochs = history['epoch']
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    metrics_list = history['metrics']
    lr = history['learning_rate']

    rmse = [m['RMSE'] for m in metrics_list]
    mape = [m['MAPE'] for m in metrics_list]

    # 1. Loss曲线（左上）
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, train_loss, label='训练Loss', color='#3498db', linewidth=2)
    ax1.plot(epochs, val_loss, label='验证Loss', color='#e74c3c', linewidth=2)
    best_epoch_idx = val_loss.index(min(val_loss))
    best_epoch = epochs[best_epoch_idx]
    best_val_loss = val_loss[best_epoch_idx]
    ax1.scatter([best_epoch], [best_val_loss], color='#2ecc71', s=100, zorder=5)
    ax1.axvline(x=best_epoch, color='#2ecc71', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MAE Loss')
    ax1.set_title('训练/验证Loss', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. RMSE和MAPE（右上）
    ax2 = fig.add_subplot(gs[0, 1])
    ax2_twin = ax2.twinx()
    ax2.plot(epochs, rmse, label='RMSE', color='#9b59b6', linewidth=2)
    ax2_twin.plot(epochs, mape, label='MAPE (%)', color='#e67e22', linewidth=2, linestyle='--')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('RMSE', color='#9b59b6')
    ax2_twin.set_ylabel('MAPE (%)', color='#e67e22')
    ax2.set_title('验证集评估指标', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#9b59b6')
    ax2_twin.tick_params(axis='y', labelcolor='#e67e22')
    ax2.grid(True, alpha=0.3)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # 3. 学习率（左下）
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, lr, color='#16a085', linewidth=2, marker='o', markersize=3)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('学习率')
    ax3.set_title('学习率变化', fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, which='both')

    # 4. 统计摘要（右下）
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    summary_text = f"""
    训练统计摘要
    {'='*40}

    总轮次: {len(epochs)}
    最佳Epoch: {best_epoch}

    最佳验证指标:
      MAE:  {min(val_loss):.4f}
      RMSE: {min(rmse):.4f}
      MAPE: {min(mape):.2f}%

    最终验证指标:
      MAE:  {val_loss[-1]:.4f}
      RMSE: {rmse[-1]:.4f}
      MAPE: {mape[-1]:.2f}%

    最终学习率: {lr[-1]:.6f}

    过拟合程度:
      训练Loss: {train_loss[-1]:.4f}
      验证Loss: {val_loss[-1]:.4f}
      差距: {abs(val_loss[-1] - train_loss[-1]):.4f}
    """

    ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    fig.suptitle('训练过程综合仪表盘', fontsize=16, fontweight='bold')

    save_path = Path(save_dir) / 'training_dashboard.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] 训练仪表盘已保存: {save_path}")


def main():
    print("=" * 70)
    print("生成训练可视化图表")
    print("=" * 70)

    history_path = 'runs/logs/training_history.json'
    save_dir = 'runs/logs/figures'

    # 创建保存目录
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 加载训练历史
    print(f"\n加载训练历史: {history_path}")
    history = load_training_history(history_path)
    print(f"[OK] 成功加载 {len(history['epoch'])} 个epoch的记录")

    # 生成各种图表
    print("\n生成图表...")
    plot_loss_curves(history, save_dir)
    plot_metrics_curves(history, save_dir)
    plot_learning_rate(history, save_dir)
    plot_training_dashboard(history, save_dir)

    print("\n" + "=" * 70)
    print("所有图表生成完成！")
    print("=" * 70)
    print(f"图表保存位置: {save_dir}/")
    print("  - loss_curves.png")
    print("  - metrics_curves.png")
    print("  - learning_rate.png")
    print("  - training_dashboard.png")


if __name__ == '__main__':
    main()