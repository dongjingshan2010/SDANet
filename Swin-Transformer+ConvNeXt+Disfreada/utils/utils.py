"""
训练工具函数
"""

from __future__ import annotations

import os
import random
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight


# ----------------------------------------------------------------------- #
# 随机种子                                                                  #
# ----------------------------------------------------------------------- #

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------- #
# 类别权重（应对不平衡数据集）                                               #
# ----------------------------------------------------------------------- #

def compute_class_weights(
    dataset, num_classes: int = 2, device: str = "cuda"
) -> torch.Tensor:
    """
    根据数据集的类别分布计算逆频率权重。
    兼容 MedicalImageDataset / MultiCenterDataset 及其 Subset 包装：
        样本元组形如 (path, class_label) 或 (path, class_label, domain_label)，
        类别标签均位于索引 [1]。
    若数据集中某些类别完全缺失（例如 chongqing_c 的 normal 为空但其他中心
    仍提供了 normal 样本），仍可正确计算；当全数据完全缺类时退化为均匀权重。
    """
    labels = []
    if hasattr(dataset, "samples"):
        labels = [s[1] for s in dataset.samples]
    elif hasattr(dataset, "dataset"):       # torch.utils.data.Subset
        indices = dataset.indices
        base = dataset.dataset
        if hasattr(base, "samples"):
            for i in indices:
                labels.append(base.samples[i][1])

    if not labels:
        return torch.ones(num_classes, device=device)

    labels = np.array(labels)

    # 构造手动逆频率权重，兼容某类缺失的情况（避免 sklearn 抛错）。
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    n_total = counts.sum()
    n_present = (counts > 0).sum()

    if n_present == 0:
        weights = np.ones(num_classes, dtype=np.float32)
    elif n_present == num_classes:
        # 等价于 sklearn 的 balanced：N / (K * count_k)
        weights = n_total / (n_present * counts)
    else:
        # 至少一个类别缺失：对存在的类别用逆频率，对缺失类别用 0 权重
        weights = np.zeros(num_classes, dtype=np.float64)
        present_mask = counts > 0
        weights[present_mask] = n_total / (n_present * counts[present_mask])

    return torch.tensor(weights, dtype=torch.float32, device=device)


# ----------------------------------------------------------------------- #
# 训练指标跟踪                                                              #
# ----------------------------------------------------------------------- #

class AverageMeter:
    """跟踪单个标量值的滚动均值。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


class MetricTracker:
    """按 epoch 收集并记录多个指标。"""

    def __init__(self):
        self.history: Dict[str, List[float]] = defaultdict(list)

    def update(self, metrics: Dict[str, float]):
        for k, v in metrics.items():
            self.history[k].append(v)

    def latest(self, key: str) -> float:
        vals = self.history.get(key, [])
        return vals[-1] if vals else 0.0

    def best(self, key: str, higher_is_better: bool = True) -> float:
        vals = self.history.get(key, [])
        if not vals:
            return 0.0
        return max(vals) if higher_is_better else min(vals)


# ----------------------------------------------------------------------- #
# 早停                                                                      #
# ----------------------------------------------------------------------- #

class EarlyStopping:
    """
    当监控指标在 patience 个 epoch 内无改善时停止训练。

    Args:
        patience        : 容忍轮数
        higher_is_better: 指标是否越大越好（如 AUC）
        min_delta       : 最小改善阈值
    """

    def __init__(
        self,
        patience: int = 15,
        higher_is_better: bool = True,
        min_delta: float = 1e-4,
    ):
        self.patience = patience
        self.higher_is_better = higher_is_better
        self.min_delta = min_delta
        self.best_score: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        """
        更新状态，返回是否应该停止训练。
        """
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (
            score > self.best_score + self.min_delta
            if self.higher_is_better
            else score < self.best_score - self.min_delta
        )

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# ----------------------------------------------------------------------- #
# 检查点                                                                    #
# ----------------------------------------------------------------------- #

def save_checkpoint(
    model: nn.Module,
    optimizer,
    epoch: int,
    metrics: dict,
    filepath: str,
):
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(state, filepath)
    print(f"[Checkpoint] 保存至 {filepath}  (epoch={epoch})")


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer=None,
    device: str = "cuda",
) -> dict:
    state = torch.load(filepath, map_location=device)
    model.load_state_dict(state["model_state"])
    if optimizer is not None and "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])
    print(f"[Checkpoint] 加载自 {filepath}  (epoch={state.get('epoch', '?')})")
    return state.get("metrics", {})


# ----------------------------------------------------------------------- #
# 训练曲线可视化                                                             #
# ----------------------------------------------------------------------- #

def plot_training_curves(tracker: MetricTracker, save_dir: str):
    """绘制并保存训练/验证损失与指标曲线。"""
    os.makedirs(save_dir, exist_ok=True)
    history = tracker.history

    # --- 损失曲线 ---
    loss_keys = [k for k in history if "loss" in k]
    if loss_keys:
        fig, ax = plt.subplots(figsize=(10, 5))
        for k in loss_keys:
            ax.plot(history[k], label=k)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Losses")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "loss_curves.png"), dpi=150)
        plt.close(fig)

    # --- 分类指标曲线 ---
    metric_keys = [k for k in history if any(m in k for m in ("acc", "auc", "f1"))]
    if metric_keys:
        fig, ax = plt.subplots(figsize=(10, 5))
        for k in metric_keys:
            ax.plot(history[k], label=k)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_title("Classification Metrics")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "metric_curves.png"), dpi=150)
        plt.close(fig)

    print(f"[Plot] 训练曲线已保存至 {save_dir}")
