"""
梯度反转层（Gradient Reversal Layer, GRL）

前向传播：恒等映射
反向传播：梯度乘以 -alpha

通过 lambda 调度使训练初期对抗强度较弱，逐渐增强：
    lambda = 2 / (1 + exp(-gamma * p)) - 1
    p = current_iter / max_iter  ∈ [0, 1]
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class _GRLFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    梯度反转层。

    Args:
        alpha (float): 反转系数，通常由外部调度器动态设置。
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GRLFunction.apply(x, self.alpha)

    def set_alpha(self, alpha: float):
        self.alpha = alpha

    def extra_repr(self) -> str:
        return f"alpha={self.alpha:.4f}"


def grl_lambda_schedule(current_iter: int, max_iter: int, gamma: float = 10.0) -> float:
    """
    计算 GRL 的 lambda 值（DANN 原文调度）。

    Args:
        current_iter: 当前训练迭代数
        max_iter:     总迭代数（用于归一化 p）
        gamma:        调度陡峭系数

    Returns:
        lambda 值 ∈ [0, 1]
    """
    import math
    p = current_iter / max_iter
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0
