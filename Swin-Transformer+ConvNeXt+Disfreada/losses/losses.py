"""
损失函数定义（基础版本）

ClassificationLoss：
    L_total = L_cls (交叉熵分类损失)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
    """交叉熵分类损失，支持类别不平衡权重。"""

    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            logits : [B, num_classes]
            labels : [B]  long tensor

        Returns:
            loss   : 标量损失值
            details: 损失详情字典
        """
        loss = self.criterion(logits, labels)
        details = {
            "loss_cls": loss.item(),
            "loss_total": loss.item(),
        }
        return loss, details
