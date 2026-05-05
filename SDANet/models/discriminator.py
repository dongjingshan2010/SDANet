"""
域判别器（Domain Discriminator）

接收幅度编码器输出的特征（经过 GRL 反转梯度），判断特征来自哪个中心。
域标签：0 = 源域（重庆/金城），1 = 目标域（sw/淄博）
"""

import torch
import torch.nn as nn


class DomainDiscriminator(nn.Module):
    """
    三层 MLP 域判别器。

    Args:
        feature_dim : 输入特征维度（与编码器投影维度一致）
        hidden_dim  : 隐层维度
        num_domains : 域数量（默认 2）
    """

    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        num_domains: int = 2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, feature_dim] 经过 GRL 的幅度特征
        Returns:
            logits: [B, num_domains]
        """
        return self.net(x)
