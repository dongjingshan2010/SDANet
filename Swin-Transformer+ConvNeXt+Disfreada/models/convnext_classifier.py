"""
ConvNeXt 医学图像分类模型

架构总览：
┌─────────────────────────────────────────┐
│  输入图像 x [B,3,H,W]                    │
│       │                                 │
│  ConvNeXt Backbone (timm)               │
│       │                                 │
│  特征 feat [B,D]                        │
│       │                                 │
│  ClassificationHead                     │
│       │                                 │
│  输出 logits [B,num_classes]            │
└─────────────────────────────────────────┘

损失：
  L_total = L_cls (交叉熵分类损失)
"""

from __future__ import annotations

import warnings

import timm
import torch
import torch.nn as nn


class ConvNeXtClassifier(nn.Module):
    """
    基于 ConvNeXt 的医学图像分类器。

    Args:
        model_name : ConvNeXt 模型名（timm），默认 convnext_tiny
        num_classes: 分类类别数
        hidden_dim : 分类头隐层维度
        pretrained : 是否使用预训练权重
    """

    def __init__(
        self,
        model_name: str = "convnext_tiny",
        num_classes: int = 2,
        hidden_dim: int = 512,
        pretrained: bool = True,
    ):
        super().__init__()

        # ConvNeXt 骨干网络
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,       # 关闭分类头
                global_pool="avg",   # 全局平均池化
            )
        except Exception as e:
            if pretrained:
                warnings.warn(
                    f"无法加载预训练权重（{e}），将使用随机初始化。"
                    f"若需离线运行，请提前手动下载模型或设置 pretrained=False。"
                )
                self.backbone = timm.create_model(
                    model_name,
                    pretrained=False,
                    num_classes=0,
                    global_pool="avg",
                )
            else:
                raise e

        backbone_dim = self.backbone.num_features  # convnext_tiny: 768

        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(backbone_dim),
            nn.Linear(backbone_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，返回分类 logits。

        Args:
            x : [B, 3, H, W] 输入图像（已归一化）

        Returns:
            logits : [B, num_classes]
        """
        feat   = self.backbone(x)       # [B, backbone_dim]
        logits = self.classifier(feat)  # [B, num_classes]
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """推理接口，返回分类 logits。"""
        return self.forward(x)
