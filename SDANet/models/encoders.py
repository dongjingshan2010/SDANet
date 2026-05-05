"""
双流编码器

PhaseEncoder    : Swin-Transformer 骨干，编码相位图像（语义结构流）
AmplitudeEncoder: ResNet-50 骨干，编码对数幅度图像（域风格流）

两个编码器均附带投影头，将各自维度投影到统一的 feature_dim，
方便后续拼接分类及计算一致性损失。
"""

import torch
import torch.nn as nn
import timm
import warnings


# --------------------------------------------------------------------------- #
# 相位编码器（Swin-Transformer）                                               #
# --------------------------------------------------------------------------- #

class PhaseEncoder(nn.Module):
    """
    以 Swin-Transformer 作为骨干，提取相位图像的语义特征。

    Args:
        model_name  : timm 中的 Swin-T 模型名称
        pretrained  : 是否加载 ImageNet 预训练权重（若网络不通会自动降级为 False）
        feature_dim : 投影后输出维度
    """

    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        feature_dim: int = 512,
    ):
        super().__init__()
        # 尝试创建带预训练权重的骨干，若失败则回退为不加载预训练
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,          # 关闭分类头
                global_pool="avg",      # 全局平均池化
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

        backbone_dim = self.backbone.num_features   # Swin-Tiny: 768

        # 投影头：将骨干输出投影到统一 feature_dim
        self.projector = nn.Sequential(
            nn.LayerNorm(backbone_dim),
            nn.Linear(backbone_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] 归一化相位图像
        Returns:
            features: [B, feature_dim]
        """
        feat = self.backbone(x)          # [B, backbone_dim]
        return self.projector(feat)      # [B, feature_dim]


# --------------------------------------------------------------------------- #
# 幅度编码器（ResNet-50）                                                       #
# --------------------------------------------------------------------------- #

class AmplitudeEncoder(nn.Module):
    """
    以 ResNet-50 作为骨干，提取对数幅度图的特征，用于域对抗训练。

    Args:
        model_name  : timm 中的 ResNet 模型名称
        pretrained  : 是否加载 ImageNet 预训练权重（若网络不通会自动降级为 False）
        feature_dim : 投影后输出维度
    """

    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = True,
        feature_dim: int = 512,
    ):
        super().__init__()
        # 尝试创建带预训练权重的骨干，若失败则回退为不加载预训练
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool="avg",
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

        backbone_dim = self.backbone.num_features   # ResNet-50: 2048

        self.projector = nn.Sequential(
            nn.BatchNorm1d(backbone_dim),
            nn.Linear(backbone_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] 归一化对数幅度图
        Returns:
            features: [B, feature_dim]
        """
        feat = self.backbone(x)          # [B, backbone_dim]
        return self.projector(feat)      # [B, feature_dim]