"""
FreqDANN / SDANet：频域双流对抗域泛化网络（含消融变体支持）

架构总览（完整模式）：
┌─────────────────────────────────────────────────────────┐
│  输入图像 x [B,3,H,W]                                    │
│       │                                                 │
│  FrequencyDecomposer  [可禁用: use_frequency_decomp]     │
│   ├── log_amplitude [B,3,H,W]                           │
│   └── phase_img     [B,3,H,W]                           │
│       │                        │                        │
│  AmplitudeEncoder         PhaseEncoder (Swin-T)         │
│   └── amp_feat [B,D]       └── phase_feat [B,D]         │
│       │                        │                        │
│       ├───── GRL ──────────────┤                        │
│       │      │           Concat [B,2D]  [stream_mode]   │
│  DomainDisc  │                 │                        │
│  (L_adv)     │          ClassHead → L_cls               │
│              │                                          │
│         ConsistencyLoss (phase_feat ↔ amp_feat)         │
└─────────────────────────────────────────────────────────┘

消融变体（通过构造参数控制）：
  use_frequency_decomp=False → 直接使用原始图像（跳过 FFT 分解）
  stream_mode='phase_only'   → 仅使用相位编码器（禁用幅度流和对抗训练）
  stream_mode='amp_only'     → 仅使用幅度编码器（禁用相位流）
  lambda_adv=0.0             → 禁用域对抗损失（在 TotalLoss 中设置）
  lambda_cons=0.0            → 禁用一致性损失（在 TotalLoss 中设置）

损失：
  L_total = L_cls + λ_adv * L_adv + λ_cons * L_cons
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .frequency import FrequencyDecomposer
from .grl import GradientReversalLayer
from .encoders import PhaseEncoder, AmplitudeEncoder
from .discriminator import DomainDiscriminator


class ClassificationHead(nn.Module):
    """
    分类头，支持双流（dual）和单流（phase_only / amp_only）两种输入模式。

    Args:
        feature_dim   : 单个编码器的输出维度 D
        num_classes   : 分类类别数
        single_stream : 若为 True，输入维度为 D（单流）；否则为 2D（双流拼接）
    """

    def __init__(self, feature_dim: int, num_classes: int, single_stream: bool = False):
        super().__init__()
        in_dim = feature_dim if single_stream else feature_dim * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, num_classes),
        )
        self.single_stream = single_stream

    def forward(
        self,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            feat_a : [B, D] — 主特征（单流时即全部输入；双流时为相位特征）
            feat_b : [B, D] — 幅度特征（仅双流时使用）
        """
        if self.single_stream or feat_b is None:
            combined = feat_a
        else:
            combined = torch.cat([feat_a, feat_b], dim=1)   # [B, 2D]
        return self.net(combined)                            # [B, num_classes]


class FreqDANN(nn.Module):
    """
    SDANet / FreqDANN：频域双流对抗域泛化网络，支持消融变体。

    Args:
        phase_encoder_name  : Swin-Transformer 模型名（timm）
        amp_encoder_name    : ResNet 模型名（timm）
        feature_dim         : 各编码器投影后的统一维度
        disc_hidden_dim     : 域判别器隐层维度
        num_classes         : 分类类别数
        num_domains         : 源域中心数量（域判别器输出维度）
        pretrained          : 是否使用预训练权重
        use_frequency_decomp: 是否启用 FFT 频域分解（消融：设 False 则直接用原图）
        stream_mode         : 流模式，'dual'（默认）|'phase_only'|'amp_only'
    """

    def __init__(
        self,
        phase_encoder_name: str = "swin_tiny_patch4_window7_224",
        amp_encoder_name: str = "resnet50",
        feature_dim: int = 512,
        disc_hidden_dim: int = 256,
        num_classes: int = 2,
        num_domains: int = 2,
        pretrained: bool = True,
        use_frequency_decomp: bool = True,
        stream_mode: str = "dual",  # "dual" | "phase_only" | "amp_only"
    ):
        super().__init__()

        assert stream_mode in ("dual", "phase_only", "amp_only"), \
            f"stream_mode 必须是 'dual', 'phase_only' 或 'amp_only'，得到: {stream_mode}"

        self.use_frequency_decomp = use_frequency_decomp
        self.stream_mode = stream_mode

        # 频域分解（无参数，可跳过）
        self.decomposer = FrequencyDecomposer()

        # 按流模式选择性创建编码器
        single_stream = stream_mode in ("phase_only", "amp_only")

        if stream_mode != "amp_only":
            self.phase_encoder = PhaseEncoder(phase_encoder_name, pretrained, feature_dim)
        else:
            self.phase_encoder = None

        if stream_mode != "phase_only":
            self.amp_encoder = AmplitudeEncoder(amp_encoder_name, pretrained, feature_dim)
            # 梯度反转层与域判别器（仅幅度流存在时有意义）
            self.grl = GradientReversalLayer(alpha=1.0)
            self.domain_disc = DomainDiscriminator(
                feature_dim, disc_hidden_dim, num_domains=num_domains,
            )
        else:
            self.amp_encoder = None
            self.grl = None
            self.domain_disc = None

        # 分类头
        self.cls_head = ClassificationHead(feature_dim, num_classes, single_stream=single_stream)

    # ------------------------------------------------------------------ #
    # 核心前向                                                             #
    # ------------------------------------------------------------------ #

    def _decompose(self, x: torch.Tensor):
        """
        根据 use_frequency_decomp 决定是否进行 FFT 分解。
        返回 (log_amplitude_or_raw, phase_img_or_raw)。
        """
        if self.use_frequency_decomp:
            return self.decomposer(x)          # (log_amplitude, phase_img)
        else:
            return x, x                        # 消融：两个流均接收原始图像

    def encode(self, x: torch.Tensor):
        """
        提取编码特征（支持消融变体）。

        Returns:
            amp_feat   : [B, D] 或 None（phase_only 模式）
            phase_feat : [B, D] 或 None（amp_only 模式）
        """
        log_amplitude, phase_img = self._decompose(x)

        amp_feat = self.amp_encoder(log_amplitude) if self.amp_encoder is not None else None
        phase_feat = self.phase_encoder(phase_img) if self.phase_encoder is not None else None

        return amp_feat, phase_feat

    def forward(self, x: torch.Tensor):
        """
        完整前向，返回用于损失计算的所有张量。

        Returns:
            cls_logits    : [B, num_classes]
            domain_logits : [B, num_domains] 或 None（phase_only 或幅度流关闭时）
            amp_feat      : [B, D] 或 None
            phase_feat    : [B, D] 或 None
        """
        amp_feat, phase_feat = self.encode(x)

        # 分类分支
        if self.stream_mode == "phase_only":
            cls_logits = self.cls_head(phase_feat)
        elif self.stream_mode == "amp_only":
            cls_logits = self.cls_head(amp_feat)
        else:  # dual
            cls_logits = self.cls_head(phase_feat, amp_feat)

        # 域对抗分支（仅当幅度编码器存在时）
        domain_logits = None
        if self.amp_encoder is not None and self.grl is not None:
            amp_reversed = self.grl(amp_feat)
            domain_logits = self.domain_disc(amp_reversed)

        return cls_logits, domain_logits, amp_feat, phase_feat

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """推理接口，仅返回分类 logits。"""
        amp_feat, phase_feat = self.encode(x)
        if self.stream_mode == "phase_only":
            return self.cls_head(phase_feat)
        elif self.stream_mode == "amp_only":
            return self.cls_head(amp_feat)
        else:
            return self.cls_head(phase_feat, amp_feat)

    def set_grl_alpha(self, alpha: float):
        """由训练循环动态更新 GRL 的反转系数。"""
        if self.grl is not None:
            self.grl.set_alpha(alpha)
