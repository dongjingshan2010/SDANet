"""
DisFreAda: Distribution-driven Multi-frequency Adaptive Network
医学图像分类模型（频域基线）

架构总览：
┌────────────────────────────────────────────────────────────────┐
│  输入图像 x [B, 3, H, W]                                        │
│       │                                                        │
│  FrequencyBandDecomposer  (FFT → 带通掩膜 → IFFT)              │
│  ─→ x_low / x_mid / x_high   各 [B, 3, H, W]                  │
│       │                                                        │
│  BandFeatureExtractor（权重共享轻量 CNN）                         │
│  ─→ f_low / f_mid / f_high   各 [B, D, H/16, W/16]            │
│       │                                                        │
│  DistributionStatisticsEncoder（均值/标准差/偏度/峰度）           │
│  ─→ s_low / s_mid / s_high   各 [B, 4D]                        │
│       │                                                        │
│  AdaptiveWeightGenerator（MLP on concat stats）                 │
│  ─→ weights [B, 3]  (softmax)                                  │
│       │                                                        │
│  加权融合：Σ wᵢ · GAP(fᵢ)  ─→  fused [B, D]                   │
│       │                                                        │
│  ClassificationHead                                            │
│  ─→ logits [B, num_classes]                                    │
└────────────────────────────────────────────────────────────────┘

设计说明：
  - FrequencyBandDecomposer：对 RGB 图像做 2D FFT，在频谱中心化后用
    圆形半径掩膜划分低/中/高三个频段，再 IFFT 回空域，保持图像尺寸不变。
  - BandFeatureExtractor：共享权重的轻量 CNN（4× Conv-BN-ReLU），
    避免三路独立骨干带来的参数爆炸。
  - DistributionStatisticsEncoder：沿空间维度计算四阶矩，
    用分布描述符刻画每个频段的内容差异。
  - AdaptiveWeightGenerator：将三路分布描述符拼接后送入 MLP，
    输出 softmax 归一化的自适应融合权重。
  - 整体为端到端可训练模型，无需额外预训练权重。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================= #
#  1. 频段分解模块                                                          #
# ======================================================================= #

class FrequencyBandDecomposer(nn.Module):
    """
    将输入图像分解为低频 / 中频 / 高频三个频段（空域表示）。

    频段半径（相对最大频率归一化，取值 0~1）：
        低频 :  [0,    r1)
        中频 :  [r1,   r2)
        高频 :  [r2,   1.0]

    Args:
        image_size : 输入分辨率 (H, W)
        band_radii : (r1, r2) 频段分割阈值，默认 (0.25, 0.6)
    """

    def __init__(
        self,
        image_size: tuple = (224, 224),
        band_radii: tuple = (0.25, 0.6),
    ):
        super().__init__()
        H, W = image_size
        cy, cx = H // 2, W // 2

        # 归一化频率半径网格
        y = torch.arange(H).float() - cy
        x = torch.arange(W).float() - cx
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        r = torch.sqrt(yy ** 2 + xx ** 2) / math.sqrt(cy ** 2 + cx ** 2)

        r1, r2 = band_radii
        # shape [1, 1, H, W] — 广播至 [B, C, H, W]
        self.register_buffer("mask_low",  (r <  r1).float().unsqueeze(0).unsqueeze(0))
        self.register_buffer("mask_mid",  ((r >= r1) & (r < r2)).float().unsqueeze(0).unsqueeze(0))
        self.register_buffer("mask_high", (r >= r2).float().unsqueeze(0).unsqueeze(0))

    # ------------------------------------------------------------------ #

    def _band_filter(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        带通滤波：FFT → 频移 → 掩膜 → 逆频移 → IFFT → 取实部。

        Args:
            x    : [B, C, H, W]
            mask : [1, 1, H, W]
        Returns:
            filtered : [B, C, H, W]  实数
        """
        X         = torch.fft.fft2(x)
        X_shift   = torch.fft.fftshift(X)
        X_masked  = X_shift * mask
        X_unshift = torch.fft.ifftshift(X_masked)
        return torch.fft.ifft2(X_unshift).real

    def forward(self, x: torch.Tensor):
        """
        Returns:
            (x_low, x_mid, x_high) : 各 [B, C, H, W]
        """
        return (
            self._band_filter(x, self.mask_low),
            self._band_filter(x, self.mask_mid),
            self._band_filter(x, self.mask_high),
        )


# ======================================================================= #
#  2. 频段特征提取器                                                        #
# ======================================================================= #

class _ConvBNReLU(nn.Module):
    """Conv2d-BatchNorm2d-ReLU 基础块。"""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _ResidualBlock(nn.Module):
    """带跳连的两层 Conv 块（步长均为 1）。"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = _ConvBNReLU(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.conv2(self.conv1(x)))


class BandFeatureExtractor(nn.Module):
    """
    权重共享的轻量 CNN，用于提取单个频段的空域特征。

    输入 :  [B, in_channels, H, W]
    输出 :  [B, feat_dim, H/16, W/16]

    网络结构：
        Stem  : 3→32  stride-2
        Stage1: 32→64  stride-2
        Stage2: 64→128 stride-2  + Residual
        Stage3: 128→feat_dim stride-2  + Residual
    """

    def __init__(self, in_channels: int = 3, feat_dim: int = 256):
        super().__init__()
        self.stem   = _ConvBNReLU(in_channels, 32,       stride=2)
        self.stage1 = _ConvBNReLU(32,          64,       stride=2)
        self.stage2 = nn.Sequential(
            _ConvBNReLU(64, 128, stride=2),
            _ResidualBlock(128),
        )
        self.stage3 = nn.Sequential(
            _ConvBNReLU(128, feat_dim, stride=2),
            _ResidualBlock(feat_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x   # [B, feat_dim, H/16, W/16]


# ======================================================================= #
#  3. 分布统计编码器                                                        #
# ======================================================================= #

class DistributionStatisticsEncoder(nn.Module):
    """
    将特征图编码为四阶矩分布描述符：均值、标准差、偏度、（超）峰度。

    输入 : [B, C, H, W]
    输出 : [B, 4 * C]

    每个统计量沿空间维度 (H×W) 逐通道计算：
        mean  = E[X]
        std   = √E[(X-μ)²]
        skew  = E[(X-μ)³] / σ³
        kurt  = E[(X-μ)⁴] / σ⁴  - 3   （超额峰度）
    """

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        x = feat.reshape(B, C, -1)                          # [B, C, N]

        mean = x.mean(dim=-1)                               # [B, C]
        std  = x.std(dim=-1).clamp(min=1e-6)               # [B, C]

        x_c  = (x - mean.unsqueeze(-1)) / std.unsqueeze(-1)
        skew = (x_c ** 3).mean(dim=-1)                     # [B, C]
        kurt = (x_c ** 4).mean(dim=-1) - 3.0               # [B, C]

        return torch.cat([mean, std, skew, kurt], dim=1)   # [B, 4C]


# ======================================================================= #
#  4. 自适应权重生成器                                                      #
# ======================================================================= #

class AdaptiveWeightGenerator(nn.Module):
    """
    以各频段的分布描述符为输入，通过 MLP 生成 softmax 归一化的
    频段融合权重。

    输入 : 各频段统计描述符拼接 → [B, num_bands * stat_dim]
    输出 : [B, num_bands]  (softmax)

    Args:
        stat_dim  : 单频段描述符维度（= 4 × feat_dim）
        num_bands : 频段数量，默认 3
        dropout   : MLP 中间 dropout 率
    """

    def __init__(
        self,
        stat_dim: int,
        num_bands: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        in_dim  = num_bands * stat_dim
        mid_dim = max(in_dim // 4, num_bands * 4)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, num_bands),
        )

    def forward(self, stats_list: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            stats_list : [s_low, s_mid, s_high], 各 [B, stat_dim]
        Returns:
            weights    : [B, num_bands]  softmax 归一化
        """
        x = torch.cat(stats_list, dim=1)        # [B, num_bands * stat_dim]
        return torch.softmax(self.mlp(x), dim=1)


# ======================================================================= #
#  5. 主模型                                                               #
# ======================================================================= #

class DisFreAdaClassifier(nn.Module):
    """
    Distribution-driven Multi-frequency Adaptive Network (DisFreAda)

    Args:
        num_classes : 分类类别数，默认 2
        feat_dim    : 每个频段的 CNN 特征维度，默认 256
        hidden_dim  : 分类头隐层维度，默认 512
        band_radii  : (r1, r2) 频段分割阈值，默认 (0.25, 0.6)
        image_size  : 输入图像分辨率 (H, W)，默认 (224, 224)
        dropout     : 自适应权重 MLP 的 dropout 率
    """

    NUM_BANDS: int = 3

    def __init__(
        self,
        num_classes: int = 2,
        feat_dim:    int = 256,
        hidden_dim:  int = 512,
        band_radii:  tuple = (0.25, 0.6),
        image_size:  tuple = (224, 224),
        dropout:     float = 0.1,
    ):
        super().__init__()

        # ── 1. 频段分解 ──────────────────────────────────────────────── #
        self.decomposer = FrequencyBandDecomposer(
            image_size=image_size,
            band_radii=band_radii,
        )

        # ── 2. 共享频段特征提取器 ──────────────────────────────────────── #
        self.band_extractor = BandFeatureExtractor(
            in_channels=3, feat_dim=feat_dim
        )

        # ── 3. 分布统计编码器（无参数） ───────────────────────────────── #
        self.dist_encoder = DistributionStatisticsEncoder()

        # ── 4. 自适应权重生成器 ───────────────────────────────────────── #
        stat_dim = 4 * feat_dim   # 四阶矩 × feat_dim 通道
        self.weight_gen = AdaptiveWeightGenerator(
            stat_dim=stat_dim,
            num_bands=self.NUM_BANDS,
            dropout=dropout,
        )

        # ── 5. 分类头 ─────────────────────────────────────────────────── #
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    # ------------------------------------------------------------------ #
    # 前向传播                                                             #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, 3, H, W] 输入图像（已归一化）

        Returns:
            logits : [B, num_classes]
        """
        # 1. 频段分解
        x_low, x_mid, x_high = self.decomposer(x)   # 各 [B,3,H,W]

        # 2. 共享 CNN 提取各频段特征
        f_low  = self.band_extractor(x_low)          # [B,D,h,w]
        f_mid  = self.band_extractor(x_mid)
        f_high = self.band_extractor(x_high)

        # 3. 全局平均池化 → 频段向量
        v_low  = F.adaptive_avg_pool2d(f_low,  1).flatten(1)  # [B,D]
        v_mid  = F.adaptive_avg_pool2d(f_mid,  1).flatten(1)
        v_high = F.adaptive_avg_pool2d(f_high, 1).flatten(1)

        # 4. 分布统计描述符
        s_low  = self.dist_encoder(f_low)    # [B,4D]
        s_mid  = self.dist_encoder(f_mid)
        s_high = self.dist_encoder(f_high)

        # 5. 自适应融合权重
        weights = self.weight_gen([s_low, s_mid, s_high])   # [B,3]

        # 6. 加权求和融合
        stacked = torch.stack([v_low, v_mid, v_high], dim=1)  # [B,3,D]
        fused   = (stacked * weights.unsqueeze(-1)).sum(dim=1) # [B,D]

        # 7. 分类
        return self.classifier(fused)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """推理接口，返回分类 logits。"""
        return self.forward(x)

    @torch.no_grad()
    def get_band_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        可解释性接口：返回当前批次的频段自适应权重。

        Returns:
            weights : [B, 3]  (low, mid, high 各频段权重)
        """
        x_low, x_mid, x_high = self.decomposer(x)
        f_low  = self.band_extractor(x_low)
        f_mid  = self.band_extractor(x_mid)
        f_high = self.band_extractor(x_high)
        s_low  = self.dist_encoder(f_low)
        s_mid  = self.dist_encoder(f_mid)
        s_high = self.dist_encoder(f_high)
        return self.weight_gen([s_low, s_mid, s_high])
