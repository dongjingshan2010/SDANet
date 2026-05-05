"""
频域分解模块
对输入图像进行 FFT，分离幅度谱与相位谱，并重建相位图像和对数幅度图像。
"""

import torch
import torch.nn as nn


class FrequencyDecomposer(nn.Module):
    """
    将输入图像分解为：
      - log_amplitude : 对数幅度图（含域风格信息）
      - phase_img     : 相位重建图（含语义结构信息）

    两者均归一化到 [0, 1] 并保持与输入相同的通道数和空间尺寸，
    可直接送入预训练 CNN / Transformer。

    前向输入：
        x : Tensor [B, C, H, W]，已经过标准化的图像

    前向输出：
        log_amplitude : Tensor [B, C, H, W]
        phase_img     : Tensor [B, C, H, W]
    """

    def forward(self, x: torch.Tensor):
        # --- 1. FFT + fftshift（将低频移到中心）---
        fft = torch.fft.fft2(x, norm="ortho")                        # [B,C,H,W] 复数
        fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))

        # --- 2. 幅度谱 ---
        amplitude = torch.abs(fft_shifted)                            # [B,C,H,W] 实数
        log_amplitude = torch.log1p(amplitude)                        # log(1+|A|) 压缩动态范围

        # --- 3. 相位谱 → 重建相位图像 ---
        phase = torch.angle(fft_shifted)                              # [B,C,H,W] 实数, [-π, π]
        # 用单位幅度 + 原始相位重建，只保留结构信息，去除风格
        unit_fft = torch.exp(1j * phase)                              # [B,C,H,W] 单位复数
        phase_img = torch.fft.ifft2(
            torch.fft.ifftshift(unit_fft, dim=(-2, -1)),
            norm="ortho"
        ).real                                                        # [B,C,H,W] 实数

        # --- 4. 逐样本归一化到 [0, 1] ---
        log_amplitude = self._minmax_normalize(log_amplitude)
        phase_img = self._minmax_normalize(phase_img)

        return log_amplitude, phase_img

    @staticmethod
    def _minmax_normalize(x: torch.Tensor) -> torch.Tensor:
        """逐样本 min-max 归一化，避免批次统计污染。"""
        B = x.shape[0]
        x_flat = x.reshape(B, -1)
        x_min = x_flat.min(dim=1)[0].view(B, 1, 1, 1)
        x_max = x_flat.max(dim=1)[0].view(B, 1, 1, 1)
        return (x - x_min) / (x_max - x_min + 1e-8)
