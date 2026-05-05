"""
项目全局配置
频域双流对抗域泛化网络 (Frequency Domain Dual-Stream Adversarial Domain Generalization)
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple


# 项目根目录（config.py 所在目录）
# _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class Config:
    # ------------------------------------------------------------------ #
    # 数据路径
    #   多中心源域：每个中心一个目录，含 cancer/ 和 normal/ 子文件夹
    #   说明：chongqing_c 中 normal 类别可能为空（无样本），代码已做容错。
    # ------------------------------------------------------------------ #
    center_dirs: List[str] = field(default_factory=lambda: [
        os.path.join(_BASE_DIR, "data", "chongqing_c"),    # 中心 0
        os.path.join(_BASE_DIR, "data", "jincheng_nc"),    # 中心 1
    ])
    center_names: List[str] = field(default_factory=lambda: [
        "chongqing_c",
        "jincheng_nc",
    ])
    # 跨中心评估的目标域（不参与训练，仅 test.py 使用）
    target_dir: str = os.path.join(_BASE_DIR, "data", "sw_2zibo")
    output_dir: str = os.path.join(_BASE_DIR, "outputs")

    # ------------------------------------------------------------------ #
    # 图像参数
    # ------------------------------------------------------------------ #
    image_size: Tuple[int, int] = (224, 224)
    num_classes: int = 2          # cancer=1, normal=0

    # ------------------------------------------------------------------ #
    # 模型架构
    # ------------------------------------------------------------------ #
    # 相位编码器：Swin-Transformer（主分类器骨干）
    phase_encoder_name: str = "swin_tiny_patch4_window7_224"
    # 幅度编码器：ResNet-50（域对抗特征提取器）
    amp_encoder_name: str = "resnet50"
    # 投影后的统一特征维度
    feature_dim: int = 512
    # 域判别器隐层维度
    disc_hidden_dim: int = 256

    # ------------------------------------------------------------------ #
    # 训练超参数
    # ------------------------------------------------------------------ #
    batch_size: int = 16
    num_epochs: int = 100
    lr_encoder: float = 1e-4      # 编码器学习率（较小，微调预训练权重）
    lr_head: float = 1e-3         # 分类头 / 判别器学习率
    weight_decay: float = 1e-4
    val_split: float = 0.2        # 源域中用于验证的比例
    warmup_epochs: int = 5

    # ------------------------------------------------------------------ #
    # 损失权重
    # ------------------------------------------------------------------ #
    lambda_adv: float = 1.0       # 域对抗损失权重
    lambda_cons: float = 0.5      # 一致性损失权重

    # ------------------------------------------------------------------ #
    # GRL（梯度反转层）调度
    # ------------------------------------------------------------------ #
    grl_gamma: float = 10.0       # GRL lambda 调度系数
    grl_max_iter: int = 1000      # 用于计算 p 的分母（按 iter 计）

    # ------------------------------------------------------------------ #
    # 其他
    # ------------------------------------------------------------------ #
    seed: int = 42
    num_workers: int = 4
    device: str = "cuda"
    pretrained: bool = True
    save_best_metric: str = "val_auc"   # 按此指标保存最优模型

    # ------------------------------------------------------------------ #
    # 派生属性（向后兼容）
    # ------------------------------------------------------------------ #
    @property
    def num_domains(self) -> int:
        """域数量 = 训练源域中心数。"""
        return len(self.center_dirs)

    @property
    def source_dir(self) -> str:
        """向后兼容字段：返回首个中心目录。新代码请使用 center_dirs。"""
        return self.center_dirs[0] if self.center_dirs else ""

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "logs"), exist_ok=True)
