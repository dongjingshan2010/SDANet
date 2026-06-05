"""
项目全局配置
Swin Transformer 医学图像分类（基础版本）
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple


# 项目根目录（config.py 所在目录）
_BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class Config:
    # ------------------------------------------------------------------ #
    # 数据路径
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
    # Swin Transformer 模型名
    model_name: str = "swin_small_patch4_window7_224"
    # 分类头隐层维度
    hidden_dim: int = 1024

    # ------------------------------------------------------------------ #
    # 训练超参数
    # ------------------------------------------------------------------ #
    batch_size: int = 16
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    val_split: float = 0.2        # 源域中用于验证的比例
    warmup_epochs: int = 5

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
