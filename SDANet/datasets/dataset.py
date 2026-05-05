"""
医学图像数据集（多中心 / 跨中心域泛化）

目录结构要求：
    center_dir/
        cancer/   ← 正样本 (label=1)
        normal/   ← 负样本 (label=0)

本模块提供两个数据集类：
  * MedicalImageDataset  ：单中心数据集（仅类别标签），用于跨中心测试评估。
  * MultiCenterDataset   ：多中心合并数据集，每条样本同时携带
                           "类别标签 (cancer/normal)" 与 "域标签 (中心索引)"，
                           用于训练域不变特征提取。

build_dataloaders：
    返回 (src_train_loader, src_val_loader, tgt_loader)
        src_train_loader / src_val_loader 来自 MultiCenterDataset，
        每个 batch 同时返回 (image, class_label, domain_label)；
        tgt_loader 来自 MedicalImageDataset，仅含类别标签，供测试评估。

鲁棒性：
  * 当某中心的某个类别目录不存在或为空（例如 chongqing_c 的 normal 为空）时，
    数据集会跳过该类别并打印警告，不会抛出异常。整个数据集只要至少有一个
    样本即可正常工作。
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
CLASS_MAP = {"cancer": 1, "normal": 0}


# ----------------------------------------------------------------------- #
# 单中心数据集（仅 (image, class_label) / (image, class_label, path)）       #
# ----------------------------------------------------------------------- #

class MedicalImageDataset(Dataset):
    """
    单中心医学图像数据集。

    Args:
        root_dir          : 数据根目录，内含 cancer/ 和 normal/ 子文件夹
        transform         : torchvision transforms
        return_path       : 若为 True，__getitem__ 额外返回文件路径（用于可视化）
        allow_empty_class : 若为 True，缺失或空的类别目录会被跳过并打印警告，
                            而不是抛出异常（默认 True，便于处理 chongqing_c
                            这类 normal 为空的中心）
    """

    def __init__(
        self,
        root_dir: str,
        transform=None,
        return_path: bool = False,
        allow_empty_class: bool = True,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.return_path = return_path
        self.allow_empty_class = allow_empty_class
        self.samples: List[Tuple[str, int]] = []
        self._scan()

    def _scan(self):
        for cls_name, label in CLASS_MAP.items():
            cls_dir = os.path.join(self.root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                if self.allow_empty_class:
                    print(
                        f"[Dataset Warning] 类别目录不存在，已跳过：{cls_dir}"
                    )
                    continue
                raise FileNotFoundError(
                    f"未找到类别目录：{cls_dir}，请检查数据路径。"
                )

            files = [
                f for f in sorted(os.listdir(cls_dir))
                if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
            ]
            if not files:
                if self.allow_empty_class:
                    print(
                        f"[Dataset Warning] 类别目录为空，已跳过：{cls_dir}"
                    )
                    continue
                raise RuntimeError(f"类别目录为空：{cls_dir}")

            for fname in files:
                self.samples.append((os.path.join(cls_dir, fname), label))

        if not self.samples:
            raise RuntimeError(
                f"数据目录 {self.root_dir} 中未找到任何图像文件。"
            )

    # -------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.return_path:
            return image, label, path
        return image, label

    # -------------------------------------------------------------- #

    def class_counts(self) -> dict:
        counts = {0: 0, 1: 0}
        for _, label in self.samples:
            counts[label] += 1
        return counts


# ----------------------------------------------------------------------- #
# 多中心数据集（image, class_label, domain_label）                          #
# ----------------------------------------------------------------------- #

class MultiCenterDataset(Dataset):
    """
    多中心合并数据集，用于训练域不变特征提取。

    每条样本携带：
        image         : 经 transform 处理的图像
        class_label   : cancer=1 / normal=0
        domain_label  : 中心索引（与 center_dirs 中的位置对应，0,1,2,...）

    目录结构：
        center_dir_i/
            cancer/
            normal/

    Args:
        center_dirs       : 各中心数据目录的列表（顺序决定 domain_label）
        transform         : torchvision transforms
        allow_empty_class : 缺失/为空的类别目录是否允许（默认 True）
        return_path       : 若为 True，__getitem__ 额外返回文件路径
    """

    def __init__(
        self,
        center_dirs: Sequence[str],
        transform=None,
        allow_empty_class: bool = True,
        return_path: bool = False,
    ):
        self.center_dirs = list(center_dirs)
        self.transform = transform
        self.allow_empty_class = allow_empty_class
        self.return_path = return_path

        # 每条样本: (path, class_label, domain_label)
        self.samples: List[Tuple[str, int, int]] = []
        self._scan()

    def _scan(self):
        if not self.center_dirs:
            raise ValueError("center_dirs 为空，至少需要一个中心目录。")

        for domain_idx, center_dir in enumerate(self.center_dirs):
            if not os.path.isdir(center_dir):
                if self.allow_empty_class:
                    print(
                        f"[Dataset Warning] 中心目录不存在，已跳过："
                        f"{center_dir} (domain={domain_idx})"
                    )
                    continue
                raise FileNotFoundError(
                    f"未找到中心目录：{center_dir}"
                )

            n_added = 0
            for cls_name, cls_label in CLASS_MAP.items():
                cls_dir = os.path.join(center_dir, cls_name)
                if not os.path.isdir(cls_dir):
                    if self.allow_empty_class:
                        print(
                            f"[Dataset Warning] 类别目录不存在，已跳过："
                            f"{cls_dir} (domain={domain_idx})"
                        )
                        continue
                    raise FileNotFoundError(
                        f"未找到类别目录：{cls_dir}"
                    )

                files = [
                    f for f in sorted(os.listdir(cls_dir))
                    if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
                ]
                if not files:
                    if self.allow_empty_class:
                        print(
                            f"[Dataset Warning] 类别目录为空，已跳过："
                            f"{cls_dir} (domain={domain_idx})"
                        )
                        continue
                    raise RuntimeError(f"类别目录为空：{cls_dir}")

                for fname in files:
                    self.samples.append(
                        (os.path.join(cls_dir, fname), cls_label, domain_idx)
                    )
                    n_added += 1

            print(
                f"[Dataset] 中心 {domain_idx} ({os.path.basename(center_dir)}): "
                f"加载 {n_added} 个样本"
            )

        if not self.samples:
            raise RuntimeError(
                f"未在任何中心目录中找到图像样本：{self.center_dirs}"
            )

    # -------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, cls_label, domain_label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.return_path:
            return image, cls_label, domain_label, path
        return image, cls_label, domain_label

    # -------------------------------------------------------------- #

    def class_counts(self) -> dict:
        counts: dict = defaultdict(int)
        for _, cls_label, _ in self.samples:
            counts[cls_label] += 1
        return dict(counts)

    def domain_counts(self) -> dict:
        counts: dict = defaultdict(int)
        for _, _, domain_label in self.samples:
            counts[domain_label] += 1
        return dict(counts)

    def class_domain_counts(self) -> dict:
        """统计每个 (domain, class) 组合的样本数。"""
        counts: dict = defaultdict(int)
        for _, cls_label, domain_label in self.samples:
            counts[(domain_label, cls_label)] += 1
        return dict(counts)


# ----------------------------------------------------------------------- #
# Transforms                                                               #
# ----------------------------------------------------------------------- #

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(image_size: Tuple[int, int], train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


# ----------------------------------------------------------------------- #
# DataLoader 构建函数                                                       #
# ----------------------------------------------------------------------- #

def build_dataloaders(
    center_dirs: Optional[Sequence[str]] = None,
    target_dir: Optional[str] = None,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 4,
    seed: int = 42,
    *,
    source_dir: Optional[str] = None,    # 向后兼容：等价于 center_dirs=[source_dir]
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    构建训练 / 验证 / 测试 三个 DataLoader（多中心版本）。

    训练与验证集来自多中心源域（MultiCenterDataset），每个 batch 返回
    (image, class_label, domain_label)；测试集来自目标域（MedicalImageDataset），
    每个 batch 返回 (image, class_label)，仅供跨中心评估使用。

    Args:
        center_dirs : 多中心源域目录列表；顺序决定 domain_label
        target_dir  : 目标域目录（仅用于测试评估）；可选
        source_dir  : 旧版接口，单一源域目录；若设置则等价于 center_dirs=[source_dir]
                      （此时无法做有意义的域对抗训练）

    Returns:
        src_train_loader : 多中心训练集（含数据增强），yield (img, cls, dom)
        src_val_loader   : 多中心验证集（无增强），   yield (img, cls, dom)
        tgt_loader       : 目标域评估集（无增强），  yield (img, cls)
                           若 target_dir 未提供则返回 None
    """
    # 兼容旧版 source_dir
    if center_dirs is None:
        if source_dir is None:
            raise ValueError("必须提供 center_dirs 或 source_dir 之一。")
        center_dirs = [source_dir]

    generator = torch.Generator().manual_seed(seed)

    # --- 多中心源域：先扫描全部样本，再按 (domain, class) 分层做 train/val 切分 ---
    src_full = MultiCenterDataset(
        center_dirs=center_dirs,
        transform=None,
        allow_empty_class=True,
    )

    # 按 (domain, class) 分组，每组内独立切分，保证 train/val 同分布
    groups: dict = defaultdict(list)
    for i, (_, cls_label, dom_label) in enumerate(src_full.samples):
        groups[(dom_label, cls_label)].append(i)

    train_idx: List[int] = []
    val_idx:   List[int] = []
    for key, indices in groups.items():
        perm = torch.randperm(len(indices), generator=generator).tolist()
        # 每组至少留 1 个给训练，避免某 (domain, class) 完全进入 val
        n_val = int(len(indices) * val_split)
        n_val = min(n_val, max(0, len(indices) - 1))
        for k, j in enumerate(perm):
            if k < n_val:
                val_idx.append(indices[j])
            else:
                train_idx.append(indices[j])

    src_train_ds = _MultiCenterIndexedSubset(
        src_full, train_idx, get_transforms(image_size, train=True),
    )
    src_val_ds = _MultiCenterIndexedSubset(
        src_full, val_idx, get_transforms(image_size, train=False),
    )

    print(
        f"[DataLoader] 训练集 {len(src_train_ds)} / 验证集 {len(src_val_ds)}"
    )
    print(f"[DataLoader] 域分布(全部源域): {src_full.domain_counts()}")
    print(f"[DataLoader] 类分布(全部源域): {src_full.class_counts()}")

    src_train_loader = DataLoader(
        src_train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    src_val_loader = DataLoader(
        src_val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # --- 目标域（仅用于评估）---
    tgt_loader: Optional[DataLoader] = None
    if target_dir is not None and os.path.isdir(target_dir):
        try:
            tgt_ds = MedicalImageDataset(
                target_dir,
                transform=get_transforms(image_size, train=False),
                allow_empty_class=True,
            )
            tgt_loader = DataLoader(
                tgt_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
        except RuntimeError as e:
            print(f"[DataLoader Warning] 目标域加载失败：{e}")
            tgt_loader = None
    else:
        if target_dir:
            print(
                f"[DataLoader Warning] target_dir 不存在或未提供："
                f"{target_dir}，跳过目标域 loader。"
            )

    return src_train_loader, src_val_loader, tgt_loader


# ----------------------------------------------------------------------- #
# 辅助：带索引和独立 transform 的多中心子集包装器                            #
# ----------------------------------------------------------------------- #

class _MultiCenterIndexedSubset(Dataset):
    """
    根据 indices 从 MultiCenterDataset 中抽取子集，并应用独立的 transform。
    保持 (image, class_label, domain_label) 的输出格式。
    """

    def __init__(
        self,
        base: MultiCenterDataset,
        indices: List[int],
        transform,
    ):
        self.base      = base
        self.indices   = indices
        self.transform = transform
        # 暴露 samples 以便 utils.compute_class_weights 等下游访问
        self.samples = [base.samples[i] for i in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        path, cls_label, domain_label = self.base.samples[self.indices[idx]]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, cls_label, domain_label

    def class_counts(self) -> dict:
        counts: dict = defaultdict(int)
        for _, cls_label, _ in self.samples:
            counts[cls_label] += 1
        return dict(counts)

    def domain_counts(self) -> dict:
        counts: dict = defaultdict(int)
        for _, _, domain_label in self.samples:
            counts[domain_label] += 1
        return dict(counts)


# ----------------------------------------------------------------------- #
# 旧版 _IndexedSubset（保留以保持向后兼容；当前流程不再使用）                #
# ----------------------------------------------------------------------- #

class _IndexedSubset(Dataset):
    """
    旧版：根据 indices 从 base dataset 中抽取子集，并应用独立的 transform。
    仅返回 (image, class_label)。保留以兼容可能直接 import 该类的旧脚本。
    """

    def __init__(self, base: MedicalImageDataset, indices: List[int], transform):
        self.base      = base
        self.indices   = indices
        self.transform = transform
        self.samples = [base.samples[i] for i in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        path, label = self.base.samples[self.indices[idx]]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label
