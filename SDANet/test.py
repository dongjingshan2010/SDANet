"""
测试脚本（跨中心评估）

在目标域（sw_2zibo）上评估已训练的最佳模型，输出完整的分类报告、
混淆矩阵、ROC 曲线，以及错误预测样本列表。

用法：
    python test.py
    python test.py --ckpt outputs/checkpoints/best_model.pth
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
)
from tqdm import tqdm

from config import Config
from datasets import MedicalImageDataset, get_transforms
from models import FreqDANN
from torch.utils.data import DataLoader
from utils import load_checkpoint, set_seed


# ----------------------------------------------------------------------- #
# 命令行参数                                                                #
# ----------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="FreqDANN 测试脚本")
    p.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="检查点路径（默认使用 outputs/checkpoints/best_model.pth）",
    )
    p.add_argument("--batch_size", type=int, default=None)
    return p.parse_args()


# ----------------------------------------------------------------------- #
# 推理                                                                      #
# ----------------------------------------------------------------------- #

@torch.no_grad()
def run_inference(model: FreqDANN, loader: DataLoader, device: str):
    """
    对目标域数据集进行完整推理。

    Returns:
        all_labels : list[int]
        all_preds  : list[int]
        all_probs  : list[float]  （cancer 类的概率）
        all_paths  : list[str]
    """
    model.eval()

    all_labels, all_preds, all_probs, all_paths = [], [], [], []

    for batch in tqdm(loader, desc="推理中", leave=False):
        if len(batch) == 3:
            imgs, labels, paths = batch
            all_paths.extend(paths)
        else:
            imgs, labels = batch

        imgs   = imgs.to(device)
        logits = model.predict(imgs)
        probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    return all_labels, all_preds, all_probs, all_paths


# ----------------------------------------------------------------------- #
# 可视化                                                                    #
# ----------------------------------------------------------------------- #

def plot_confusion_matrix(cm: np.ndarray, save_path: str):
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal", "Cancer"],
        yticklabels=["Normal", "Cancer"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Target Domain)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] 混淆矩阵保存至 {save_path}")


def plot_roc_curve(labels: list, probs: list, save_path: str):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Target Domain)")
    ax.legend(loc="lower right")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] ROC 曲线保存至 {save_path}")


# ----------------------------------------------------------------------- #
# 主测试流程                                                                #
# ----------------------------------------------------------------------- #

def main():
    args = parse_args()
    cfg  = Config()

    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    ckpt_path = args.ckpt or os.path.join(
        cfg.output_dir, "checkpoints", "best_model.pth"
    )
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"未找到检查点文件：{ckpt_path}\n"
            "请先运行 train.py 完成训练，或通过 --ckpt 指定路径。"
        )

    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    # ------------------------------------------------------------------ #
    # 目标域数据集                                                          #
    # ------------------------------------------------------------------ #
    tgt_ds = MedicalImageDataset(
        root_dir    = cfg.target_dir,
        transform   = get_transforms(cfg.image_size, train=False),
        return_path = True,
    )
    tgt_loader = DataLoader(
        tgt_ds,
        batch_size  = cfg.batch_size,
        shuffle     = False,
        num_workers = cfg.num_workers,
        pin_memory  = True,
    )
    counts = tgt_ds.class_counts()
    print(f"\n目标域数据集: Normal={counts[0]}, Cancer={counts[1]}")

    # ------------------------------------------------------------------ #
    # 模型加载                                                             #
    # ------------------------------------------------------------------ #
    print("\n>>> 加载模型...")
    model = FreqDANN(
        phase_encoder_name = cfg.phase_encoder_name,
        amp_encoder_name   = cfg.amp_encoder_name,
        feature_dim        = cfg.feature_dim,
        disc_hidden_dim    = cfg.disc_hidden_dim,
        num_classes        = cfg.num_classes,
        pretrained         = False,   # 测试时不需要重新加载 ImageNet 权重
    ).to(device)
    load_checkpoint(ckpt_path, model, device=str(device))

    # ------------------------------------------------------------------ #
    # 推理                                                                 #
    # ------------------------------------------------------------------ #
    print("\n>>> 开始推理...")
    all_labels, all_preds, all_probs, all_paths = run_inference(
        model, tgt_loader, str(device)
    )

    # ------------------------------------------------------------------ #
    # 评估指标                                                             #
    # ------------------------------------------------------------------ #
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(all_labels, all_preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
    else:
        sensitivity = specificity = float("nan")

    print("\n" + "=" * 55)
    print("  跨中心测试结果（目标域：sw_2zibo）")
    print("=" * 55)
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  AUC         : {auc:.4f}")
    print(f"  F1 Score    : {f1:.4f}")
    print(f"  Sensitivity : {sensitivity:.4f}  (Recall for Cancer)")
    print(f"  Specificity : {specificity:.4f}  (Recall for Normal)")
    print("=" * 55)
    print("\n详细分类报告：")
    print(classification_report(
        all_labels, all_preds,
        target_names=["Normal", "Cancer"],
        digits=4,
    ))

    # ------------------------------------------------------------------ #
    # 保存结果                                                             #
    # ------------------------------------------------------------------ #
    result_dir = os.path.join(cfg.output_dir, "test_results")
    os.makedirs(result_dir, exist_ok=True)

    # 混淆矩阵图
    plot_confusion_matrix(cm, os.path.join(result_dir, "confusion_matrix.png"))

    # ROC 曲线图
    if not np.isnan(auc):
        plot_roc_curve(all_labels, all_probs, os.path.join(result_dir, "roc_curve.png"))

    # 预测结果 CSV
    label_name = {0: "Normal", 1: "Cancer"}
    result_df = pd.DataFrame({
        "path":       all_paths if all_paths else ["N/A"] * len(all_labels),
        "true_label": [label_name[l] for l in all_labels],
        "pred_label": [label_name[p] for p in all_preds],
        "prob_cancer": [f"{p:.4f}" for p in all_probs],
        "correct":    [str(t == p) for t, p in zip(all_labels, all_preds)],
    })
    csv_path = os.path.join(result_dir, "predictions.csv")
    result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n预测结果已保存至 {csv_path}")

    # 错误预测汇总
    error_df = result_df[result_df["correct"] == "False"]
    print(f"\n错误预测数量: {len(error_df)} / {len(result_df)}")

    # 指标摘要
    summary = {
        "accuracy":    acc,
        "auc":         auc,
        "f1":          f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "n_correct":   int(acc * len(all_labels)),
        "n_total":     len(all_labels),
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(
        os.path.join(result_dir, "summary.csv"),
        index=False, encoding="utf-8-sig",
    )
    print(f"指标摘要已保存至 {os.path.join(result_dir, 'summary.csv')}")


if __name__ == "__main__":
    main()
