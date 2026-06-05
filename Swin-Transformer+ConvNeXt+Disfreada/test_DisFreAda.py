"""
测试脚本（DisFreAda 版本 - 跨中心评估）

在目标域（sw_2zibo）上评估已训练的 DisFreAda 最佳模型，输出完整的
分类报告、混淆矩阵、ROC 曲线，以及频段权重分析（可解释性）。

用法：
    python test_DisFreAda.py
    python test_DisFreAda.py --ckpt outputs_disfreada/checkpoints/best_model.pth
    python test_DisFreAda.py --feat_dim 128 --band_radii 0.2 0.55
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from datasets import MedicalImageDataset, get_transforms
from models import DisFreAdaClassifier
from utils import load_checkpoint, set_seed


# ----------------------------------------------------------------------- #
# 命令行参数                                                                #
# ----------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="DisFreAda 测试脚本")
    p.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="检查点路径（默认使用 outputs_disfreada/checkpoints/best_model.pth）",
    )
    p.add_argument("--batch_size",  type=int,   default=None)
    p.add_argument("--feat_dim",    type=int,   default=1024,
                   help="须与训练时一致（默认 256）")
    p.add_argument("--hidden_dim",  type=int,   default=None)
    p.add_argument("--band_radii",  type=float, nargs=2, default=[0.25, 0.6],
                   metavar=("R1", "R2"),
                   help="须与训练时一致（默认 0.25 0.6）")
    return p.parse_args()


# ----------------------------------------------------------------------- #
# 推理                                                                      #
# ----------------------------------------------------------------------- #

@torch.no_grad()
def run_inference(model: DisFreAdaClassifier, loader: DataLoader, device: str):
    """
    对目标域数据集进行完整推理，同时收集频段自适应权重。

    Returns:
        all_labels   : list[int]
        all_preds    : list[int]
        all_probs    : list[float]   （cancer 类的概率）
        all_paths    : list[str]
        all_weights  : np.ndarray    [N, 3]  (低/中/高频段权重)
    """
    model.eval()

    all_labels, all_preds, all_probs, all_paths = [], [], [], []
    all_weights = []

    for batch in tqdm(loader, desc="推理中", leave=False):
        if len(batch) == 3:
            imgs, labels, paths = batch
            all_paths.extend(paths)
        else:
            imgs, labels = batch

        imgs = imgs.to(device)

        logits  = model(imgs)
        weights = model.get_band_weights(imgs)   # [B, 3]

        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())
        all_weights.append(weights.cpu().numpy())

    all_weights = np.concatenate(all_weights, axis=0)   # [N, 3]
    return all_labels, all_preds, all_probs, all_paths, all_weights


# ----------------------------------------------------------------------- #
# 可视化                                                                    #
# ----------------------------------------------------------------------- #

def plot_confusion_matrix(cm: np.ndarray, save_path: str):
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Purples",
        xticklabels=["Normal", "Cancer"],
        yticklabels=["Normal", "Cancer"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix - DisFreAda (Target Domain)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] 混淆矩阵保存至 {save_path}")


def plot_roc_curve(labels: list, probs: list, save_path: str):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="mediumorchid", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - DisFreAda (Target Domain)")
    ax.legend(loc="lower right")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] ROC 曲线保存至 {save_path}")


def plot_band_weights(
    weights: np.ndarray,
    labels: list,
    save_path: str,
):
    """
    绘制三个频段自适应权重的分布（按类别分组箱线图）。

    Args:
        weights  : [N, 3]  (low, mid, high)
        labels   : [N]  真实标签
    """
    band_names = ["Low-freq", "Mid-freq", "High-freq"]
    label_names = {0: "Normal", 1: "Cancer"}
    colors = {"Normal": "#5B9BD5", "Cancer": "#ED7D31"}

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for i, (ax, name) in enumerate(zip(axes, band_names)):
        data_by_class = {
            label_names[c]: weights[np.array(labels) == c, i]
            for c in sorted(set(labels))
        }
        parts = ax.violinplot(
            list(data_by_class.values()),
            positions=range(len(data_by_class)),
            showmedians=True,
        )
        for pc, cls_name in zip(parts["bodies"], data_by_class):
            pc.set_facecolor(colors.get(cls_name, "gray"))
            pc.set_alpha(0.7)

        ax.set_xticks(range(len(data_by_class)))
        ax.set_xticklabels(list(data_by_class.keys()))
        ax.set_title(name)
        ax.set_ylabel("Adaptive Weight" if i == 0 else "")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("DisFreAda – Per-band Adaptive Weights by Class", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] 频段权重分布图保存至 {save_path}")


# ----------------------------------------------------------------------- #
# 主测试流程                                                                #
# ----------------------------------------------------------------------- #

def main():
    args = parse_args()
    cfg  = Config()

    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    feat_dim   = args.feat_dim
    hidden_dim = args.hidden_dim or cfg.hidden_dim
    band_radii = tuple(args.band_radii)

    disfreada_output_dir = os.path.join(
        os.path.dirname(cfg.output_dir), "outputs_disfreada"
    )
    ckpt_path = args.ckpt or os.path.join(
        disfreada_output_dir, "checkpoints", "best_model.pth"
    )
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"未找到检查点文件：{ckpt_path}\n"
            "请先运行 train_DisFreAda.py 完成训练，或通过 --ckpt 指定路径。"
        )

    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")
    print(f"DisFreAda | feat_dim={feat_dim}  hidden_dim={hidden_dim}  "
          f"band_radii={band_radii}")

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
    print("\n>>> 加载 DisFreAda 模型...")
    model = DisFreAdaClassifier(
        num_classes = cfg.num_classes,
        feat_dim    = feat_dim,
        hidden_dim  = hidden_dim,
        band_radii  = band_radii,
        image_size  = cfg.image_size,
    ).to(device)
    load_checkpoint(ckpt_path, model, device=str(device))

    # ------------------------------------------------------------------ #
    # 推理                                                                 #
    # ------------------------------------------------------------------ #
    print("\n>>> 开始推理...")
    all_labels, all_preds, all_probs, all_paths, all_weights = run_inference(
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

    # 频段权重统计
    mean_weights = all_weights.mean(axis=0)   # [3]

    print("\n" + "=" * 60)
    print("  DisFreAda 跨中心测试结果（目标域：sw_2zibo）")
    print("=" * 60)
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  AUC         : {auc:.4f}")
    print(f"  F1 Score    : {f1:.4f}")
    print(f"  Sensitivity : {sensitivity:.4f}  (Recall for Cancer)")
    print(f"  Specificity : {specificity:.4f}  (Recall for Normal)")
    print("-" * 60)
    print(f"  平均频段权重 | Low: {mean_weights[0]:.4f}  "
          f"Mid: {mean_weights[1]:.4f}  High: {mean_weights[2]:.4f}")
    print("=" * 60)
    print("\n详细分类报告：")
    print(classification_report(
        all_labels, all_preds,
        target_names=["Normal", "Cancer"],
        digits=4,
    ))

    # ------------------------------------------------------------------ #
    # 保存结果                                                             #
    # ------------------------------------------------------------------ #
    result_dir = os.path.join(disfreada_output_dir, "test_results")
    os.makedirs(result_dir, exist_ok=True)

    # 混淆矩阵图
    plot_confusion_matrix(cm, os.path.join(result_dir, "confusion_matrix.png"))

    # ROC 曲线图
    if not np.isnan(auc):
        plot_roc_curve(all_labels, all_probs, os.path.join(result_dir, "roc_curve.png"))

    # 频段权重分布图（可解释性）
    plot_band_weights(
        all_weights, all_labels,
        os.path.join(result_dir, "band_weights.png"),
    )

    # 预测结果 CSV
    label_name = {0: "Normal", 1: "Cancer"}
    result_df  = pd.DataFrame({
        "path":        all_paths if all_paths else ["N/A"] * len(all_labels),
        "true_label":  [label_name[l] for l in all_labels],
        "pred_label":  [label_name[p] for p in all_preds],
        "prob_cancer": [f"{p:.4f}" for p in all_probs],
        "w_low":       [f"{w:.4f}" for w in all_weights[:, 0]],
        "w_mid":       [f"{w:.4f}" for w in all_weights[:, 1]],
        "w_high":      [f"{w:.4f}" for w in all_weights[:, 2]],
        "correct":     [str(t == p) for t, p in zip(all_labels, all_preds)],
    })
    csv_path = os.path.join(result_dir, "predictions.csv")
    result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n预测结果（含频段权重）已保存至 {csv_path}")

    error_df = result_df[result_df["correct"] == "False"]
    print(f"\n错误预测数量: {len(error_df)} / {len(result_df)}")

    # 指标摘要
    summary = {
        "accuracy":       acc,
        "auc":            auc,
        "f1":             f1,
        "sensitivity":    sensitivity,
        "specificity":    specificity,
        "n_correct":      int(acc * len(all_labels)),
        "n_total":        len(all_labels),
        "w_low_mean":     float(mean_weights[0]),
        "w_mid_mean":     float(mean_weights[1]),
        "w_high_mean":    float(mean_weights[2]),
        "feat_dim":       feat_dim,
        "band_r1":        band_radii[0],
        "band_r2":        band_radii[1],
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(
        os.path.join(result_dir, "summary.csv"),
        index=False, encoding="utf-8-sig",
    )
    print(f"指标摘要已保存至 {os.path.join(result_dir, 'summary.csv')}")


if __name__ == "__main__":
    main()
