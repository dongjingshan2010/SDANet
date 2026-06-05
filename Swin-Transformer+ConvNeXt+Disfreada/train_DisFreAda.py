"""
训练脚本（DisFreAda 版本 - 频域分布自适应分类）

数据假设：
    cfg.center_dirs 列出多个中心目录（例如 chongqing_c, jincheng_nc），
    每条样本携带 "类别标签"。

模型：DisFreAda（Distribution-driven Multi-frequency Adaptive Network）
    输入图像 → FFT 频段分解（低/中/高）→ 共享 CNN 提取特征
    → 分布统计编码 → 自适应融合权重 → 分类头 → 交叉熵损失

用法：
    python train_DisFreAda.py
    python train_DisFreAda.py --epochs 50 --batch_size 16 --lr 1e-4
    python train_DisFreAda.py --feat_dim 128 --band_radii 0.2 0.55
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from config import Config
from datasets import build_dataloaders
from losses import ClassificationLoss
from models import DisFreAdaClassifier
from utils import (
    AverageMeter,
    EarlyStopping,
    MetricTracker,
    compute_class_weights,
    load_checkpoint,
    plot_training_curves,
    save_checkpoint,
    set_seed,
)


# ----------------------------------------------------------------------- #
# 命令行参数                                                                #
# ----------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="DisFreAda 训练脚本")
    p.add_argument("--epochs",       type=int,   default=None)
    p.add_argument("--batch_size",   type=int,   default=None)
    p.add_argument("--lr",           type=float, default=None)
    p.add_argument("--feat_dim",     type=int,   default=1024,
                   help="每个频段 CNN 的特征维度（默认 256）")
    p.add_argument("--hidden_dim",   type=int,   default=None,
                   help="分类头隐层维度（默认沿用 cfg.hidden_dim）")
    p.add_argument("--band_radii",   type=float, nargs=2, default=[0.25, 0.6],
                   metavar=("R1", "R2"),
                   help="频段分割半径阈值，归一化至 [0,1]（默认 0.25 0.6）")
    p.add_argument("--resume",       type=str,   default=None,
                   help="从检查点路径恢复训练")
    return p.parse_args()


# ----------------------------------------------------------------------- #
# 单轮训练                                                                  #
# ----------------------------------------------------------------------- #

def train_one_epoch(
    model: DisFreAdaClassifier,
    train_loader,
    criterion: ClassificationLoss,
    optimizer,
    device: str,
) -> dict:
    model.train()

    loss_meter = AverageMeter()
    all_preds, all_labels, all_probs = [], [], []

    for batch in train_loader:
        if len(batch) == 3:
            imgs, cls_labels, _ = batch
        else:
            imgs, cls_labels = batch

        imgs       = imgs.to(device)
        cls_labels = cls_labels.to(device)

        logits = model(imgs)
        loss, _ = criterion(logits, cls_labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        loss_meter.update(loss.item(), imgs.size(0))

        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu()
        preds = logits.argmax(dim=1).detach().cpu()
        all_probs.extend(probs.numpy())
        all_preds.extend(preds.numpy())
        all_labels.extend(cls_labels.detach().cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return {
        "train_loss": loss_meter.avg,
        "train_acc":  acc,
        "train_auc":  auc,
        "train_f1":   f1,
    }


# ----------------------------------------------------------------------- #
# 验证                                                                      #
# ----------------------------------------------------------------------- #

@torch.no_grad()
def validate(model: DisFreAdaClassifier, val_loader, device: str) -> dict:
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    loss_meter = AverageMeter()
    criterion  = nn.CrossEntropyLoss()

    for batch in val_loader:
        if len(batch) == 3:
            imgs, labels, _ = batch
        else:
            imgs, labels = batch

        imgs   = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss_meter.update(loss.item(), imgs.size(0))

        probs = torch.softmax(logits, dim=1)[:, 1].cpu()
        preds = logits.argmax(dim=1).cpu()
        all_probs.extend(probs.numpy())
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return {
        "val_loss": loss_meter.avg,
        "val_acc":  acc,
        "val_auc":  auc,
        "val_f1":   f1,
    }


# ----------------------------------------------------------------------- #
# 主训练流程                                                                #
# ----------------------------------------------------------------------- #

def main():
    args = parse_args()
    cfg  = Config()

    if args.epochs     is not None: cfg.num_epochs    = args.epochs
    if args.batch_size is not None: cfg.batch_size    = args.batch_size
    if args.lr         is not None: cfg.learning_rate = args.lr

    feat_dim    = args.feat_dim
    hidden_dim  = args.hidden_dim or cfg.hidden_dim
    band_radii  = tuple(args.band_radii)

    # DisFreAda 专属输出目录
    disfreada_output_dir = os.path.join(
        os.path.dirname(cfg.output_dir), "outputs_disfreada"
    )
    os.makedirs(os.path.join(disfreada_output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(disfreada_output_dir, "logs"),        exist_ok=True)

    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")
    print(f"DisFreAda | feat_dim={feat_dim}  hidden_dim={hidden_dim}  "
          f"band_radii={band_radii}")

    # ------------------------------------------------------------------ #
    # 数据                                                                 #
    # ------------------------------------------------------------------ #
    print("\n>>> 加载数据集...")
    print(f"  中心列表: {cfg.center_names}")

    src_train_loader, src_val_loader, tgt_loader = build_dataloaders(
        center_dirs = cfg.center_dirs,
        target_dir  = cfg.target_dir,
        image_size  = cfg.image_size,
        batch_size  = cfg.batch_size,
        val_split   = cfg.val_split,
        num_workers = cfg.num_workers,
        seed        = cfg.seed,
    )
    print(f"  训练: {len(src_train_loader.dataset)} 样本")
    print(f"  验证: {len(src_val_loader.dataset)} 样本")
    if tgt_loader is not None:
        print(f"  目标域(仅供测试参考): {len(tgt_loader.dataset)} 样本")

    # ------------------------------------------------------------------ #
    # 类别权重                                                             #
    # ------------------------------------------------------------------ #
    class_weights = compute_class_weights(
        src_train_loader.dataset, num_classes=cfg.num_classes, device=str(device)
    )
    print(f"  类别权重: {class_weights.tolist()}")

    # ------------------------------------------------------------------ #
    # 模型                                                                 #
    # ------------------------------------------------------------------ #
    print("\n>>> 构建 DisFreAda 模型...")
    model = DisFreAdaClassifier(
        num_classes = cfg.num_classes,
        feat_dim    = feat_dim,
        hidden_dim  = hidden_dim,
        band_radii  = band_radii,
        image_size  = cfg.image_size,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  总参数量: {total_params:.2f}M")

    # ------------------------------------------------------------------ #
    # 优化器 & 调度器                                                       #
    # ------------------------------------------------------------------ #
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs - cfg.warmup_epochs, eta_min=1e-6
    )

    # ------------------------------------------------------------------ #
    # 损失函数                                                             #
    # ------------------------------------------------------------------ #
    criterion = ClassificationLoss(class_weights=class_weights)

    # ------------------------------------------------------------------ #
    # （可选）恢复训练                                                      #
    # ------------------------------------------------------------------ #
    start_epoch = 0
    best_metric = 0.0
    tracker     = MetricTracker()

    if args.resume and os.path.isfile(args.resume):
        prev_metrics = load_checkpoint(args.resume, model, optimizer, str(device))
        start_epoch  = prev_metrics.get("epoch", 0) + 1
        best_metric  = prev_metrics.get(cfg.save_best_metric, 0.0)

    # ------------------------------------------------------------------ #
    # 早停                                                                 #
    # ------------------------------------------------------------------ #
    early_stopper = EarlyStopping(patience=15, higher_is_better=True)

    # ------------------------------------------------------------------ #
    # 训练循环                                                             #
    # ------------------------------------------------------------------ #
    print("\n>>> 开始训练...\n")
    for epoch in range(start_epoch, cfg.num_epochs):
        t0 = time.time()

        # Warmup：线性增加学习率
        if epoch < cfg.warmup_epochs:
            warmup_factor = (epoch + 1) / cfg.warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.learning_rate * warmup_factor

        train_metrics = train_one_epoch(
            model, src_train_loader, criterion, optimizer, str(device),
        )
        val_metrics = validate(model, src_val_loader, str(device))

        if epoch >= cfg.warmup_epochs:
            scheduler.step()

        epoch_metrics = {**train_metrics, **val_metrics, "epoch": epoch + 1}
        tracker.update(epoch_metrics)

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch+1:03d}/{cfg.num_epochs}] "
            f"({elapsed:.1f}s) | "
            f"Loss: {train_metrics['train_loss']:.4f} | "
            f"train_acc={train_metrics['train_acc']:.4f} "
            f"train_auc={train_metrics['train_auc']:.4f} | "
            f"val_acc={val_metrics['val_acc']:.4f} "
            f"val_auc={val_metrics['val_auc']:.4f} "
            f"val_f1={val_metrics['val_f1']:.4f}"
        )

        cur_metric = val_metrics[cfg.save_best_metric]
        if cur_metric > best_metric:
            best_metric = cur_metric
            save_checkpoint(
                model, optimizer, epoch, epoch_metrics,
                os.path.join(disfreada_output_dir, "checkpoints", "best_model.pth"),
            )

        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, epoch_metrics,
                os.path.join(disfreada_output_dir, "checkpoints", f"epoch_{epoch+1:03d}.pth"),
            )

        if early_stopper.step(cur_metric):
            print(f"\n[早停] {cfg.save_best_metric} 连续 {early_stopper.patience} 轮无改善，停止训练。")
            break

    # ------------------------------------------------------------------ #
    # 训练结束                                                             #
    # ------------------------------------------------------------------ #
    plot_training_curves(tracker, os.path.join(disfreada_output_dir, "logs"))
    print(f"\n训练完成！最佳 {cfg.save_best_metric} = {best_metric:.4f}")
    print(f"最佳模型保存于: {os.path.join(disfreada_output_dir, 'checkpoints', 'best_model.pth')}")


if __name__ == "__main__":
    main()
