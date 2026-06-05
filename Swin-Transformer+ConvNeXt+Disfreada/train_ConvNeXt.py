"""
训练脚本（ConvNeXt 版本 - 仅分类）

数据假设：
    cfg.center_dirs 列出多个中心目录（例如 chongqing_c, jincheng_nc），
    每条样本携带 "类别标签"。

简单的监督学习：
    输入图像 → ConvNeXt → 分类头 → 交叉熵损失

用法：
    python train_ConvNeXt.py
    python train_ConvNeXt.py --epochs 50 --batch_size 16 --lr 1e-4
    python train_ConvNeXt.py --model_name convnext_small
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

from config import Config
from datasets import build_dataloaders
from losses import ClassificationLoss
from models import ConvNeXtClassifier
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
    p = argparse.ArgumentParser(description="ConvNeXt 分类器 训练脚本")
    p.add_argument("--epochs",      type=int,   default=None)
    p.add_argument("--batch_size",  type=int,   default=None)
    p.add_argument("--lr",          type=float, default=None)
    p.add_argument("--model_name",  type=str,   default='convnext_small',
                   help="ConvNeXt 模型名，如 convnext_tiny / convnext_small / convnext_base")
    p.add_argument("--resume",      type=str,   default=None,
                   help="从检查点路径恢复训练")
    p.add_argument("--no_pretrain", action="store_true",
                   help="不加载预训练权重")
    return p.parse_args()


# ----------------------------------------------------------------------- #
# 单轮训练                                                                  #
# ----------------------------------------------------------------------- #

def train_one_epoch(
    model: ConvNeXtClassifier,
    train_loader,
    criterion: ClassificationLoss,
    optimizer,
    device: str,
) -> dict:
    """
    单轮训练。

    Returns:
        metrics : 本轮训练的指标字典
    """
    model.train()

    loss_meter = AverageMeter()
    all_preds, all_labels, all_probs = [], [], []

    for batch in train_loader:
        # 兼容三元组 (img, cls, dom) 与二元组 (img, cls)
        if len(batch) == 3:
            imgs, cls_labels, _ = batch
        else:
            imgs, cls_labels = batch

        imgs       = imgs.to(device)
        cls_labels = cls_labels.to(device)

        # 前向
        logits = model(imgs)

        # 计算损失
        loss, _ = criterion(logits, cls_labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # 记录损失
        B = imgs.size(0)
        loss_meter.update(loss.item(), B)

        # 分类预测
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
def validate(model: ConvNeXtClassifier, val_loader, device: str) -> dict:
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

    # 命令行覆盖配置
    if args.epochs     is not None: cfg.num_epochs    = args.epochs
    if args.batch_size is not None: cfg.batch_size    = args.batch_size
    if args.lr         is not None: cfg.learning_rate = args.lr
    if args.no_pretrain:            cfg.pretrained    = False

    # ConvNeXt 专属：模型名与独立输出目录
    convnext_model_name = args.model_name or "convnext_tiny"
    convnext_output_dir = os.path.join(
        os.path.dirname(cfg.output_dir), "outputs_convnext"
    )
    os.makedirs(os.path.join(convnext_output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(convnext_output_dir, "logs"),        exist_ok=True)

    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")
    print(f"ConvNeXt 模型：{convnext_model_name}")

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
    # 类别权重（处理不平衡）                                                #
    # ------------------------------------------------------------------ #
    class_weights = compute_class_weights(
        src_train_loader.dataset, num_classes=cfg.num_classes, device=str(device)
    )
    print(f"  类别权重: {class_weights.tolist()}")

    # ------------------------------------------------------------------ #
    # 模型                                                                 #
    # ------------------------------------------------------------------ #
    print("\n>>> 构建 ConvNeXt 模型...")
    model = ConvNeXtClassifier(
        model_name  = convnext_model_name,
        num_classes = cfg.num_classes,
        hidden_dim  = cfg.hidden_dim,
        pretrained  = cfg.pretrained,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  总参数量: {total_params:.2f}M")

    # ------------------------------------------------------------------ #
    # 优化器                                                               #
    # ------------------------------------------------------------------ #
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # Cosine 退火调度器
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

        # Warmup 阶段：线性增加学习率
        if epoch < cfg.warmup_epochs:
            warmup_factor = (epoch + 1) / cfg.warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.learning_rate * warmup_factor

        train_metrics = train_one_epoch(
            model, src_train_loader, criterion, optimizer, str(device),
        )
        val_metrics = validate(model, src_val_loader, str(device))

        # Cosine 退火（warmup 结束后）
        if epoch >= cfg.warmup_epochs:
            scheduler.step()

        # 汇总
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

        # 保存最优模型
        cur_metric = val_metrics[cfg.save_best_metric]
        if cur_metric > best_metric:
            best_metric = cur_metric
            save_checkpoint(
                model, optimizer, epoch, epoch_metrics,
                os.path.join(convnext_output_dir, "checkpoints", "best_model.pth"),
            )

        # 每 10 个 epoch 保存一次普通检查点
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, epoch_metrics,
                os.path.join(convnext_output_dir, "checkpoints", f"epoch_{epoch+1:03d}.pth"),
            )

        # 早停检查
        if early_stopper.step(cur_metric):
            print(f"\n[早停] {cfg.save_best_metric} 连续 {early_stopper.patience} 轮无改善，停止训练。")
            break

    # ------------------------------------------------------------------ #
    # 训练结束                                                             #
    # ------------------------------------------------------------------ #
    plot_training_curves(tracker, os.path.join(convnext_output_dir, "logs"))
    print(f"\n训练完成！最佳 {cfg.save_best_metric} = {best_metric:.4f}")
    print(f"最佳模型保存于: {os.path.join(convnext_output_dir, 'checkpoints', 'best_model.pth')}")


if __name__ == "__main__":
    main()
