"""
训练脚本（多中心域不变特征学习）

数据假设：
    cfg.center_dirs 列出多个中心目录（例如 chongqing_c, jincheng_nc），
    每条样本同时携带 "类别标签" 与 "域标签（中心索引）"。

域无关特征提取：
    幅度分支 → GRL → DomainDiscriminator(num_domains=中心数)
    通过预测样本所属中心的对抗目标，迫使幅度特征与中心信息脱耦。

用法：
    python train.py
    python train.py --epochs 50 --batch_size 16 --lambda_adv 0.5
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from tqdm import tqdm

from config import Config
from datasets import build_dataloaders
from losses import TotalLoss
from models import FreqDANN
from models.grl import grl_lambda_schedule
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
    p = argparse.ArgumentParser(description="FreqDANN 训练脚本（多中心）")
    p.add_argument("--epochs",      type=int,   default=None)
    p.add_argument("--batch_size",  type=int,   default=None)
    p.add_argument("--lr",          type=float, default=None)
    p.add_argument("--lambda_adv",  type=float, default=None)
    p.add_argument("--lambda_cons", type=float, default=None)
    p.add_argument("--resume",      type=str,   default=None,
                   help="从检查点路径恢复训练")
    p.add_argument("--no_pretrain", action="store_true",
                   help="不加载预训练权重")
    return p.parse_args()


# ----------------------------------------------------------------------- #
# 单轮训练                                                                  #
# ----------------------------------------------------------------------- #

def train_one_epoch(
    model: FreqDANN,
    src_loader,
    criterion: TotalLoss,
    optimizer,
    device: str,
    global_iter: int,
    max_iter: int,
    cfg: Config,
) -> tuple[dict, int]:
    """
    单轮训练。源域 loader 同时提供 (image, class_label, domain_label)，
    域对抗损失基于真实的中心索引（而非合成的 src/tgt 标签）。

    Returns:
        metrics      : 本轮训练的指标字典
        global_iter  : 更新后的全局迭代数
    """
    model.train()

    meters = {k: AverageMeter() for k in
              ["loss_total", "loss_cls", "loss_adv", "loss_cons"]}
    all_preds, all_labels, all_probs = [], [], []
    all_dom_preds, all_dom_labels = [], []
    lam = 0.0

    for batch in src_loader:
        # 兼容三元组 (img, cls, dom) 与二元组 (img, cls)
        if len(batch) == 3:
            imgs, cls_labels, dom_labels = batch
        else:
            imgs, cls_labels = batch
            dom_labels = torch.zeros(imgs.size(0), dtype=torch.long)

        imgs       = imgs.to(device)
        cls_labels = cls_labels.to(device)
        dom_labels = dom_labels.to(device).long()

        # 更新 GRL lambda
        lam = grl_lambda_schedule(global_iter, max_iter, cfg.grl_gamma)
        model.set_grl_alpha(lam)

        # ---- 前向：单批次同时承担 (1) 分类  (2) 域对抗 ----
        cls_logits, dom_logits, amp_feat, phase_feat = model(imgs)

        # ---- 计算总损失（使用真实域标签）----
        total_loss, loss_details = criterion(
            cls_logits    = cls_logits,
            labels        = cls_labels,
            domain_logits = dom_logits,
            domain_labels = dom_labels,
            amp_feat      = amp_feat,
            phase_feat    = phase_feat,
        )

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # 记录损失
        B = imgs.size(0)
        for k, v in loss_details.items():
            meters[k].update(v, B)

        # 分类预测（用于 epoch 指标）
        probs = torch.softmax(cls_logits, dim=1)[:, 1].detach().cpu()
        preds = cls_logits.argmax(dim=1).detach().cpu()
        all_probs.extend(probs.numpy())
        all_preds.extend(preds.numpy())
        all_labels.extend(cls_labels.detach().cpu().numpy())

        # 域判别准确率（仅作监控；越高表示域信息越未被剥离）
        dom_preds = dom_logits.argmax(dim=1).detach().cpu()
        all_dom_preds.extend(dom_preds.numpy())
        all_dom_labels.extend(dom_labels.detach().cpu().numpy())

        global_iter += 1

    # 计算分类指标
    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    # 域判别准确率（监控用：理想情况下，对抗训练后此值会下降到 ~ 1/num_domains）
    try:
        dom_acc = accuracy_score(all_dom_labels, all_dom_preds)
    except ValueError:
        dom_acc = 0.0

    metrics = {
        "train_loss":   meters["loss_total"].avg,
        "train_cls":    meters["loss_cls"].avg,
        "train_adv":    meters["loss_adv"].avg,
        "train_cons":   meters["loss_cons"].avg,
        "train_acc":    acc,
        "train_auc":    auc,
        "train_f1":     f1,
        "train_dom_acc": dom_acc,
        "grl_lambda":   lam,
    }
    return metrics, global_iter


# ----------------------------------------------------------------------- #
# 验证                                                                      #
# ----------------------------------------------------------------------- #

@torch.no_grad()
def validate(model: FreqDANN, val_loader, device: str) -> dict:
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    loss_meter = AverageMeter()
    criterion = nn.CrossEntropyLoss()

    for batch in val_loader:
        # 兼容三元组与二元组
        if len(batch) == 3:
            imgs, labels, _ = batch
        else:
            imgs, labels = batch

        imgs   = imgs.to(device)
        labels = labels.to(device)

        logits = model.predict(imgs)
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
    if args.epochs      is not None: cfg.num_epochs   = args.epochs
    if args.batch_size  is not None: cfg.batch_size   = args.batch_size
    if args.lr          is not None: cfg.lr_encoder   = args.lr
    if args.lambda_adv  is not None: cfg.lambda_adv   = args.lambda_adv
    if args.lambda_cons is not None: cfg.lambda_cons  = args.lambda_cons
    if args.no_pretrain:             cfg.pretrained    = False

    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    # ------------------------------------------------------------------ #
    # 数据                                                                 #
    # ------------------------------------------------------------------ #
    print("\n>>> 加载多中心数据集...")
    print(f"  中心列表: {cfg.center_names} → {cfg.center_dirs}")

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
    else:
        print("  目标域: 未配置或不可用（不影响训练）")

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
    print("\n>>> 构建模型...")
    model = FreqDANN(
        phase_encoder_name = cfg.phase_encoder_name,
        amp_encoder_name   = cfg.amp_encoder_name,
        feature_dim        = cfg.feature_dim,
        disc_hidden_dim    = cfg.disc_hidden_dim,
        num_classes        = cfg.num_classes,
        num_domains        = cfg.num_domains,
        pretrained         = cfg.pretrained,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  总参数量: {total_params:.2f}M  (域数={cfg.num_domains})")

    # ------------------------------------------------------------------ #
    # 优化器（区分编码器与其他模块的学习率）                                 #
    # ------------------------------------------------------------------ #
    encoder_params = list(model.phase_encoder.parameters()) + \
                     list(model.amp_encoder.parameters())
    other_params   = list(model.cls_head.parameters()) + \
                     list(model.domain_disc.parameters())

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": cfg.lr_encoder},
        {"params": other_params,   "lr": cfg.lr_head},
    ], weight_decay=cfg.weight_decay)

    # Cosine 退火调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs - cfg.warmup_epochs, eta_min=1e-6
    )

    # ------------------------------------------------------------------ #
    # 损失函数                                                             #
    # ------------------------------------------------------------------ #
    criterion = TotalLoss(
        lambda_adv   = cfg.lambda_adv,
        lambda_cons  = cfg.lambda_cons,
        class_weights = class_weights,
    )

    # ------------------------------------------------------------------ #
    # （可选）恢复训练                                                      #
    # ------------------------------------------------------------------ #
    start_epoch  = 0
    global_iter  = 0
    best_metric  = 0.0
    tracker      = MetricTracker()

    if args.resume and os.path.isfile(args.resume):
        prev_metrics = load_checkpoint(args.resume, model, optimizer, str(device))
        start_epoch  = prev_metrics.get("epoch", 0) + 1
        best_metric  = prev_metrics.get(cfg.save_best_metric, 0.0)

    # ------------------------------------------------------------------ #
    # 早停                                                                 #
    # ------------------------------------------------------------------ #
    early_stopper = EarlyStopping(patience=15, higher_is_better=True)

    # GRL 总迭代数
    max_iter = cfg.num_epochs * len(src_train_loader)

    # ------------------------------------------------------------------ #
    # 训练循环                                                             #
    # ------------------------------------------------------------------ #
    print("\n>>> 开始训练...\n")
    for epoch in range(start_epoch, cfg.num_epochs):
        t0 = time.time()

        # -- Warmup 阶段：线性增加编码器 lr --
        if epoch < cfg.warmup_epochs:
            warmup_factor = (epoch + 1) / cfg.warmup_epochs
            for pg in optimizer.param_groups[:1]:   # 只对编码器
                pg["lr"] = cfg.lr_encoder * warmup_factor

        train_metrics, global_iter = train_one_epoch(
            model, src_train_loader,
            criterion, optimizer, str(device),
            global_iter, max_iter, cfg,
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
            f"Loss: {train_metrics['train_loss']:.4f} "
            f"(cls={train_metrics['train_cls']:.4f} "
            f"adv={train_metrics['train_adv']:.4f} "
            f"cons={train_metrics['train_cons']:.4f}) | "
            f"dom_acc={train_metrics['train_dom_acc']:.3f} | "
            f"val_acc={val_metrics['val_acc']:.4f} "
            f"val_auc={val_metrics['val_auc']:.4f} "
            f"val_f1={val_metrics['val_f1']:.4f} | "
            f"λ_grl={train_metrics['grl_lambda']:.3f}"
        )

        # 保存最优模型
        cur_metric = val_metrics[cfg.save_best_metric]
        if cur_metric > best_metric:
            best_metric = cur_metric
            save_checkpoint(
                model, optimizer, epoch, epoch_metrics,
                os.path.join(cfg.output_dir, "checkpoints", "best_model.pth"),
            )

        # 每 10 个 epoch 保存一次普通检查点
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, epoch_metrics,
                os.path.join(cfg.output_dir, "checkpoints", f"epoch_{epoch+1:03d}.pth"),
            )

        # 早停检查
        if early_stopper.step(cur_metric):
            print(f"\n[早停] {cfg.save_best_metric} 连续 {early_stopper.patience} 轮无改善，停止训练。")
            break

    # ------------------------------------------------------------------ #
    # 训练结束                                                             #
    # ------------------------------------------------------------------ #
    plot_training_curves(tracker, os.path.join(cfg.output_dir, "logs"))
    print(f"\n训练完成！最佳 {cfg.save_best_metric} = {best_metric:.4f}")
    print(f"最佳模型保存于: {os.path.join(cfg.output_dir, 'checkpoints', 'best_model.pth')}")


if __name__ == "__main__":
    main()
