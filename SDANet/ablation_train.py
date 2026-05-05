"""
消融实验训练脚本

对 SDANet 的三个核心设计进行系统性消融：
  (1) 频域分解模块（use_frequency_decomp）
  (2) 域对抗训练（lambda_adv）
  (3) 跨流一致性损失（lambda_cons）
  (4) 流模式：仅相位流 / 仅幅度流 / 双流

运行方式：
    python ablation_train.py                    # 运行全部变体
    python ablation_train.py --variant full     # 仅运行指定变体
    python ablation_train.py --epochs 50        # 覆盖训练轮数

消融变体说明：
  full          : 完整 SDANet（基准）
  no_cons       : 移除一致性损失（λ_cons=0）
  no_adv        : 移除域对抗损失（λ_adv=0）
  no_freq       : 移除频域分解（原始图像送双流编码器）
  phase_only    : 仅使用相位流（无幅度流，无对抗训练）
  amp_only      : 仅使用幅度流（无相位流）
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, confusion_matrix,
)
from tqdm import tqdm

from config import Config
from datasets import build_dataloaders, MedicalImageDataset, get_transforms
from losses import TotalLoss
from models import FreqDANN
from models.grl import grl_lambda_schedule
from torch.utils.data import DataLoader
from utils import (
    AverageMeter, EarlyStopping, MetricTracker,
    compute_class_weights, load_checkpoint, plot_training_curves,
    save_checkpoint, set_seed,
)


# ======================================================================= #
# 消融变体定义
# 格式：(use_frequency_decomp, stream_mode, lambda_adv, lambda_cons)
# ======================================================================= #

ABLATION_VARIANTS = {
    "full":       (True,  "dual",       1.0, 0.5),
    "no_cons":    (True,  "dual",       1.0, 0.0),
    "no_adv":     (True,  "dual",       0.0, 0.5),
    "no_freq":    (False, "dual",       1.0, 0.5),
    "phase_only": (True,  "phase_only", 0.0, 0.0),
    "amp_only":   (True,  "amp_only",   1.0, 0.0),
}


# ======================================================================= #
# 命令行参数
# ======================================================================= #

def parse_args():
    p = argparse.ArgumentParser(description="SDANet 消融实验")
    p.add_argument(
        "--variant", type=str, default=None,
        choices=list(ABLATION_VARIANTS.keys()) + ["all"],
        help="运行指定消融变体（默认运行全部）",
    )
    p.add_argument("--epochs",      type=int,   default=None, help="覆盖训练轮数")
    p.add_argument("--batch_size",  type=int,   default=None, help="覆盖批次大小")
    p.add_argument("--seed",        type=int,   default=42,   help="随机种子")
    p.add_argument("--no_pretrain", action="store_true",      help="不加载预训练权重")
    return p.parse_args()


# ======================================================================= #
# 安全计算损失（处理消融模式下 domain_logits / phase_feat / amp_feat 为 None）
# ======================================================================= #

def safe_compute_loss(
    criterion: TotalLoss,
    cls_logits, labels,
    amp_feat, phase_feat,
    domain_logits, domain_labels,
    feature_dim: int,
    num_domains: int,
    device,
):
    """
    对消融模式下可能为 None 的张量用零张量替代后计算总损失。
    当对应损失权重为 0 时，零张量不影响梯度或结果。
    """
    B = cls_logits.size(0)
    _amp   = amp_feat   if amp_feat   is not None else \
             torch.zeros(B, feature_dim, device=device)
    _phase = phase_feat if phase_feat is not None else \
             torch.zeros(B, feature_dim, device=device)
    _dlogits = domain_logits if domain_logits is not None else \
               torch.zeros(B, num_domains, device=device)
    _dlabels = domain_labels if domain_labels is not None else \
               torch.zeros(B, dtype=torch.long, device=device)

    return criterion(
        cls_logits    = cls_logits,
        labels        = labels,
        amp_feat      = _amp,
        phase_feat    = _phase,
        domain_logits = _dlogits,
        domain_labels = _dlabels,
    )


# ======================================================================= #
# 单轮训练
# ======================================================================= #

def train_one_epoch(model, src_loader, criterion, optimizer, device,
                    global_iter, max_iter, cfg):
    model.train()
    meters = {k: AverageMeter() for k in
              ["loss_total", "loss_cls", "loss_adv", "loss_cons"]}
    all_preds, all_labels, all_probs = [], [], []

    for batch in src_loader:
        if len(batch) == 3:
            imgs, cls_labels, dom_labels = batch
        else:
            imgs, cls_labels = batch
            dom_labels = torch.zeros(imgs.size(0), dtype=torch.long)

        imgs       = imgs.to(device)
        cls_labels = cls_labels.to(device)
        dom_labels = dom_labels.to(device).long()

        # GRL 调度
        lam = grl_lambda_schedule(global_iter, max_iter, cfg.grl_gamma)
        model.set_grl_alpha(lam)

        optimizer.zero_grad()
        cls_logits, domain_logits, amp_feat, phase_feat = model(imgs)

        loss, details = safe_compute_loss(
            criterion, cls_logits, cls_labels,
            amp_feat, phase_feat,
            domain_logits, dom_labels,
            feature_dim=cfg.feature_dim,
            num_domains=cfg.num_domains,
            device=device,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        B = imgs.size(0)
        for k, v in details.items():
            meters[k].update(v, B)

        probs = torch.softmax(cls_logits.detach(), dim=1)[:, 1].cpu().numpy()
        preds = cls_logits.detach().argmax(dim=1).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(cls_labels.cpu().numpy().tolist())
        global_iter += 1

    metrics = {k: v.avg for k, v in meters.items()}
    metrics["train_acc"] = accuracy_score(all_labels, all_preds)
    try:
        metrics["train_auc"] = roc_auc_score(all_labels, all_probs)
    except ValueError:
        metrics["train_auc"] = float("nan")
    return metrics, global_iter


# ======================================================================= #
# 验证
# ======================================================================= #

@torch.no_grad()
def validate(model, val_loader, criterion, device, cfg):
    model.eval()
    meters = {k: AverageMeter() for k in
              ["loss_total", "loss_cls", "loss_adv", "loss_cons"]}
    all_preds, all_labels, all_probs = [], [], []

    for batch in val_loader:
        if len(batch) == 3:
            imgs, cls_labels, dom_labels = batch
        else:
            imgs, cls_labels = batch
            dom_labels = torch.zeros(imgs.size(0), dtype=torch.long)

        imgs       = imgs.to(device)
        cls_labels = cls_labels.to(device)
        dom_labels = dom_labels.to(device).long()

        cls_logits, domain_logits, amp_feat, phase_feat = model(imgs)

        _, details = safe_compute_loss(
            criterion, cls_logits, cls_labels,
            amp_feat, phase_feat,
            domain_logits, dom_labels,
            feature_dim=cfg.feature_dim,
            num_domains=cfg.num_domains,
            device=device,
        )
        B = imgs.size(0)
        for k, v in details.items():
            meters[k].update(v, B)

        probs = torch.softmax(cls_logits, dim=1)[:, 1].cpu().numpy()
        preds = cls_logits.argmax(dim=1).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(cls_labels.cpu().numpy().tolist())

    metrics = {k: v.avg for k, v in meters.items()}
    metrics["val_acc"] = accuracy_score(all_labels, all_preds)
    try:
        metrics["val_auc"] = roc_auc_score(all_labels, all_probs)
    except ValueError:
        metrics["val_auc"] = float("nan")
    return metrics


# ======================================================================= #
# 测试（目标域）
# ======================================================================= #

@torch.no_grad()
def test_on_target(model, cfg, device):
    tgt_ds = MedicalImageDataset(
        root_dir  = cfg.target_dir,
        transform = get_transforms(cfg.image_size, train=False),
    )
    tgt_loader = DataLoader(
        tgt_ds, batch_size=cfg.batch_size,
        shuffle=False, num_workers=cfg.num_workers,
        pin_memory=True,
    )
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for imgs, labels in tqdm(tgt_loader, desc="  测试中", leave=False):
        imgs = imgs.to(device)
        logits = model.predict(imgs)
        probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(all_labels, all_preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
    else:
        sens = spec = float("nan")

    return {"accuracy": acc, "auc": auc, "f1": f1,
            "sensitivity": sens, "specificity": spec}


# ======================================================================= #
# 单变体训练主流程
# ======================================================================= #

def run_variant(variant_name: str, base_cfg: Config,
                override_epochs=None, override_batch=None,
                pretrained=True) -> dict:
    use_freq, stream_mode, lam_adv, lam_cons = ABLATION_VARIANTS[variant_name]

    # ---- 复制配置，避免污染全局 ----
    cfg = copy.deepcopy(base_cfg)
    cfg.lambda_adv = lam_adv
    cfg.lambda_cons = lam_cons
    if override_epochs is not None:
        cfg.num_epochs = override_epochs
    if override_batch is not None:
        cfg.batch_size = override_batch

    # 各变体独立输出目录
    ablation_root = os.path.join(os.path.dirname(base_cfg.output_dir), "ablation_outputs")
    cfg.output_dir = os.path.join(ablation_root, variant_name)
    os.makedirs(os.path.join(cfg.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "logs"), exist_ok=True)

    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  消融变体: {variant_name}")
    print(f"  use_freq={use_freq}, stream_mode={stream_mode}, "
          f"λ_adv={lam_adv}, λ_cons={lam_cons}")
    print(f"  设备: {device}")
    print(f"{'='*60}")

    # ---- 数据加载（正确传递关键字参数，解包3个返回值）----
    src_train_loader, src_val_loader, _ = build_dataloaders(
        center_dirs = cfg.center_dirs,
        target_dir  = cfg.target_dir,
        image_size  = cfg.image_size,
        batch_size  = cfg.batch_size,
        val_split   = cfg.val_split,
        num_workers = cfg.num_workers,
        seed        = cfg.seed,
    )
    print(f"  训练样本: {len(src_train_loader.dataset)}")
    print(f"  验证样本: {len(src_val_loader.dataset)}")

    class_weights = compute_class_weights(
        src_train_loader.dataset,
        num_classes = cfg.num_classes,
        device      = str(device),
    )
    print(f"  类别权重: {class_weights.tolist()}")

    # ---- 模型 ----
    model = FreqDANN(
        phase_encoder_name   = cfg.phase_encoder_name,
        amp_encoder_name     = cfg.amp_encoder_name,
        feature_dim          = cfg.feature_dim,
        disc_hidden_dim      = cfg.disc_hidden_dim,
        num_classes          = cfg.num_classes,
        num_domains          = cfg.num_domains,
        pretrained           = pretrained,
        use_frequency_decomp = use_freq,
        stream_mode          = stream_mode,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  参数量: {total_params:.2f}M")

    # ---- 损失 ----
    criterion = TotalLoss(
        lambda_adv    = lam_adv,
        lambda_cons   = lam_cons,
        class_weights = class_weights,
    )

    # ---- 优化器（按模块分学习率，兼容 None 编码器）----
    encoder_params, head_params = [], []
    for name, p in model.named_parameters():
        if any(k in name for k in ("phase_encoder", "amp_encoder")):
            encoder_params.append(p)
        else:
            head_params.append(p)

    param_groups = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": cfg.lr_encoder})
    if head_params:
        param_groups.append({"params": head_params, "lr": cfg.lr_head})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

    total_steps  = cfg.num_epochs * len(src_train_loader)
    warmup_steps = cfg.warmup_epochs * len(src_train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(1e-6 / cfg.lr_encoder,
                   0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler  = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    early_stop = EarlyStopping(patience=15, higher_is_better=True, min_delta=1e-4)
    tracker    = MetricTracker()

    # ---- 训练循环 ----
    best_auc    = 0.0
    global_iter = 0
    max_iter    = total_steps

    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()
        train_metrics, global_iter = train_one_epoch(
            model, src_train_loader, criterion, optimizer,
            device, global_iter, max_iter, cfg,
        )
        val_metrics = validate(model, src_val_loader, criterion, device, cfg)
        scheduler.step()

        val_auc = val_metrics.get("val_auc", 0.0)
        tracker.update({**train_metrics, **val_metrics})

        if not math.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                os.path.join(cfg.output_dir, "checkpoints", "best_model.pth"),
            )

        print(f"  Epoch [{epoch:3d}/{cfg.num_epochs}]  "
              f"loss={train_metrics['loss_total']:.4f}  "
              f"val_auc={val_auc:.4f} (best={best_auc:.4f})  "
              f"({time.time()-t0:.1f}s)")

        if early_stop.step(val_auc):
            print(f"  [EarlyStopping] 连续 {early_stop.patience} 轮未提升，停止。")
            break

    # ---- 绘制曲线 ----
    try:
        plot_training_curves(tracker.history,
                             save_dir=os.path.join(cfg.output_dir, "logs"))
    except Exception as e:
        print(f"  [警告] 绘图失败（不影响结果）: {e}")

    # ---- 加载最优模型并在目标域测试 ----
    ckpt_path = os.path.join(cfg.output_dir, "checkpoints", "best_model.pth")
    if os.path.isfile(ckpt_path):
        load_checkpoint(ckpt_path, model, device=str(device))
    else:
        print("  [警告] 未找到最优检查点，使用最后一轮权重进行测试。")

    test_results = test_on_target(model, cfg, device)

    print(f"\n  [测试结果 - {variant_name}]")
    for k, v in test_results.items():
        print(f"    {k:15s}: {v:.4f}")

    return {"variant": variant_name, **test_results}


# ======================================================================= #
# 主函数
# ======================================================================= #

def main():
    args = parse_args()
    base_cfg = Config()

    # 决定运行哪些变体
    if args.variant is None or args.variant == "all":
        variants_to_run = list(ABLATION_VARIANTS.keys())
    else:
        variants_to_run = [args.variant]

    all_results = []
    for variant in variants_to_run:
        result = run_variant(
            variant_name    = variant,
            base_cfg        = base_cfg,
            override_epochs = args.epochs,
            override_batch  = args.batch_size,
            pretrained      = not args.no_pretrain,
        )
        all_results.append(result)

    # ---- 汇总结果 ----
    ablation_root = os.path.join(os.path.dirname(base_cfg.output_dir), "ablation_outputs")
    os.makedirs(ablation_root, exist_ok=True)
    csv_path = os.path.join(ablation_root, "ablation_results.csv")

    fieldnames = ["variant", "accuracy", "auc", "f1", "sensitivity", "specificity"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow(
                {k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k])
                 for k in fieldnames}
            )

    print(f"\n{'='*60}")
    print("  消融实验汇总")
    print(f"{'='*60}")
    header = f"  {'Variant':<14} {'Acc':>7} {'AUC':>7} {'F1':>7} {'Sens':>7} {'Spec':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in all_results:
        print(f"  {r['variant']:<14} "
              f"{r['accuracy']:>7.4f} {r['auc']:>7.4f} "
              f"{r['f1']:>7.4f} {r['sensitivity']:>7.4f} {r['specificity']:>7.4f}")
    print(f"\n  结果已保存至 {csv_path}")


if __name__ == "__main__":
    main()
