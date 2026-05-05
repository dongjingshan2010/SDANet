"""
损失函数定义

TotalLoss：
    L_total = L_cls + λ_adv * L_adv + λ_cons * L_cons

L_cls  : 交叉熵分类损失（源域，有类别标签）
L_adv  : 域对抗损失，CE(domain_logits, domain_labels)
         多中心场景下，domain_labels 为真实的中心索引（0,1,...）
L_cons : 幅度-相位一致性损失，约束两个分支的语义表示相近，
         防止幅度分支在消除域差异时过度破坏病理信息。
         使用 1 - cosine_similarity(amp_feat, phase_feat) 的均值。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    """交叉熵分类损失，支持类别不平衡权重。"""

    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits : [B, num_classes]
            labels : [B]  long tensor
        """
        return self.criterion(logits, labels)


class DomainAdversarialLoss(nn.Module):
    """
    域对抗损失。

    通过 GRL 使特征提取器学会提取域不变特征。
    domain_labels 应为每个样本所属中心的索引（0,1,...,num_domains-1）。
    """

    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self, domain_logits: torch.Tensor, domain_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            domain_logits  : [B, num_domains]  判别器输出
            domain_labels  : [B]               long tensor，每个样本的中心索引
        """
        return self.criterion(domain_logits, domain_labels)


class ConsistencyLoss(nn.Module):
    """
    幅度-相位一致性损失。

    核心思想：相位特征（语义主导）与幅度特征（经过域无关化后）
    在语义空间上应保持一致，防止幅度分支过度移除病理信息。

    L_cons = mean(1 - cos_sim(amp_feat, phase_feat))
    """

    def forward(
        self, amp_feat: torch.Tensor, phase_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            amp_feat   : [B, D]
            phase_feat : [B, D]
        """
        cos_sim = F.cosine_similarity(amp_feat, phase_feat, dim=1)  # [B]
        return (1.0 - cos_sim).mean()


class TotalLoss(nn.Module):
    """
    组合损失：
        L_total = L_cls + λ_adv * L_adv + λ_cons * L_cons

    Args:
        lambda_adv  : 域对抗损失权重
        lambda_cons : 一致性损失权重
        class_weights: 类别不平衡权重（可选）
    """

    def __init__(
        self,
        lambda_adv: float = 1.0,
        lambda_cons: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.lambda_adv = lambda_adv
        self.lambda_cons = lambda_cons

        self.cls_loss  = ClassificationLoss(class_weights)
        self.adv_loss  = DomainAdversarialLoss()
        self.cons_loss = ConsistencyLoss()

    def forward(
        self,
        cls_logits: torch.Tensor,
        labels: torch.Tensor,
        amp_feat: torch.Tensor,
        phase_feat: torch.Tensor,
        domain_logits: Optional[torch.Tensor] = None,
        domain_labels: Optional[torch.Tensor] = None,
        # ----- 旧版接口（向后兼容，已弃用）-----
        src_domain_logits: Optional[torch.Tensor] = None,
        tgt_domain_logits: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            cls_logits    : [B, num_classes]   分类 logits（带类别标签的样本）
            labels        : [B]                类别标签
            amp_feat      : [B, D]             幅度特征（用于一致性）
            phase_feat    : [B, D]             相位特征（用于一致性）
            domain_logits : [B', num_domains]  域判别器 logits（可来自所有
                                              参与对抗训练的样本）
            domain_labels : [B']               真实的域标签（中心索引）

            兼容旧接口（已弃用，保留以避免破坏外部脚本）：
            src_domain_logits : 源域域判别 logits（自动赋 domain=0）
            tgt_domain_logits : 目标域域判别 logits（自动赋 domain=1）

        Returns:
            total   : 标量总损失
            details : 各分项损失字典（用于日志）
        """
        device = cls_logits.device

        # 解析域对抗输入：优先使用新接口（domain_logits + domain_labels）
        if domain_logits is None or domain_labels is None:
            # 回退到旧版双分支接口
            if src_domain_logits is None or tgt_domain_logits is None:
                raise ValueError(
                    "TotalLoss.forward 需要提供 (domain_logits, domain_labels) "
                    "或 (src_domain_logits, tgt_domain_logits) 之一。"
                )
            B_src = src_domain_logits.size(0)
            B_tgt = tgt_domain_logits.size(0)
            src_dom_labels = torch.zeros(B_src, dtype=torch.long, device=device)
            tgt_dom_labels = torch.ones(B_tgt,  dtype=torch.long, device=device)
            domain_logits = torch.cat([src_domain_logits, tgt_domain_logits], dim=0)
            domain_labels = torch.cat([src_dom_labels,    tgt_dom_labels],    dim=0)

        # 分类损失（仅在有类别标签的样本上）
        l_cls = self.cls_loss(cls_logits, labels)

        # 域对抗损失（使用真实域标签）
        l_adv = self.adv_loss(domain_logits, domain_labels)

        # 一致性损失（保护病理信息）
        l_cons = self.cons_loss(amp_feat, phase_feat)

        total = l_cls + self.lambda_adv * l_adv + self.lambda_cons * l_cons

        details = {
            "loss_cls":   l_cls.item(),
            "loss_adv":   l_adv.item(),
            "loss_cons":  l_cons.item(),
            "loss_total": total.item(),
        }
        return total, details
