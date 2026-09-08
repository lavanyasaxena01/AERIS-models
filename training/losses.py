"""Loss functions for highly imbalanced binary change detection."""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss computed on sigmoid probabilities."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        dims = (1, 2, 3)
        inter = (prob * target).sum(dims)
        denom = prob.sum(dims) + target.sum(dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Loss = bce_weight * BCEWithLogits(pos_weight) + dice_weight * Dice."""

    def __init__(self, pos_weight: float = 1.0, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)
        return self.bce_weight * bce + self.dice_weight * self.dice(logits, target)


class FocalLoss(nn.Module):
    """Binary focal loss on logits."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        pt = torch.where(target > 0.5, p, 1 - p)
        alpha_t = torch.where(target > 0.5, self.alpha, 1 - self.alpha)
        return (alpha_t * (1 - pt) ** self.gamma * bce).mean()


class FocalDiceLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, focal_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.focal = FocalLoss(alpha, gamma)
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        return self.focal_weight * self.focal(logits, target) + self.dice_weight * self.dice(logits, target)


class TverskyLoss(nn.Module):
    """Tversky loss; alpha penalises FP, beta penalises FN. beta>alpha favours recall."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        dims = (1, 2, 3)
        tp = (prob * target).sum(dims)
        fp = (prob * (1 - target)).sum(dims)
        fn = ((1 - prob) * target).sum(dims)
        t = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - t.mean()


def build_loss(cfg: Dict) -> nn.Module:
    cfg = cfg or {}
    kind = cfg.get("type", "bce_dice")
    if kind == "bce_dice":
        return BCEDiceLoss(
            pos_weight=cfg.get("pos_weight", 1.0),
            bce_weight=cfg.get("bce_weight", 1.0),
            dice_weight=cfg.get("dice_weight", 1.0),
        )
    if kind == "focal":
        return FocalLoss(alpha=cfg.get("alpha", 0.25), gamma=cfg.get("gamma", 2.0))
    if kind == "focal_dice":
        return FocalDiceLoss(
            alpha=cfg.get("alpha", 0.25), gamma=cfg.get("gamma", 2.0),
            focal_weight=cfg.get("focal_weight", 1.0), dice_weight=cfg.get("dice_weight", 1.0),
        )
    if kind == "tversky":
        return TverskyLoss(alpha=cfg.get("alpha", 0.3), beta=cfg.get("beta", 0.7))
    raise ValueError(f"Unknown loss type: {kind}")
