"""Pixel-level metrics for binary change detection."""
from typing import Dict

import torch


class ChangeMetrics:
    """Accumulates a global confusion matrix over batches, then derives metrics.
    Pixel-level (micro) metrics; use `per_image` for per-image scores."""

    def __init__(self, threshold: float = 0.5, eps: float = 1e-7):
        self.threshold = threshold
        self.eps = eps
        self.reset()

    def reset(self) -> None:
        self.tp = self.fp = self.fn = self.tn = 0.0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = torch.sigmoid(logits) >= self.threshold
        t = target > 0.5
        self.tp += (pred & t).sum().item()
        self.fp += (pred & ~t).sum().item()
        self.fn += (~pred & t).sum().item()
        self.tn += (~pred & ~t).sum().item()

    def compute(self) -> Dict[str, float]:
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        precision = tp / (tp + fp + self.eps)
        recall = tp / (tp + fn + self.eps)
        f1 = 2 * precision * recall / (precision + recall + self.eps)
        iou = tp / (tp + fp + fn + self.eps)
        dice = 2 * tp / (2 * tp + fp + fn + self.eps)
        accuracy = (tp + tn) / (tp + fp + fn + tn + self.eps)
        return {
            "precision": precision, "recall": recall, "f1": f1,
            "iou": iou, "dice": dice, "accuracy": accuracy,
        }


@torch.no_grad()
def per_image_metrics(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7):
    """Per-image F1 and IoU -> two [B] tensors (useful for mean/std reporting)."""
    pred = torch.sigmoid(logits) >= threshold
    t = target > 0.5
    dims = (1, 2, 3)
    tp = (pred & t).sum(dims).float()
    fp = (pred & ~t).sum(dims).float()
    fn = (~pred & t).sum(dims).float()
    f1 = 2 * tp / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    return f1, iou
