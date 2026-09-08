"""Visualization helpers for change-detection results."""
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def to_rgb(x: np.ndarray) -> np.ndarray:
    """Convert a [C,H,W] or [H,W,C] array (any channel count, any scale) to displayable RGB [H,W,3] in [0,1].
    Uses per-image min-max stretch so normalized tensors display sensibly."""
    if x.ndim == 3 and x.shape[0] <= 16 and (x.shape[0] < x.shape[-1] or x.shape[-1] > 16):
        x = np.transpose(x, (1, 2, 0))
    if x.ndim == 2:
        x = np.stack([x] * 3, axis=-1)
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)
    elif x.shape[-1] > 3:
        x = x[..., :3]
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, 2), np.percentile(x, 98)
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    return np.clip((x - lo) / (hi - lo), 0, 1)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=(1.0, 0.1, 0.1), alpha: float = 0.5) -> np.ndarray:
    """Blend a binary mask over an RGB image."""
    out = rgb.copy()
    m = mask.astype(bool)
    for c in range(3):
        out[..., c][m] = (1 - alpha) * rgb[..., c][m] + alpha * color[c]
    return np.clip(out, 0, 1)


def error_map(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """TP=white, TN=black, FP=green, FN=red -> [H,W,3]."""
    gt_b, pr_b = gt.astype(bool), pred.astype(bool)
    out = np.zeros((*gt.shape, 3), dtype=np.float32)          # TN = black
    out[gt_b & pr_b] = (1, 1, 1)                              # TP = white
    out[~gt_b & pr_b] = (0, 1, 0)                             # FP = green
    out[gt_b & ~pr_b] = (1, 0, 0)                             # FN = red
    return out


def save_panel(
    path: str,
    t1: Optional[np.ndarray] = None,
    t2: Optional[np.ndarray] = None,
    gt: Optional[np.ndarray] = None,
    prob: Optional[np.ndarray] = None,
    pred: Optional[np.ndarray] = None,
    title: str = "",
) -> None:
    """Save the standard comparison panel: T1 | T2 | GT | Prob | Pred | Error."""
    panels, titles = [], []
    if t1 is not None:
        panels.append(to_rgb(t1)); titles.append("T1")
    if t2 is not None:
        panels.append(to_rgb(t2)); titles.append("T2")
    if gt is not None:
        panels.append(np.stack([gt] * 3, -1).astype(np.float32)); titles.append("Ground Truth")
    if prob is not None:
        panels.append(plt.get_cmap("viridis")(prob)[..., :3]); titles.append("Probability")
    if pred is not None:
        panels.append(np.stack([pred] * 3, -1).astype(np.float32)); titles.append("Prediction")
    if gt is not None and pred is not None:
        panels.append(error_map(gt, pred)); titles.append("Error (TP=w,FP=g,FN=r)")

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4))
    if n == 1:
        axes = [axes]
    for ax, img, t in zip(axes, panels, titles):
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
