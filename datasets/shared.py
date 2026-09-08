
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    import tifffile
    _HAS_TIFF = True
except Exception:  # pragma: no cover
    _HAS_TIFF = False

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def load_image(path) -> np.ndarray:
    """Load any supported image -> np.ndarray [H, W] or [H, W, C]. Dtype preserved."""
    path = str(path)
    if Path(path).suffix.lower() in (".tif", ".tiff") and _HAS_TIFF:
        return tifffile.imread(path)
    return np.array(Image.open(path))


def load_mask(path) -> np.ndarray:
    """Load a binary change mask -> uint8 [H, W] with values {0, 1}."""
    arr = load_image(path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.max() <= 1:
        return arr.astype(np.uint8)
    return (arr > 127).astype(np.uint8)


def resize_image(arr: np.ndarray, size_hw: Tuple[int, int], is_mask: bool = False) -> np.ndarray:
    """Resize [H,W] or [H,W,C] to size_hw (h, w). Per-channel so multispectral works."""
    h, w = size_hw
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    if arr.ndim == 2:
        return np.array(Image.fromarray(arr).resize((w, h), resample))
    chans = [np.array(Image.fromarray(arr[..., c]).resize((w, h), resample)) for c in range(arr.shape[-1])]
    return np.stack(chans, axis=-1)


def adjust_channels(arr: np.ndarray, in_channels: Optional[int]) -> np.ndarray:
    """Make arr [H, W, C] with C == in_channels (truncate or zero-pad)."""
    if arr.ndim == 2:
        arr = arr[..., None]
    if in_channels is None or arr.shape[-1] == in_channels:
        return arr
    c = arr.shape[-1]
    if c > in_channels:
        return arr[..., :in_channels]
    pad = np.zeros((*arr.shape[:2], in_channels - c), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=-1)


def normalize_image(
    arr: np.ndarray,
    mean: Sequence[float],
    std: Sequence[float],
    clip_range: Optional[Sequence[float]] = None,
    log_db: bool = False,
) -> np.ndarray:
    """Per-modality normalization -> float32 [H, W, C].

    clip_range: optional [lo, hi] clip before scaling (e.g. [-35, 5] dB for SAR).
    log_db: 10*log10(x) for SAR stored as linear power/backscatter.
    """
    x = arr.astype(np.float32)
    if x.ndim == 2:
        x = x[..., None]
    if clip_range is not None:
        x = np.clip(x, float(clip_range[0]), float(clip_range[1]))
    if log_db:
        x = 10.0 * np.log10(np.maximum(x, 1e-6))
    c = x.shape[-1]
    mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, -1)
    std = np.asarray(std, dtype=np.float32).reshape(1, 1, -1)
    if mean.shape[-1] < c:  # pad stats for extra channels (mean 0, std 1)
        mean = np.concatenate([mean, np.zeros((1, 1, c - mean.shape[-1]), np.float32)], -1)
        std = np.concatenate([std, np.ones((1, 1, c - std.shape[-1]), np.float32)], -1)
    return (x - mean[..., :c]) / std[..., :c]


def random_brightness_contrast(img: np.ndarray, b: float = 0.15, c: float = 0.15) -> np.ndarray:
    """Mild photometric jitter. Only applied to 8-bit optical imagery."""
    if img.dtype != np.uint8:
        return img
    x = img.astype(np.float32)
    alpha = 1.0 + random.uniform(-c, c)
    beta = random.uniform(-b, b) * 255.0
    return np.clip(alpha * x + beta, 0, 255).astype(np.uint8)


def random_augment(
    images: List[np.ndarray],
    mask: np.ndarray,
    cfg: Optional[Dict],
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Joint spatial augmentation: identical transform for all images + mask.

    cfg keys: scale_min (random downscale prob 0.5), crop_size (int or [h, w]),
    hflip_p, vflip_p, rotate90. All ops are alignment-preserving.
    """
    cfg = cfg or {}
    scale_min = float(cfg.get("scale_min", 1.0) or 1.0)
    if scale_min < 1.0 and random.random() < float(cfg.get("scale_p", 0.5)):
        h, w = images[0].shape[:2]
        s = random.uniform(scale_min, 1.0)
        nh, nw = max(32, int(round(h * s))), max(32, int(round(w * s)))
        images = [resize_image(im, (nh, nw)) for im in images]
        mask = resize_image(mask, (nh, nw), is_mask=True)
    crop = cfg.get("crop_size")
    if crop:
        ch, cw = (int(crop), int(crop)) if isinstance(crop, (int, float)) else (int(crop[0]), int(crop[1]))
        h, w = images[0].shape[:2]
        if h < ch or w < cw:
            nh, nw = max(h, ch), max(w, cw)
            images = [resize_image(im, (nh, nw)) for im in images]
            mask = resize_image(mask, (nh, nw), is_mask=True)
            h, w = nh, nw
        i = random.randint(0, h - ch)
        j = random.randint(0, w - cw)
        images = [im[i:i + ch, j:j + cw] for im in images]
        mask = mask[i:i + ch, j:j + cw]
    if random.random() < float(cfg.get("hflip_p", 0.5)):
        images = [np.ascontiguousarray(im[:, ::-1]) for im in images]
        mask = np.ascontiguousarray(mask[:, ::-1])
    if random.random() < float(cfg.get("vflip_p", 0.5)):
        images = [np.ascontiguousarray(im[::-1]) for im in images]
        mask = np.ascontiguousarray(mask[::-1])
    if cfg.get("rotate90", True):
        k = random.randint(0, 3)
        if k:
            images = [np.ascontiguousarray(np.rot90(im, k)) for im in images]
            mask = np.ascontiguousarray(np.rot90(mask, k))
    return images, mask
