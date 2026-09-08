
import random
import warnings
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.shared import (
    IMG_EXTS,
    adjust_channels,
    load_image,
    load_mask,
    normalize_image,
    random_augment,
    random_brightness_contrast,
    resize_image,
)


class OpticalChangeDataset(Dataset):
    PRESETS = {
        "levir": ("A", "B", "label"),
        "whu": ("A", "B", "label"),
        "dsifn": ("t1", "t2", "label"),
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        dataset_type: str = "levir",
        in_channels: int = 3,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        augment: Optional[Dict] = None,
        subdirs: Optional[Tuple[str, str, str]] = None,  # (t1_dir, t2_dir, mask_dir) for custom layouts
        strict: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.in_channels = in_channels
        self.augment = augment
        self.strict = strict
        subs = subdirs or self.PRESETS.get(dataset_type.lower())
        if subs is None:
            raise ValueError(f"unknown dataset_type '{dataset_type}'; pass subdirs=(t1, t2, label)")
        self.d1 = self._resolve(subs[0])
        self.d2 = self._resolve(subs[1])
        self.dm = self._resolve(subs[2])
        names = sorted(p.name for p in self.d1.iterdir() if p.suffix.lower() in IMG_EXTS)
        self.samples, missing = [], []
        for n in names:
            if (self.d2 / n).exists() and (self.dm / n).exists():
                self.samples.append(n)
            else:
                missing.append(n)
        if missing:
            warnings.warn(f"{len(missing)} file(s) lack matching T2/label and were skipped (e.g. {missing[:3]})")
        if not self.samples:
            raise RuntimeError(f"no samples found under {self.root} for split '{split}'")
        self.mean = list(mean) if mean is not None else [0.0] * in_channels
        self.std = list(std) if std is not None else [1.0] * in_channels

    def _resolve(self, sub: str) -> Path:
        for cand in (self.root / self.split / sub, self.root / sub / self.split, self.root / sub):
            if cand.is_dir():
                if self.split not in ("all",) and self.split not in cand.parts:
                    warnings.warn(
                        f"'{cand}' contains no '{self.split}' split folder -> using ALL files. "
                        "Create per-split folders to guarantee no train/val leakage."
                    )
                return cand
        raise FileNotFoundError(f"could not locate '{sub}' under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        name = self.samples[idx]
        t1 = adjust_channels(load_image(self.d1 / name), self.in_channels)
        t2 = adjust_channels(load_image(self.d2 / name), self.in_channels)
        mask = load_mask(self.dm / name)

        # ---- spatial alignment validation (never silently assumed) ----
        if not (t1.shape[:2] == t2.shape[:2] == mask.shape[:2]):
            msg = f"[{name}] spatial mismatch T1{t1.shape[:2]} T2{t2.shape[:2]} mask{mask.shape[:2]}"
            if self.strict:
                raise ValueError(msg)
            warnings.warn(msg + " -> resizing T2/mask to T1 (strict=False)")
            t2 = resize_image(t2, t1.shape[:2])
            mask = resize_image(mask, t1.shape[:2], is_mask=True)

        if self.augment:
            (t1, t2), mask = random_augment([t1, t2], mask, self.augment)
            if random.random() < float(self.augment.get("brightness_contrast_p", 0.0)):
                t1 = random_brightness_contrast(t1)
                t2 = random_brightness_contrast(t2)

        t1 = torch.from_numpy(np.ascontiguousarray(normalize_image(t1, self.mean, self.std).transpose(2, 0, 1)))
        t2 = torch.from_numpy(np.ascontiguousarray(normalize_image(t2, self.mean, self.std).transpose(2, 0, 1)))
        mask = torch.from_numpy(mask[None].astype(np.float32))
        return {"t1": t1, "t2": t2, "mask": mask, "name": name}
