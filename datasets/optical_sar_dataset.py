
import re
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

try:
    import rasterio
    _HAS_RASTERIO = True
except Exception:  # pragma: no cover
    _HAS_RASTERIO = False


def _years(name: str):
    return set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", name))


def _georef(path: Path):
    """(crs, transform, width, height) for GeoTIFFs, else None."""
    if not _HAS_RASTERIO or path.suffix.lower() not in (".tif", ".tiff"):
        return None
    try:
        with rasterio.open(path) as src:
            return str(src.crs), tuple(src.transform)[:6], src.width, src.height
    except Exception:
        return None


class OpticalSARDataset(Dataset):
    PRESETS = {
        "slag": ("optical_t1", "optical_t2", "sar_t1", "sar_t2", "labels"),
        "default": ("optical_t1", "optical_t2", "sar_t1", "sar_t2", "labels"),
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        dataset_type: str = "slag",
        optical_channels: int = 3,
        sar_channels: int = 2,
        optical_mean: Optional[Sequence[float]] = None,
        optical_std: Optional[Sequence[float]] = None,
        sar_mean: Optional[Sequence[float]] = None,
        sar_std: Optional[Sequence[float]] = None,
        sar_clip: Optional[Sequence[float]] = None,   # e.g. [-35, 5] for dB
        sar_log_db: bool = False,                      # True if SAR stored as linear power
        augment: Optional[Dict] = None,
        subdirs: Optional[Tuple[str, str, str, str, str]] = None,
        strict: bool = True,
        check_georef: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.optical_channels = optical_channels
        self.sar_channels = sar_channels
        self.sar_clip = sar_clip
        self.sar_log_db = sar_log_db
        self.augment = augment
        self.strict = strict
        subs = subdirs or self.PRESETS.get(dataset_type.lower())
        if subs is None:
            raise ValueError(f"unknown dataset_type '{dataset_type}'; pass subdirs=(ot1, ot2, st1, st2, label)")
        self.dirs = [self._resolve(s) for s in subs]
        sets = []
        for d in self.dirs:
            sets.append({p.name for p in d.iterdir() if p.suffix.lower() in IMG_EXTS})
        common = set.intersection(*sets)
        dropped = len(sets[0]) - len(common)
        if dropped:
            warnings.warn(f"{dropped} file(s) dropped: not present in ALL modality/label folders")
        self.samples = sorted(common)
        if not self.samples:
            raise RuntimeError(f"no aligned samples found under {self.root} for split '{split}'")
        self.optical_mean = list(optical_mean) if optical_mean is not None else [0.0] * optical_channels
        self.optical_std = list(optical_std) if optical_std is not None else [1.0] * optical_channels
        self.sar_mean = list(sar_mean) if sar_mean is not None else [0.0] * sar_channels
        self.sar_std = list(sar_std) if sar_std is not None else [1.0] * sar_channels
        self._validate(sample_n=3, check_georef=check_georef)

    def _resolve(self, sub: str) -> Path:
        for cand in (self.root / self.split / sub, self.root / sub / self.split, self.root / sub):
            if cand.is_dir():
                if self.split not in ("all",) and self.split not in cand.parts:
                    warnings.warn(f"'{cand}' has no '{self.split}' split folder -> using ALL files (leakage risk)")
                return cand
        raise FileNotFoundError(f"could not locate '{sub}' under {self.root}")

    def _validate(self, sample_n: int, check_georef: bool) -> None:
        for name in self.samples[:sample_n]:
            shapes = []
            for d in self.dirs:
                arr = load_image(d / name)
                shapes.append(arr.shape[:2])
            if len(set(shapes)) != 1:
                msg = f"[{name}] modality/mask spatial mismatch: {shapes}"
                if self.strict:
                    raise ValueError(msg)
                warnings.warn(msg + " (will be resized at load time; strict=False)")
            y1, y2 = _years(self.dirs[0].name + name), _years(self.dirs[1].name + name)
            if y1 and y2 and y1 == y2:
                warnings.warn(f"[{name}] T1/T2 filenames share year(s) {y1} — verify timestamp pairing")
            if check_georef:
                for o_idx, s_idx in ((0, 2), (1, 3)):  # opt_t1 vs sar_t1, opt_t2 vs sar_t2
                    g_o = _georef(self.dirs[o_idx] / name)
                    g_s = _georef(self.dirs[s_idx] / name)
                    if g_o is None or g_s is None:
                        continue
                    crs_ok = g_o[0] == g_s[0]
                    tr_ok = np.allclose(g_o[1], g_s[1], atol=1e-4)
                    wh_ok = (g_o[2], g_o[3]) == (g_s[2], g_s[3])
                    if not (crs_ok and tr_ok and wh_ok):
                        msg = (f"[{name}] optical/SAR georeferencing mismatch "
                               f"(CRS {g_o[0]} vs {g_s[0]}; transform/shape equal: {tr_ok and wh_ok})")
                        if self.strict:
                            raise ValueError(msg)
                        warnings.warn(msg)
        if check_georef and not _HAS_RASTERIO:
            warnings.warn("rasterio not installed -> CRS/transform checks skipped (pixel-shape checks still active)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        name = self.samples[idx]
        ot1 = adjust_channels(load_image(self.dirs[0] / name), self.optical_channels)
        ot2 = adjust_channels(load_image(self.dirs[1] / name), self.optical_channels)
        st1 = adjust_channels(load_image(self.dirs[2] / name), self.sar_channels)
        st2 = adjust_channels(load_image(self.dirs[3] / name), self.sar_channels)
        mask = load_mask(self.dirs[4] / name)

        ref = ot1.shape[:2]
        if not (ot2.shape[:2] == st1.shape[:2] == st2.shape[:2] == mask.shape[:2] == ref):
            msg = f"[{name}] spatial mismatch ot2{ot2.shape[:2]} st1{st1.shape[:2]} st2{st2.shape[:2]} mask{mask.shape[:2]} ref{ref}"
            if self.strict:
                raise ValueError(msg)
            ot2 = resize_image(ot2, ref)
            st1 = resize_image(st1, ref)
            st2 = resize_image(st2, ref)
            mask = resize_image(mask, ref, is_mask=True)

        if self.augment:
            imgs, mask = random_augment([ot1, ot2, st1, st2], mask, self.augment)
            ot1, ot2, st1, st2 = imgs
            # photometric jitter: OPTICAL ONLY, never SAR
            import random as _r
            if _r.random() < float(self.augment.get("brightness_contrast_p", 0.0)):
                ot1 = random_brightness_contrast(ot1)
                ot2 = random_brightness_contrast(ot2)

        ot1 = torch.from_numpy(np.ascontiguousarray(normalize_image(ot1, self.optical_mean, self.optical_std).transpose(2, 0, 1)))
        ot2 = torch.from_numpy(np.ascontiguousarray(normalize_image(ot2, self.optical_mean, self.optical_std).transpose(2, 0, 1)))
        st1 = torch.from_numpy(np.ascontiguousarray(
            normalize_image(st1, self.sar_mean, self.sar_std, clip_range=self.sar_clip, log_db=self.sar_log_db).transpose(2, 0, 1)))
        st2 = torch.from_numpy(np.ascontiguousarray(
            normalize_image(st2, self.sar_mean, self.sar_std, clip_range=self.sar_clip, log_db=self.sar_log_db).transpose(2, 0, 1)))
        mask = torch.from_numpy(mask[None].astype(np.float32))
        return {"opt_t1": ot1, "opt_t2": ot2, "sar_t1": st1, "sar_t2": st2, "mask": mask, "name": name}
