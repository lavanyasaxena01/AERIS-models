"""Shared training loop: AMP, AdamW, schedulers, clipping, early stopping,
checkpointing (best/last), history + config logging, optional val visualizations."""
import json
from pathlib import Path
from typing import Dict, Optional
import torch
import torch.nn.functional as F

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from tqdm import tqdm

from training.losses import build_loss
from training.metrics import ChangeMetrics
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.visualization import save_panel


def get_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def build_scheduler(name, optimizer, epochs, steps_per_epoch, lr):
    name = (name or "none").lower()
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs), "epoch"
    if name == "onecycle":
        return OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps_per_epoch), "step"
    return None, "none"


def forward_batch(model, batch):
    if "sar_t1" in batch:
        logits = model(
            batch["opt_t1"],
            batch["opt_t2"],
            batch["sar_t1"],
            batch["sar_t2"],
        )
    else:
        logits = model(batch["t1"], batch["t2"])

    # Make model output spatial size match the ground-truth mask
    target_size = batch["mask"].shape[-2:]

    if logits.shape[-2:] != target_size:
        logits = torch.nn.functional.interpolate(
            logits,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

    return logits


def _move(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


class _Amp:
    """Version-tolerant AMP context/GradScaler."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=enabled)
            self._new = True
        except (AttributeError, TypeError):  # torch < 2.3
            self.scaler = torch.cuda.amp.GradScaler(enabled=enabled)
            self._new = False

    def autocast(self):
        if self._new:
            return torch.amp.autocast("cuda", enabled=self.enabled)
        return torch.cuda.amp.autocast(enabled=self.enabled)


def run_training(model, train_loader, val_loader, cfg: Dict, device: torch.device, log_name: str = ""):
    tcfg = cfg["training"]
    epochs = int(tcfg.get("epochs", 100))
    lr = float(tcfg.get("lr", 3e-4))
    wd = float(tcfg.get("weight_decay", 1e-2))
    clip = float(tcfg.get("grad_clip", 0.0) or 0.0)
    patience = int(tcfg.get("early_stopping_patience", 0) or 0)
    use_amp = bool(tcfg.get("amp", True)) and device.type == "cuda"
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "val_predictions").mkdir(exist_ok=True)

    criterion = build_loss(cfg.get("loss"))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler, sched_step = build_scheduler(tcfg.get("scheduler", "cosine"), optimizer, epochs, len(train_loader), lr)
    amp = _Amp(use_amp)
    threshold = float(cfg.get("eval", {}).get("threshold", 0.5))
    select_metric = cfg.get("eval", {}).get("select_metric", "f1")
    n_save = int(cfg.get("eval", {}).get("save_val_predictions", 0))

    start_epoch, best, history = 1, -1.0, []
    resume = tcfg.get("resume")
    if resume:
        ckpt = load_checkpoint(resume, model, optimizer, scheduler, map_location="cpu")
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best = float(ckpt.get("best", best))
        history = list(ckpt.get("history", []))
        print(f"[resume] {resume} -> starting at epoch {start_epoch}")

    bad = 0
    for epoch in range(start_epoch, epochs + 1):
        # ---------------- train ----------------
        model.train()
        train_loss, n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"{log_name} e{epoch}/{epochs} [train]", leave=False)
        for batch in pbar:
            batch = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast():
                logits = forward_batch(model, batch)
                target = batch["mask"]

                if logits.shape[-2:] != target.shape[-2:]:
                    logits = F.interpolate(
                        logits,
                        size=target.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                loss = criterion(logits, target)
            amp.scaler.scale(loss).backward()
            if clip > 0:
                amp.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            amp.scaler.step(optimizer)
            amp.scaler.update()
            if scheduler is not None and sched_step == "step":
                scheduler.step()
            bs = batch["mask"].size(0)
            train_loss += loss.item() * bs
            n += bs
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss /= max(n, 1)

        # ---------------- validate ----------------
        model.eval()
        val_loss, vn, saved = 0.0, 0, 0
        metrics = ChangeMetrics(threshold=threshold)
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"{log_name} e{epoch}/{epochs} [val]", leave=False):
                batch = _move(batch, device)
                with amp.autocast():
                    logits = forward_batch(model, batch)
                    loss = criterion(logits, batch["mask"])
                bs = batch["mask"].size(0)
                val_loss += loss.item() * bs
                vn += bs
                metrics.update(logits.float(), batch["mask"])
                if saved < n_save:
                    prob = torch.sigmoid(logits.float())
                    pred = (prob >= threshold).float()
                    k1, k2 = ("opt_t1", "opt_t2") if "sar_t1" in batch else ("t1", "t2")
                    for b in range(bs):
                        if saved >= n_save:
                            break
                        save_panel(
                            str(out_dir / "val_predictions" / f"e{epoch:03d}_{saved}.png"),
                            t1=batch[k1][b].cpu().numpy(), t2=batch[k2][b].cpu().numpy(),
                            gt=batch["mask"][b, 0].cpu().numpy(),
                            prob=prob[b, 0].cpu().numpy(), pred=pred[b, 0].cpu().numpy(),
                            title=f"epoch {epoch} sample {saved}",
                        )
                        saved += 1
        val_loss /= max(vn, 1)
        m = metrics.compute()
        if scheduler is not None and sched_step == "epoch":
            scheduler.step()

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **m}
        history.append(row)
        print(
            f"{log_name} epoch {epoch:3d}/{epochs} | train {train_loss:.4f} | val {val_loss:.4f} | "
            f"P {m['precision']:.4f} R {m['recall']:.4f} F1 {m['f1']:.4f} "
            f"IoU {m['iou']:.4f} Dice {m['dice']:.4f} Acc {m['accuracy']:.4f}"
        )

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "best": best,
            "history": history,
            "config": cfg,
        }
        save_checkpoint(state, out_dir / "last.pt")
        if m[select_metric] > best:
            best = m[select_metric]
            state["best"] = best
            save_checkpoint(state, out_dir / "best.pt")
            bad = 0
        else:
            bad += 1
            if patience and bad >= patience:
                print(f"early stopping: no val {select_metric} improvement for {patience} epochs")
                break
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    with open(out_dir / "config_used.json", "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    print(f"done. best val {select_metric} = {best:.4f}; artifacts in {out_dir}")
    return history
