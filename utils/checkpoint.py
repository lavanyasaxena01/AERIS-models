"""Checkpoint save/load helpers."""
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """Load a checkpoint. Accepts either a full trainer state dict (with 'model' key)
    or a raw model state_dict."""
    ckpt = torch.load(str(path), map_location=map_location, weights_only=False)
    model_state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(model_state)
    if optimizer is not None and isinstance(ckpt, dict) and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and isinstance(ckpt, dict) and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt
