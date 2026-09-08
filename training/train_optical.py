
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from torch.utils.data import DataLoader

from datasets.optical_change_dataset import OpticalChangeDataset
from models.temporal_optical import SiameseTemporalCD
from training.common import get_device, run_training
from utils.seed import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))
    print("device:", device)

    dcfg = cfg["dataset"]
    common = dict(
        root=dcfg["root"],
        dataset_type=dcfg.get("type", "levir"),
        in_channels=int(dcfg.get("in_channels", 3)),
        mean=dcfg.get("mean"),
        std=dcfg.get("std"),
    )
    train_ds = OpticalChangeDataset(split="train", augment=dcfg.get("augment"), **common)
    val_ds = OpticalChangeDataset(split=dcfg.get("val_split", "val"), augment=None, **common)
    print(f"train: {len(train_ds)} pairs | val: {len(val_ds)} pairs")

    tcfg = cfg["training"]
    if args.resume:
        tcfg["resume"] = args.resume
    train_loader = DataLoader(
        train_ds, batch_size=int(tcfg.get("batch_size", 16)), shuffle=True,
        num_workers=int(tcfg.get("num_workers", 4)), pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(tcfg.get("batch_size", 16)), shuffle=False,
        num_workers=int(tcfg.get("num_workers", 4)), pin_memory=True,
    )

    model = SiameseTemporalCD(**cfg["model"]).to(device)
    print(f"model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    run_training(model, train_loader, val_loader, cfg, device, log_name="optical")


if __name__ == "__main__":
    main()
