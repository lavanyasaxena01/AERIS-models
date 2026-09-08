
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from torch.utils.data import DataLoader

from datasets.optical_sar_dataset import OpticalSARDataset
from models.optical_sar_multimodal import OpticalSARChangeNet
from training.common import get_device, run_training
from utils.seed import set_seed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))
    print("device:", device)

    dcfg = cfg["dataset"]
    common = dict(
        root=dcfg["root"],
        dataset_type=dcfg.get("type", "slag"),
        optical_channels=int(dcfg.get("optical_channels", 3)),
        sar_channels=int(dcfg.get("sar_channels", 2)),
        optical_mean=dcfg.get("optical_mean"),
        optical_std=dcfg.get("optical_std"),
        sar_mean=dcfg.get("sar_mean"),
        sar_std=dcfg.get("sar_std"),
        sar_clip=dcfg.get("sar_clip"),
        sar_log_db=bool(dcfg.get("sar_log_db", False)),
        check_georef=bool(dcfg.get("check_georef", True)),
    )
    train_ds = OpticalSARDataset(split="train", augment=dcfg.get("augment"), **common)
    val_ds = OpticalSARDataset(split=dcfg.get("val_split", "val"), augment=None, **common)
    print(f"train: {len(train_ds)} samples | val: {len(val_ds)} samples")

    tcfg = cfg["training"]
    if args.resume:
        tcfg["resume"] = args.resume
    train_loader = DataLoader(
        train_ds, batch_size=int(tcfg.get("batch_size", 8)), shuffle=True,
        num_workers=int(tcfg.get("num_workers", 4)), pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(tcfg.get("batch_size", 8)), shuffle=False,
        num_workers=int(tcfg.get("num_workers", 4)), pin_memory=True,
    )

    model = OpticalSARChangeNet(**cfg["model"]).to(device)
    print(f"model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    run_training(model, train_loader, val_loader, cfg, device, log_name="optical_sar")


if __name__ == "__main__":
    main()
