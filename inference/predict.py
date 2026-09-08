
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from datasets.shared import adjust_channels, load_image, normalize_image, resize_image
from models.optical_sar_multimodal import OpticalSARChangeNet
from models.temporal_optical import SiameseTemporalCD
from utils.visualization import overlay_mask, save_panel, to_rgb


def _prep(path, channels, mean, std, clip=None, log_db=False, size=None):
    arr = adjust_channels(load_image(path), channels)
    if size is not None and tuple(arr.shape[:2]) != tuple(size):
        arr = resize_image(arr, size)
    arr = normalize_image(arr, mean, std, clip_range=clip, log_db=log_db)
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))[None]  # [1,C,H,W]


def _pad_to_stride(x: torch.Tensor, stride: int):
    """Pad H/W up to a multiple of `stride`; returns (padded, (h, w))."""
    h, w = x.shape[-2:]
    ph = (stride - h % stride) % stride
    pw = (stride - w % stride) % stride
    if ph or pw:
        x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, (h, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["optical", "optical_sar"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--t1"); ap.add_argument("--t2")
    ap.add_argument("--opt-t1", dest="opt_t1"); ap.add_argument("--opt-t2", dest="opt_t2")
    ap.add_argument("--sar-t1", dest="sar_t1"); ap.add_argument("--sar-t2", dest="sar_t2")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="outputs/prediction")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    mcfg = cfg.get("model", {})
    dcfg = cfg.get("dataset", {})
    stride = int(max(mcfg.get("out_strides", [4, 8, 16, 32])))

    if args.model == "optical":
        assert args.t1 and args.t2, "--t1 and --t2 are required"
        model = SiameseTemporalCD(**mcfg) if mcfg else SiameseTemporalCD()
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        c = int(mcfg.get("in_channels", dcfg.get("in_channels", 3)))
        mean, std = dcfg.get("mean", [0.0] * c), dcfg.get("std", [1.0] * c)
        t1 = _prep(args.t1, c, mean, std)
        t2 = _prep(args.t2, c, mean, std, size=t1.shape[-2:])
        inputs, raw_a, raw_b = (t1, t2), load_image(args.t1), load_image(args.t2)
    else:
        assert all([args.opt_t1, args.opt_t2, args.sar_t1, args.sar_t2]), \
            "--opt-t1/--opt-t2/--sar-t1/--sar-t2 are required"
        model = OpticalSARChangeNet(**mcfg) if mcfg else OpticalSARChangeNet()
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        co = int(mcfg.get("optical_channels", dcfg.get("optical_channels", 3)))
        cs = int(mcfg.get("sar_channels", dcfg.get("sar_channels", 2)))
        size = load_image(args.opt_t1).shape[:2]
        ot1 = _prep(args.opt_t1, co, dcfg.get("optical_mean", [0.0] * co), dcfg.get("optical_std", [1.0] * co))
        ot2 = _prep(args.opt_t2, co, dcfg.get("optical_mean", [0.0] * co), dcfg.get("optical_std", [1.0] * co), size=size)
        st1 = _prep(args.sar_t1, cs, dcfg.get("sar_mean", [0.0] * cs), dcfg.get("sar_std", [1.0] * cs),
                    clip=dcfg.get("sar_clip"), log_db=bool(dcfg.get("sar_log_db", False)), size=size)
        st2 = _prep(args.sar_t2, cs, dcfg.get("sar_mean", [0.0] * cs), dcfg.get("sar_std", [1.0] * cs),
                    clip=dcfg.get("sar_clip"), log_db=bool(dcfg.get("sar_log_db", False)), size=size)
        inputs, raw_a, raw_b = (ot1, ot2, st1, st2), load_image(args.opt_t1), load_image(args.opt_t2)

    model.to(device).eval()
    with torch.no_grad():
        xs, shapes = [], []
        for x in inputs:
            xp, hw = _pad_to_stride(x.to(device), stride)
            xs.append(xp); shapes.append(hw)
        logits = model(*xs)
        prob = torch.sigmoid(logits)[0, 0, : shapes[0][0], : shapes[0][1]].cpu().numpy()
    pred = (prob >= args.threshold).astype(np.uint8)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "probability.npy", prob.astype(np.float32))
    Image.fromarray(pred * 255).save(out / "mask.png")
    rgb = to_rgb(adjust_channels(raw_a, 3))
    if rgb.shape[:2] != pred.shape:
        from datasets.shared import resize_image as _ri
        rgb = _ri(rgb, pred.shape)
    Image.fromarray((overlay_mask(rgb, pred) * 255).astype(np.uint8)).save(out / "overlay.png")
    save_panel(str(out / "panel.png"), t1=raw_a, t2=raw_b, prob=prob, pred=pred, title=args.out)
    print(f"changed pixels: {int(pred.sum())} / {pred.size} ({100 * pred.mean():.2f}%)")
    print(f"saved: {out}/probability.npy, mask.png, overlay.png, panel.png")


if __name__ == "__main__":
    main()
