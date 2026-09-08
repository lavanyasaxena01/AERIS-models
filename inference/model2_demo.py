import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.shared import (
    adjust_channels,
    load_image,
    normalize_image,
    resize_image,
)
from models.optical_sar_multimodal import OpticalSARChangeNet
from utils.visualization import overlay_mask
from inference.semantic_analysis import analyze_semantic_change

def prep(path, channels, mean, std, clip=None, log_db=False, size=None):
    arr = adjust_channels(load_image(path), channels)

    if size is not None and tuple(arr.shape[:2]) != tuple(size):
        arr = resize_image(arr, size)

    arr = normalize_image(
        arr,
        mean,
        std,
        clip_range=clip,
        log_db=log_db,
    )

    x = torch.from_numpy(
        np.ascontiguousarray(arr.transpose(2, 0, 1))
    ).float()

    return x.unsqueeze(0)


def pad_to_stride(x, stride):
    h, w = x.shape[-2:]

    ph = (stride - h % stride) % stride
    pw = (stride - w % stride) % stride

    if ph or pw:
        x = torch.nn.functional.pad(
            x,
            (0, pw, 0, ph),
            mode="reflect",
        )

    return x, h, w


def main():

    parser = argparse.ArgumentParser(
        description="AERIS Model 2 - Optical + SAR Change Detection"
    )

    parser.add_argument("--query", required=True)

    parser.add_argument("--opt-t1", required=True)
    parser.add_argument("--opt-t2", required=True)

    parser.add_argument("--sar-t1", required=True)
    parser.add_argument("--sar-t2", required=True)

    parser.add_argument(
        "--checkpoint",
        default="runs/optical_sar/best.pt"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3
    )

    parser.add_argument(
        "--out",
        default="outputs/model2_demo"
    )

    args = parser.parse_args()

    print("\n==============================")
    print(" AERIS - MODEL 2")
    print(" Optical + SAR Change Detection")
    print("==============================\n")

    print("User query:")
    print(args.query)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nDevice:", device)

    # --------------------------------------------------
    # Load checkpoint if available
    # --------------------------------------------------

    checkpoint_path = Path(args.checkpoint)

    if checkpoint_path.exists():

        print("Loading checkpoint:", checkpoint_path)

        ckpt = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

        model_cfg = cfg.get("model", {})
        dataset_cfg = cfg.get("dataset", {})

        model_cfg["fusion_mode"] = "gated"
        model_cfg["encoder_widths"] = (64, 128, 256)
        model_cfg["out_strides"] = (8, 16, 32)

        model = OpticalSARChangeNet(**model_cfg)

        state = (
            ckpt["model"]
            if isinstance(ckpt, dict) and "model" in ckpt
            else ckpt
        )

        model.load_state_dict(state)

        print("Checkpoint loaded.")

    else:

        print("\nWARNING:")
        print("No trained Model-2 checkpoint found.")
        print("Running architecture-only demo with random weights.\n")

        model_cfg = {}
        dataset_cfg = {}

        model = OpticalSARChangeNet(
        optical_channels=3,
        sar_channels=2,
            encoder_widths=(64, 128, 256),
            out_strides=(8, 16, 32),
            fusion_mode="gated",
            max_tokens=4096,
        )

    model.to(device)
    model.eval()

    # --------------------------------------------------
    # Dataset configuration
    # --------------------------------------------------

    optical_channels = int(
        model_cfg.get("optical_channels", 3)
    )

    sar_channels = int(
        model_cfg.get("sar_channels", 2)
    )

    optical_mean = dataset_cfg.get(
        "optical_mean",
        [0.0] * optical_channels
    )

    optical_std = dataset_cfg.get(
        "optical_std",
        [1.0] * optical_channels
    )

    sar_mean = dataset_cfg.get(
        "sar_mean",
        [0.0] * sar_channels
    )

    sar_std = dataset_cfg.get(
        "sar_std",
        [1.0] * sar_channels
    )

    sar_clip = dataset_cfg.get("sar_clip")

    sar_log_db = bool(
        dataset_cfg.get("sar_log_db", False)
    )

    # --------------------------------------------------
    # Load images
    # --------------------------------------------------

    print("\nLoading images...")

    optical_t1_raw = load_image(args.opt_t1)

    size = optical_t1_raw.shape[:2]

    opt_t1 = prep(
        args.opt_t1,
        optical_channels,
        optical_mean,
        optical_std,
    )

    opt_t2 = prep(
        args.opt_t2,
        optical_channels,
        optical_mean,
        optical_std,
        size=size,
    )

    sar_t1 = prep(
        args.sar_t1,
        sar_channels,
        sar_mean,
        sar_std,
        clip=sar_clip,
        log_db=sar_log_db,
        size=size,
    )

    sar_t2 = prep(
        args.sar_t2,
        sar_channels,
        sar_mean,
        sar_std,
        clip=sar_clip,
        log_db=sar_log_db,
        size=size,
    )

    inputs = [
        opt_t1,
        opt_t2,
        sar_t1,
        sar_t2,
    ]

    # --------------------------------------------------
    # Pad
    # --------------------------------------------------

    stride = int(
        max(
            model_cfg.get(
                "out_strides",
                [4, 8, 16, 32]
            )
        )
    )

    padded = []

    for x in inputs:

        x, _, _ = pad_to_stride(
            x.to(device),
            stride
        )

        padded.append(x)

    # --------------------------------------------------
    # Inference
    # --------------------------------------------------

    print("Running Optical + SAR model...")

    with torch.no_grad():

        logits = model(*padded)

        probability = torch.sigmoid(logits)

        probability = probability[
            0,
            0,
            :size[0],
            :size[1]
        ].cpu().numpy()

    prediction = (
        probability >= args.threshold
    ).astype(np.uint8)

    # --------------------------------------------------
    # Semantic interpretation
    # --------------------------------------------------

    semantic = analyze_semantic_change(
        args.opt_t1,
        args.opt_t2,
        prediction,
    )

    print("\nSemantic interpretation:")
    print("Category:", semantic["category"])
    print("Confidence:", f"{semantic['confidence']:.2f}")
    print("Description:", semantic["description"])

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    out = Path(args.out)
    out.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        out / "probability.npy",
        probability.astype(np.float32)
    )

    Image.fromarray(
        prediction * 255
    ).save(
        out / "mask.png"
    )

    rgb = adjust_channels(
        optical_t1_raw,
        3
    )

    if rgb.shape[:2] != prediction.shape:

        rgb = resize_image(
            rgb,
            prediction.shape
        )

    overlay = overlay_mask(
        rgb,
        prediction
    )

    Image.fromarray(
        (overlay * 255).astype(np.uint8)
    ).save(
        out / "overlay.png"
    )

    # --------------------------------------------------
    # Text result
    # --------------------------------------------------

    changed_pixels = int(
        prediction.sum()
    )

    total_pixels = prediction.size

    percentage = (
        changed_pixels /
        total_pixels *
        100
    )

    if percentage < 1:
        conclusion = "No significant change detected."

    elif percentage < 10:
        conclusion = "Localized change detected."

    else:
        conclusion = "Significant spatial change detected."

    result = f"""
AERIS Model 2 Result
====================

Query:
{args.query}

Analysis:
Bi-temporal Optical + SAR imagery

Changed pixels:
{changed_pixels:,}

Total pixels:
{total_pixels:,}

Changed area:
{percentage:.2f}%

Conclusion:
{conclusion}

Threshold:
{args.threshold}
"""

    print(result)

    (out / "result.txt").write_text(
        result,
        encoding="utf-8"
    )

    print("Saved:")
    print(out / "probability.npy")
    print(out / "mask.png")
    print(out / "overlay.png")
    print(out / "result.txt")


if __name__ == "__main__":
    main()