import cv2
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from datasets.shared import adjust_channels, load_image, load_mask, normalize_image, resize_image
from models.optical_sar_multimodal import OpticalSARChangeNet
from models.temporal_optical import SiameseTemporalCD
from utils.visualization import save_panel, to_rgb

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

def create_region_visualization(
    image,
    pred,
    min_region_area=50,
    max_regions=20,
):
    """
    Convert Model 1 binary change prediction into a semantic-model-style
    region visualization.

    IMPORTANT:
    This does NOT change Model 1 predictions.
    It only groups connected changed pixels into regions and visualizes them.

    Output:
        image with:
        - highlighted predicted regions
        - white contours
        - white bounding boxes
        - R1/R2/R3... labels
        - region area
    """

    # ---------------------------------------------------------
    # Convert image from [0,1] RGB to uint8
    # ---------------------------------------------------------
    if image.dtype != np.uint8:
        base = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    else:
        base = image.copy()

    if base.ndim == 2:
        base = np.stack([base] * 3, axis=-1)

    # ---------------------------------------------------------
    # Binary prediction
    # ---------------------------------------------------------
    mask = (pred > 0).astype(np.uint8)

    # ---------------------------------------------------------
    # Small morphological cleanup
    #
    # This is ONLY visualization cleanup.
    # It does not modify the actual prediction_mask.png.
    # ---------------------------------------------------------
    kernel = np.ones((3, 3), np.uint8)

    clean_mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    clean_mask = cv2.morphologyEx(
        clean_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    # ---------------------------------------------------------
    # Connected components
    # ---------------------------------------------------------
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        clean_mask,
        connectivity=8
    )

    regions = []

    for label_id in range(1, num_labels):

        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_id, cv2.CC_STAT_AREA])

        if area < min_region_area:
            continue

        regions.append({
            "label": label_id,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": area,
            "cx": float(centroids[label_id][0]),
            "cy": float(centroids[label_id][1]),
        })

    # Largest regions first
    regions.sort(
        key=lambda r: r["area"],
        reverse=True
    )

    regions = regions[:max_regions]

    # ---------------------------------------------------------
    # Create output canvas
    # ---------------------------------------------------------
    canvas = base.copy()

    # Slight darkening so white region annotations stand out
    canvas = (
        canvas.astype(np.float32) * 0.78
    ).clip(0, 255).astype(np.uint8)

    # ---------------------------------------------------------
    # Highlight each region
    # ---------------------------------------------------------
    for region in regions:

        label_id = region["label"]

        region_mask = (
            labels == label_id
        )

        # Highlight detected change region
        overlay = canvas.copy()

        # Red/pink highlight
        overlay[region_mask] = [255, 70, 70]

        alpha = 0.40

        canvas[region_mask] = (
            canvas[region_mask].astype(np.float32) * (1 - alpha)
            +
            overlay[region_mask].astype(np.float32) * alpha
        ).astype(np.uint8)

        # -----------------------------------------------------
        # Contour
        # -----------------------------------------------------
        region_mask_uint8 = (
            region_mask.astype(np.uint8) * 255
        )

        contours, _ = cv2.findContours(
            region_mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            canvas,
            contours,
            -1,
            (255, 255, 255),
            3
        )

        # -----------------------------------------------------
        # Bounding box
        # -----------------------------------------------------
        x = region["x"]
        y = region["y"]
        w = region["w"]
        h = region["h"]

        cv2.rectangle(
            canvas,
            (x, y),
            (x + w, y + h),
            (255, 255, 255),
            3
        )

    # ---------------------------------------------------------
    # Add region labels using OpenCV
    # ---------------------------------------------------------
    for idx, region in enumerate(regions, start=1):

        x = region["x"]
        y = region["y"]
        w = region["w"]
        h = region["h"]
        area = region["area"]

        label = f"R{idx}: detected change"
        area_text = f"{area:,} px"

        # Put label above bounding box
        text_y = max(30, y - 10)

        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75
        thickness = 2

        # Calculate text box
        (tw, th), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness
        )

        # Black background behind label
        cv2.rectangle(
            canvas,
            (x, text_y - th - baseline - 6),
            (x + tw + 8, text_y + 4),
            (0, 0, 0),
            -1
        )

        # White label
        cv2.putText(
            canvas,
            label,
            (x + 4, text_y - 4),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

        # -----------------------------------------------------
        # Region area text
        # -----------------------------------------------------
        info_y = min(
            canvas.shape[0] - 10,
            y + h + 25
        )

        cv2.putText(
            canvas,
            area_text,
            (x, info_y),
            font,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return canvas, regions


def create_probability_region_visualization(image, prob, threshold, max_regions=20):
    """Build semantic-style regions from continuous probability values only."""
    base = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    min_region_area = max(20, int(prob.size * 0.00002))
    candidate_thresholds = [max(0.15, threshold * 0.5), 0.10, 0.075, 0.05]
    selected_threshold = candidate_thresholds[-1]
    selected_regions = []
    selected_labels = None
    selected_mask = None

    for candidate in candidate_thresholds:
        region_mask = (prob >= candidate).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        region_mask = cv2.dilate(region_mask, kernel, iterations=1)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(region_mask, 8)
        regions = []
        for label_id in range(1, count):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < min_region_area:
                continue
            regions.append({
                "label": label_id,
                "x": int(stats[label_id, cv2.CC_STAT_LEFT]),
                "y": int(stats[label_id, cv2.CC_STAT_TOP]),
                "w": int(stats[label_id, cv2.CC_STAT_WIDTH]),
                "h": int(stats[label_id, cv2.CC_STAT_HEIGHT]),
                "area": area,
                "cx": float(centroids[label_id][0]),
                "cy": float(centroids[label_id][1]),
            })
        regions.sort(key=lambda item: item["area"], reverse=True)
        if regions:
            selected_threshold = candidate
            selected_regions = regions[:max_regions]
            selected_labels = labels
            selected_mask = region_mask
            break

    canvas = (base.astype(np.float32) * 0.78).clip(0, 255).astype(np.uint8)
    for region in selected_regions:
        actual = selected_labels == region["label"]
        highlight = np.zeros_like(canvas)
        highlight[actual] = (255, 55, 55)
        canvas[actual] = (canvas[actual] * 0.45 + highlight[actual] * 0.55).astype(np.uint8)
        contours, _ = cv2.findContours(actual.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (255, 255, 255), 4)
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 255, 255), 4)

    cv2.rectangle(canvas, (10, 10), (390, 58), (0, 0, 0), -1)
    cv2.putText(canvas, "DETECTED CHANGE REGIONS", (20, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "R1, R2, R3...", (20, 51), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (255, 255, 255), 1, cv2.LINE_AA)

    for index, region in enumerate(selected_regions, start=1):
        label = f"R{index}: detected change"
        area_text = f"{region['area']:,} px"
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 0.72, 2
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
        text_x = min(max(x, 5), max(5, canvas.shape[1] - tw - 12))
        text_y = y - 10 if y > th + baseline + 12 else min(canvas.shape[0] - 8, y + th + baseline + 8)
        cv2.rectangle(canvas, (text_x, text_y - th - baseline - 6),
                      (text_x + tw + 8, text_y + 5), (0, 0, 0), -1)
        cv2.putText(canvas, label, (text_x + 4, text_y - 4), font, scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)
        info_y = min(canvas.shape[0] - 10, y + h + 25)
        cv2.putText(canvas, area_text, (x, info_y), font, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)

    if selected_mask is None:
        selected_mask = np.zeros_like(prob, dtype=np.uint8)
    return canvas, selected_mask, selected_regions, selected_threshold

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["optical", "optical_sar"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--t1"); ap.add_argument("--t2")
    ap.add_argument("--opt-t1", dest="opt_t1"); ap.add_argument("--opt-t2", dest="opt_t2")
    ap.add_argument("--sar-t1", dest="sar_t1"); ap.add_argument("--sar-t2", dest="sar_t2")
    ap.add_argument("--gt", help="optional ground-truth mask for metrics and visualization")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--out", default="outputs/prediction")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    mcfg = cfg.get("model", {})
    dcfg = cfg.get("dataset", {})
    threshold = args.threshold if args.threshold is not None else float(cfg.get("eval", {}).get("threshold", 0.5))
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
    pred = (prob >= threshold).astype(np.uint8)

    gt = None
    if args.gt:
        gt = load_mask(args.gt)
        if gt.shape != pred.shape:
            gt = resize_image(gt, pred.shape, is_mask=True)
        gt = (gt > 0).astype(np.uint8)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t1_rgb = to_rgb(adjust_channels(raw_a, 3))
    t2_rgb = to_rgb(adjust_channels(raw_b, 3))
    if t1_rgb.shape[:2] != pred.shape:
        from datasets.shared import resize_image as _ri
        t1_rgb = _ri(t1_rgb, pred.shape)
        t2_rgb = _ri(t2_rgb, pred.shape)
    # =========================================================
    # SEMANTIC-STYLE CHANGE REGION VISUALIZATION
    # =========================================================

    region_visualization, region_mask, detected_regions, visualization_threshold = create_probability_region_visualization(
        image=t2_rgb,
        prob=prob,
        threshold=threshold,
        max_regions=20,
    )
    changed_regions = region_visualization.astype(np.float32) / 255.0
    Image.fromarray((t1_rgb * 255).astype(np.uint8)).save(out / "image_t1.png")
    Image.fromarray((t2_rgb * 255).astype(np.uint8)).save(out / "image_t2.png")
    if gt is not None:
        Image.fromarray(gt * 255).save(out / "ground_truth.png")
    Image.fromarray((pred * 255).astype(np.uint8)).save(out / "prediction_mask.png")
    Image.fromarray((plt.get_cmap("viridis")(prob)[..., :3] * 255).astype(np.uint8)).save(out / "probability_map.png")
    Image.fromarray((changed_regions * 255).astype(np.uint8)).save(out / "changed_regions.png")
    Image.fromarray((region_mask * 255).astype(np.uint8)).save(out / "region_mask.png")
    np.save(out / "probability.npy", prob.astype(np.float32))
    Image.fromarray(pred * 255).save(out / "mask.png")
    Image.fromarray((changed_regions * 255).astype(np.uint8)).save(out / "overlay.png")
    save_panel(str(out / "comparison.png"), t1=t1_rgb, t2=t2_rgb, gt=gt,
               prob=prob, pred=pred, changed_regions=changed_regions, title=args.out)
    save_panel(str(out / "panel.png"), t1=raw_a, t2=raw_b, gt=gt, prob=prob, pred=pred,
               changed_regions=changed_regions, title=args.out)
    if gt is not None:
        gt_b, pred_b = gt.astype(bool), pred.astype(bool)
        tp = np.logical_and(gt_b, pred_b).sum()
        fp = np.logical_and(~gt_b, pred_b).sum()
        fn = np.logical_and(gt_b, ~pred_b).sum()
        tn = np.logical_and(~gt_b, ~pred_b).sum()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        dice = 2 * tp / max(2 * tp + fp + fn, 1)
        iou = tp / max(tp + fp + fn, 1)
        accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
        print(f"IoU: {iou:.6f}")
        print(f"Dice/F1: {dice:.6f}")
        print(f"Precision: {precision:.6f}")
        print(f"Recall: {recall:.6f}")
        print(f"Accuracy: {accuracy:.6f}")
    print(f"checkpoint used: {args.checkpoint}")
    print(f"test image pair: {args.t1} | {args.t2}")
    print(f"ground-truth file: {args.gt or 'not provided'}")
    print(f"prediction threshold: {threshold}")
    print(f"output directory: {out}")
    print("generated files: " + ", ".join(p.name for p in sorted(out.glob("*.png"))))
    print(f"changed pixels: {int(pred.sum())} / {pred.size} ({100 * pred.mean():.2f}%)")
    print(f"visualization threshold: {visualization_threshold}")
    print(f"detected regions: {len(detected_regions)}")
    for idx, region in enumerate(detected_regions, start=1):
        print(
            f"R{idx}: x={region['x']}, y={region['y']}, "
            f"w={region['w']}, h={region['h']}, area={region['area']} px"
        )
    print(f"saved: {out}/probability.npy, mask.png, overlay.png, panel.png")


if __name__ == "__main__":
    main()
