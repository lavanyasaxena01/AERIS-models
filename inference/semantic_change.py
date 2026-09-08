import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from datasets.shared import (
    adjust_channels,
    load_image,
    load_mask,
    normalize_image,
    resize_image,
)
from models.optical_sar_multimodal import OpticalSARChangeNet
from utils.visualization import overlay_mask, to_rgb


# ============================================================
# IMAGE PREPARATION
# ============================================================

def prep(path, channels, mean, std, size=None, clip=None, log_db=False):
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


# ============================================================
# CHANGE REGION EXTRACTION
# ============================================================

def clean_mask(mask, min_area=20):
    """
    Remove tiny isolated change regions.
    """

    mask = (mask > 0).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    cleaned = np.zeros_like(mask)

    regions = []

    for i in range(1, num_labels):

        area = int(stats[i, cv2.CC_STAT_AREA])

        if area < min_area:
            continue

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])

        cleaned[labels == i] = 1

        regions.append(
            {
                "area": area,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
            }
        )

    regions.sort(
        key=lambda r: r["area"],
        reverse=True,
    )

    return cleaned, regions


# ============================================================
# QUERY INTERPRETATION
# ============================================================

def query_target(query):

    q = query.lower()

    if any(x in q for x in [
        "road",
        "highway",
        "street",
        "road constructed",
        "road construction",
        "infrastructure",
    ]):
        return "road"

    if any(x in q for x in [
        "lake",
        "pond",
        "reservoir",
        "water body",
        "waterbody",
    ]):
        return "water"

    if any(x in q for x in [
        "river",
        "river channel",
        "stream",
        "water channel",
    ]):
        return "river"

    if any(x in q for x in [
        "building",
        "buildings",
        "construction",
        "constructed",
        "urban",
        "structure",
    ]):
        return "building"

    if any(x in q for x in [
        "vegetation",
        "forest",
        "trees",
        "deforestation",
        "vegetation loss",
        "greenery",
    ]):
        return "vegetation"

    if any(x in q for x in [
        "land use",
        "land-use",
        "land cover",
        "land-cover",
    ]):
        return "landuse"

    return "general"


# ============================================================
# SEMANTIC HEURISTICS
# ============================================================

def classify_region(region, optical_before, optical_after, sar_before, sar_after):
    """
    Lightweight interpretation.

    IMPORTANT:
    This is NOT a trained semantic segmentation classifier.
    It uses shape + optical/SAR change characteristics.
    """

    x = region["x"]
    y = region["y"]
    w = region["width"]
    h = region["height"]
    area = region["area"]

    aspect = max(w, h) / max(min(w, h), 1)

    # Crop changed region
    ob = optical_before[y:y+h, x:x+w]
    oa = optical_after[y:y+h, x:x+w]

    sb = sar_before[y:y+h, x:x+w]
    sa = sar_after[y:y+h, x:x+w]

    if ob.size == 0 or oa.size == 0:
        return "other", 0.40

    # --------------------------------------------------------
    # Optical change
    # --------------------------------------------------------

    optical_difference = np.mean(
        np.abs(
            oa.astype(np.float32)
            - ob.astype(np.float32)
        )
    )

    # --------------------------------------------------------
    # SAR change
    # --------------------------------------------------------

    sar_difference = np.mean(
        np.abs(
            sa.astype(np.float32)
            - sb.astype(np.float32)
        )
    )

    # Normalize rough image differences
    optical_score = min(
        optical_difference / 60.0,
        1.0
    )

    sar_score = min(
        sar_difference / 60.0,
        1.0
    )

    # --------------------------------------------------------
    # Shape heuristics
    # --------------------------------------------------------

    elongated = aspect >= 4.0

    very_elongated = aspect >= 7.0

    large_region = area >= 1000

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    # Roads / linear infrastructure
    if very_elongated:
        confidence = 0.65

        if sar_score > 0.25:
            confidence += 0.10

        if optical_score > 0.20:
            confidence += 0.05

        return "road / linear infrastructure", min(confidence, 0.90)

    if elongated:
        confidence = 0.55

        if optical_score > 0.20:
            confidence += 0.08

        if sar_score > 0.20:
            confidence += 0.08

        return "road / linear infrastructure", min(confidence, 0.85)

    # Large water-body-like change
    if large_region and optical_score > 0.25:
        confidence = 0.52

        if sar_score > 0.25:
            confidence += 0.10

        return "water body / land-cover change", min(confidence, 0.78)

    # Compact structural change
    if not elongated and area >= 300:
        confidence = 0.50

        if sar_score > 0.25:
            confidence += 0.10

        if optical_score > 0.25:
            confidence += 0.08

        return "building / structural change", min(confidence, 0.78)

    return "other land-use change", 0.50


# ============================================================
# QUERY-BASED RESULT
# ============================================================

def interpret_query(query, region_results):

    target = query_target(query)

    if not region_results:
        return (
            "NO SIGNIFICANT CHANGE",
            0.0,
            "The Optical + SAR model did not identify a sufficiently large "
            "changed region."
        )

    # --------------------------------------------------------
    # Specific query
    # --------------------------------------------------------

    if target == "road":

        road_regions = [
            r for r in region_results
            if "road" in r["type"]
        ]

        if road_regions:

            confidence = max(
                r["confidence"]
                for r in road_regions
            )

            return (
                "LIKELY ROAD / LINEAR INFRASTRUCTURE CHANGE",
                confidence,
                "The detected change contains elongated spatial regions "
                "consistent with a road or other linear infrastructure."
            )

        return (
            "CHANGE DETECTED — ROAD NOT CONFIRMED",
            0.45,
            "Spatial change was detected, but the current lightweight "
            "interpretation does not provide enough evidence to classify "
            "it specifically as a road."
        )

    if target in ("water", "river"):

        water_regions = [
            r for r in region_results
            if "water" in r["type"]
        ]

        if water_regions:

            confidence = max(
                r["confidence"]
                for r in water_regions
            )

            if target == "river":
                text = (
                    "A water-related spatial change was detected. "
                    "The changed region may represent a river/channel "
                    "modification."
                )
            else:
                text = (
                    "A water-related spatial change was detected. "
                    "The changed region may represent a lake, pond, "
                    "reservoir, or other water-body modification."
                )

            return (
                "LIKELY WATER-BODY CHANGE",
                confidence,
                text
            )

        return (
            "CHANGE DETECTED — WATER CHANGE NOT CONFIRMED",
            0.45,
            "Change is present, but the available imagery and "
            "lightweight interpretation layer cannot confidently "
            "attribute it specifically to a water body."
        )

    if target == "building":

        building_regions = [
            r for r in region_results
            if "building" in r["type"]
        ]

        if building_regions:

            confidence = max(
                r["confidence"]
                for r in building_regions
            )

            return (
                "LIKELY BUILDING / STRUCTURAL CHANGE",
                confidence,
                "Compact changed regions are consistent with "
                "construction or structural modification."
            )

        return (
            "CHANGE DETECTED — STRUCTURE NOT CONFIRMED",
            0.45,
            "Spatial change was detected, but a specific building "
            "interpretation cannot be confirmed."
        )

    if target == "vegetation":

        return (
            "VEGETATION / LAND-COVER CHANGE POSSIBLE",
            0.55,
            "Optical and SAR differences indicate a possible "
            "land-cover or vegetation change."
        )

    # General query
    best = max(
        region_results,
        key=lambda r: r["confidence"]
    )

    return (
        best["type"].upper(),
        best["confidence"],
        "The detected spatial pattern is most consistent with "
        + best["type"] + "."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="AERIS Model-2 semantic change interpretation"
    )

    parser.add_argument(
        "--query",
        required=True,
    )

    parser.add_argument(
        "--opt-t1",
        required=True,
    )

    parser.add_argument(
        "--opt-t2",
        required=True,
    )

    parser.add_argument(
        "--sar-t1",
        required=True,
    )

    parser.add_argument(
        "--sar-t2",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        default="runs/optical_sar/best.pt",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--gt",
        help="real ground-truth annotation for visualization and metrics",
    )

    parser.add_argument(
        "--min-region",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--out",
        default="outputs/semantic_test",
    )

    args = parser.parse_args()

    print("\n======================================")
    print(" AERIS - MODEL 2")
    print(" Optical + SAR Semantic Analysis")
    print("======================================\n")

    print("Query:")
    print(args.query)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nDevice:", device)

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    print(
        "\nLoading checkpoint:",
        checkpoint_path
    )

    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    cfg = (
        ckpt.get("config", {})
        if isinstance(ckpt, dict)
        else {}
    )

    model_cfg = dict(
        cfg.get("model", {})
    )

    dataset_cfg = dict(
        cfg.get("dataset", {})
    )

    model = OpticalSARChangeNet(
        **model_cfg
    )

    state = (
        ckpt["model"]
        if isinstance(ckpt, dict)
        and "model" in ckpt
        else ckpt
    )

    model.load_state_dict(
        state
    )

    model.to(device)
    model.eval()

    print("Checkpoint loaded.")

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    optical_channels = int(
        model_cfg.get(
            "optical_channels",
            3
        )
    )

    sar_channels = int(
        model_cfg.get(
            "sar_channels",
            2
        )
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

    sar_clip = dataset_cfg.get(
        "sar_clip"
    )

    sar_log_db = bool(
        dataset_cfg.get(
            "sar_log_db",
            False
        )
    )

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    print("\nLoading images...")

    optical_before_raw = load_image(
        args.opt_t1
    )

    size = optical_before_raw.shape[:2]

    optical_after_raw = load_image(
        args.opt_t2
    )

    sar_before_raw = load_image(
        args.sar_t1
    )

    sar_after_raw = load_image(
        args.sar_t2
    )

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
        size=size,
        clip=sar_clip,
        log_db=sar_log_db,
    )

    sar_t2 = prep(
        args.sar_t2,
        sar_channels,
        sar_mean,
        sar_std,
        size=size,
        clip=sar_clip,
        log_db=sar_log_db,
    )

    # --------------------------------------------------------
    # Pad
    # --------------------------------------------------------

    stride = int(
        max(
            model_cfg.get(
                "out_strides",
                [8, 16, 32]
            )
        )
    )

    inputs = [
        opt_t1,
        opt_t2,
        sar_t1,
        sar_t2,
    ]

    padded = []

    for x in inputs:

        x, _, _ = pad_to_stride(
            x.to(device),
            stride
        )

        padded.append(x)

    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    print(
        "\nRunning Optical + SAR model..."
    )

    with torch.no_grad():

        logits = model(
            *padded
        )

        probability = torch.sigmoid(
            logits
        )

        probability = probability[
            0,
            0,
            :size[0],
            :size[1],
        ].cpu().numpy()

    mask = (
        probability >= args.threshold
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Clean mask
    # --------------------------------------------------------

    clean, regions = clean_mask(
        mask,
        min_area=args.min_region
    )

    ground_truth = None
    if args.gt:
        ground_truth = load_mask(args.gt)
        if ground_truth.shape != clean.shape:
            ground_truth = resize_image(ground_truth, clean.shape, is_mask=True)
        ground_truth = (ground_truth > 0).astype(np.uint8)

    print(
        f"\nDetected change regions: {len(regions)}"
    )

    # --------------------------------------------------------
    # Prepare arrays for semantic analysis
    # --------------------------------------------------------

    optical_before = adjust_channels(
        optical_before_raw,
        3
    )

    optical_after = adjust_channels(
        optical_after_raw,
        3
    )

    sar_before = adjust_channels(
        sar_before_raw,
        sar_channels
    )

    sar_after = adjust_channels(
        sar_after_raw,
        sar_channels
    )

    if optical_after.shape[:2] != size:

        optical_after = resize_image(
            optical_after,
            size
        )

    if sar_before.shape[:2] != size:

        sar_before = resize_image(
            sar_before,
            size
        )

    if sar_after.shape[:2] != size:

        sar_after = resize_image(
            sar_after,
            size
        )

    # --------------------------------------------------------
    # Classify regions
    # --------------------------------------------------------

    region_results = []

    for i, region in enumerate(regions):

        region_type, confidence = classify_region(
            region,
            optical_before,
            optical_after,
            sar_before,
            sar_after,
        )

        region_results.append(
            {
                **region,
                "id": i + 1,
                "type": region_type,
                "confidence": confidence,
            }
        )

    # --------------------------------------------------------
    # Query interpretation
    # --------------------------------------------------------

    conclusion, confidence, explanation = interpret_query(
        args.query,
        region_results,
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    out = Path(args.out)

    out.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        out / "probability.npy",
        probability.astype(
            np.float32
        )
    )

    Image.fromarray(
        clean * 255
    ).save(
        out / "mask.png"
    )

    optical_before_rgb = to_rgb(optical_before)
    optical_after_rgb = to_rgb(optical_after_raw)
    if optical_after_rgb.shape[:2] != clean.shape:
        optical_after_rgb = resize_image(optical_after_rgb, clean.shape)

    overlay = overlay_mask(
        optical_after_rgb,
        clean
    )

    Image.fromarray(
        (overlay * 255).astype(
            np.uint8
        )
    ).save(
        out / "overlay.png"
    )

    # --------------------------------------------------------
    # Region visualization
    # --------------------------------------------------------

    visualization = (optical_after_rgb * 255).astype(np.uint8)

    visualization = np.clip(
        visualization,
        0,
        255
    ).astype(np.uint8)

    if visualization.shape[2] != 3:

        visualization = cv2.cvtColor(
            visualization,
            cv2.COLOR_GRAY2RGB
        )

    for region in region_results:

        x = region["x"]
        y = region["y"]
        w = region["width"]
        h = region["height"]

        cv2.rectangle(
            visualization,
            (x, y),
            (x + w, y + h),
            (255, 255, 255),
            2,
        )

        label = (
            f"R{region['id']}: "
            f"{region['type']}"
        )

        cv2.putText(
            visualization,
            label,
            (x, max(15, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    Image.fromarray(
        visualization
    ).save(
        out / "change_regions.png"
    )
    Image.fromarray(visualization).save(out / "semantic_change.png")

    Image.fromarray((to_rgb(optical_before) * 255).astype(np.uint8)).save(out / "image_t1.png")
    Image.fromarray((optical_after_rgb * 255).astype(np.uint8)).save(out / "image_t2.png")
    if ground_truth is not None:
        Image.fromarray(ground_truth * 255).save(out / "ground_truth.png")
    Image.fromarray(clean * 255).save(out / "prediction_mask.png")
    probability_rgb = plt.get_cmap("viridis")(probability)[..., :3]
    Image.fromarray((probability_rgb * 255).astype(np.uint8)).save(out / "probability_map.png")
    Image.fromarray((overlay * 255).astype(np.uint8)).save(out / "changed_regions.png")

    panels = [
        to_rgb(optical_before), optical_after_rgb,
        np.stack([ground_truth] * 3, -1).astype(np.float32) if ground_truth is not None else np.zeros((*clean.shape, 3), dtype=np.float32),
        np.stack([clean] * 3, -1).astype(np.float32), probability_rgb, overlay,
    ]
    titles = ["T1", "T2", "Ground Truth", "Prediction", "Probability Map", "Changed Regions"]
    fig, axes = plt.subplots(1, 6, figsize=(19, 3.8))
    for ax, panel, title in zip(axes, panels, titles):
        image = ax.imshow(np.clip(panel, 0, 1), cmap="viridis" if title == "Probability Map" else None)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        if title == "Probability Map":
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    if ground_truth is not None:
        gt_b, pred_b = ground_truth.astype(bool), clean.astype(bool)
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
    print(f"input image pair: {args.opt_t1} | {args.opt_t2} plus SAR pair {args.sar_t1} | {args.sar_t2}")
    print(f"ground-truth annotation: {args.gt or 'not provided'}")
    print(f"threshold: {args.threshold}; minimum region area: {args.min_region}")
    print(f"prediction statistics: changed pixels={int(clean.sum())}, total pixels={clean.size}, changed percent={100 * clean.mean():.2f}%, regions={len(region_results)}")
    print(f"output directory: {out}")
    print("generated PNG files: " + ", ".join(p.name for p in sorted(out.glob("*.png"))))

    # --------------------------------------------------------
    # Result text
    # --------------------------------------------------------

    changed_pixels = int(
        clean.sum()
    )

    total_pixels = clean.size

    percentage = (
        changed_pixels /
        max(total_pixels, 1) *
        100
    )

    result = f"""
AERIS MODEL 2
OPTICAL + SAR SEMANTIC CHANGE ANALYSIS
=======================================

Query:
{args.query}

Change detected:
{"YES" if changed_pixels > 0 else "NO"}

Detected change regions:
{len(region_results)}

Changed pixels:
{changed_pixels:,}

Changed image percentage:
{percentage:.2f}%

Interpretation:
{conclusion}

Interpretation confidence:
{confidence * 100:.1f}%

Explanation:
{explanation}

REGIONS
=======

"""

    for region in region_results:

        result += (
            f"Region {region['id']}\n"
            f"  Type: {region['type']}\n"
            f"  Confidence: "
            f"{region['confidence'] * 100:.1f}%\n"
            f"  Bounding box: "
            f"x={region['x']}, "
            f"y={region['y']}, "
            f"w={region['width']}, "
            f"h={region['height']}\n"
            f"  Pixels: {region['area']}\n\n"
        )

    print(result)

    (out / "result.txt").write_text(
        result,
        encoding="utf-8"
    )

    print("Saved:")
    print(
        out / "probability.npy"
    )
    print(
        out / "mask.png"
    )
    print(
        out / "overlay.png"
    )
    print(
        out / "change_regions.png"
    )
    print(
        out / "result.txt"
    )


if __name__ == "__main__":
    main()