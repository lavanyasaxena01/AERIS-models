from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def to_rgb(x: np.ndarray) -> np.ndarray:
    if x.ndim == 3 and x.shape[0] <= 16 and (x.shape[0] < x.shape[-1] or x.shape[-1] > 16):
        x = np.transpose(x, (1, 2, 0))
    if x.ndim == 2:
        x = np.stack([x] * 3, axis=-1)
    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)
    elif x.shape[-1] > 3:
        x = x[..., :3]
    x = x.astype(np.float32)
    if x.max() > 1.0:
        x /= 255.0
    lo, hi = np.percentile(x, 2), np.percentile(x, 98)
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    return np.clip((x - lo) / (hi - lo), 0, 1)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=(1.0, 0.1, 0.1), alpha: float = 0.5) -> np.ndarray:
    """Blend a binary mask over an RGB image in [0, 1]."""
    out = rgb.copy()
    changed = mask.astype(bool)
    for channel in range(3):
        out[..., channel][changed] = (
            (1 - alpha) * rgb[..., channel][changed] + alpha * color[channel]
        )
    return np.clip(out, 0, 1)


def error_map(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Render TP white, TN black, FP green, and FN red."""
    gt_b, pred_b = gt.astype(bool), pred.astype(bool)
    out = np.zeros((*gt.shape, 3), dtype=np.float32)
    out[gt_b & pred_b] = (1, 1, 1)
    out[~gt_b & pred_b] = (0, 1, 0)
    out[gt_b & ~pred_b] = (1, 0, 0)
    return out


def save_panel(path, t1=None, t2=None, gt=None, prob=None, pred=None,
               changed_regions=None, title=""):
    panels, titles = [], []
    if t1 is not None:
        panels.append(to_rgb(t1)); titles.append("T1")
    if t2 is not None:
        panels.append(to_rgb(t2)); titles.append("T2")
    if gt is not None:
        panels.append(np.stack([gt] * 3, axis=-1).astype(np.float32)); titles.append("Ground Truth")
    if pred is not None:
        panels.append(np.stack([pred] * 3, axis=-1).astype(np.float32)); titles.append("Prediction")
    if prob is not None:
        panels.append(plt.get_cmap("viridis")(prob)[..., :3]); titles.append("Probability")
    if changed_regions is not None:
        panels.append(to_rgb(changed_regions)); titles.append("Changed Regions")
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.4))
    axes = [axes] if len(panels) == 1 else axes
    for ax, panel, panel_title in zip(axes, panels, titles):
        ax.imshow(np.clip(panel, 0, 1))
        ax.set_title(panel_title, fontsize=10)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_semantic_style_changed_regions(
    image,
    prediction_mask,
    output_path,
    min_region_area=50,
    max_regions=20,
):
    """
    Create a semantic-model-style visualization for Model 1.

    Shows:
        - T2/current image as background
        - connected predicted change regions
        - bounding boxes
        - region IDs (R1, R2, ...)
        - region contours

    IMPORTANT:
        This does NOT assign semantic classes.
        It only organizes Model 1's binary change prediction
        into visually distinct regions.
    """

    # ---------------------------------------------------------
    # Convert image to RGB numpy array
    # ---------------------------------------------------------
    if isinstance(image, Image.Image):
        base = np.array(image.convert("RGB"))
    else:
        base = np.asarray(image)

        if base.ndim == 2:
            base = np.stack([base] * 3, axis=-1)

        if base.shape[-1] == 4:
            base = base[..., :3]

        base = base.astype(np.uint8)

    # ---------------------------------------------------------
    # Convert prediction to binary mask
    # ---------------------------------------------------------
    if isinstance(prediction_mask, Image.Image):
        mask = np.array(prediction_mask.convert("L"))
    else:
        mask = np.asarray(prediction_mask)

    if mask.ndim == 3:
        mask = mask[..., 0]

    mask = (mask > 0).astype(np.uint8)

    # ---------------------------------------------------------
    # Connected components
    # ---------------------------------------------------------
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    regions = []

    for region_id in range(1, num_labels):

        x = int(stats[region_id, cv2.CC_STAT_LEFT])
        y = int(stats[region_id, cv2.CC_STAT_TOP])
        w = int(stats[region_id, cv2.CC_STAT_WIDTH])
        h = int(stats[region_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[region_id, cv2.CC_STAT_AREA])

        if area < min_region_area:
            continue

        regions.append({
            "label": region_id,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": area,
            "cx": float(centroids[region_id][0]),
            "cy": float(centroids[region_id][1]),
        })

    # Largest regions first
    regions.sort(key=lambda r: r["area"], reverse=True)

    regions = regions[:max_regions]

    # ---------------------------------------------------------
    # Create visualization
    # ---------------------------------------------------------
    canvas = base.copy()

    # Slight darkening makes boxes/text easier to see
    canvas = (canvas.astype(np.float32) * 0.82).clip(0, 255).astype(np.uint8)

    # ---------------------------------------------------------
    # Draw every detected region
    # ---------------------------------------------------------
    for idx, region in enumerate(regions, start=1):

        x = region["x"]
        y = region["y"]
        w = region["w"]
        h = region["h"]

        # Region mask
        region_mask = (labels == region["label"]).astype(np.uint8)

        # -----------------------------------------------------
        # Create transparent-looking overlay
        # -----------------------------------------------------
        overlay = canvas.copy()

        # Use a visible highlight
        overlay[region_mask > 0] = [255, 80, 80]

        alpha = 0.35

        canvas = np.where(
            region_mask[..., None] > 0,
            (canvas * (1 - alpha) + overlay * alpha).astype(np.uint8),
            canvas
        )

        # -----------------------------------------------------
        # Contour
        # -----------------------------------------------------
        contours, _ = cv2.findContours(
            region_mask,
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
        cv2.rectangle(
            canvas,
            (x, y),
            (x + w, y + h),
            (255, 255, 255),
            3
        )

    # ---------------------------------------------------------
    # PIL for better text rendering
    # ---------------------------------------------------------
    result = Image.fromarray(canvas)
    draw = ImageDraw.Draw(result)

    try:
        font = ImageFont.truetype("arial.ttf", 26)
        small_font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    # ---------------------------------------------------------
    # Draw labels
    # ---------------------------------------------------------
    for idx, region in enumerate(regions, start=1):

        x = region["x"]
        y = region["y"]
        w = region["w"]
        h = region["h"]
        area = region["area"]

        label = f"R{idx}: detected change"

        # Put label above the box whenever possible
        text_x = x
        text_y = max(5, y - 32)

        # Text background
        bbox = draw.textbbox(
            (text_x, text_y),
            label,
            font=font
        )

        draw.rectangle(
            bbox,
            fill=(0, 0, 0)
        )

        draw.text(
            (text_x, text_y),
            label,
            fill=(255, 255, 255),
            font=font
        )

        # Area information inside/near region
        info = f"{area:,} px"

        info_y = min(
            result.height - 25,
            y + h + 5
        )

        draw.text(
            (x, info_y),
            info,
            fill=(255, 255, 255),
            font=small_font
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result.save(output_path)

    return regions