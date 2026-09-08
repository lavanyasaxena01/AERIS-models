from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img).astype(np.float32) / 255.0


def analyze_semantic_change(
    opt_before_path,
    opt_after_path,
    change_mask,
):
    """
    Lightweight semantic interpretation of a binary change mask.

    Returns human-readable categories such as:
    road, water, built-up, vegetation, agriculture, bare land.
    """

    before = load_rgb(opt_before_path)
    after = load_rgb(opt_after_path)

    if before.shape[:2] != change_mask.shape:
        before = np.asarray(
            Image.fromarray(
                (before * 255).astype(np.uint8)
            ).resize(
                (change_mask.shape[1], change_mask.shape[0])
            )
        ).astype(np.float32) / 255.0

    if after.shape[:2] != change_mask.shape:
        after = np.asarray(
            Image.fromarray(
                (after * 255).astype(np.uint8)
            ).resize(
                (change_mask.shape[1], change_mask.shape[0])
            )
        ).astype(np.float32) / 255.0

    changed = change_mask > 0

    if changed.sum() < 20:
        return {
            "category": "No significant change",
            "confidence": 0.95,
            "description": "Very little spatial change was detected."
        }

    # -------------------------------------------------
    # Extract changed pixels
    # -------------------------------------------------

    b = before[changed]
    a = after[changed]

    # RGB channels
    br, bg, bb = b[:, 0], b[:, 1], b[:, 2]
    ar, ag, ab = a[:, 0], a[:, 1], a[:, 2]

    # Vegetation index approximation
    before_green = (bg - (br + bb) / 2)
    after_green = (ag - (ar + ab) / 2)

    green_change = np.mean(after_green - before_green)

    # Brightness
    before_brightness = np.mean(b, axis=1)
    after_brightness = np.mean(a, axis=1)

    brightness_change = np.mean(
        after_brightness - before_brightness
    )

    # -------------------------------------------------
    # Colour characteristics
    # -------------------------------------------------

    green_after = np.mean(
        (ag > ar * 1.05) &
        (ag > ab * 1.05)
    )

    blue_after = np.mean(
        (ab > ar * 1.05) &
        (ab > ag * 0.95)
    )

    dark_after = np.mean(
        after_brightness < 0.25
    )

    bright_after = np.mean(
        after_brightness > 0.65
    )

    red_brown_after = np.mean(
        (ar > ag * 1.05) &
        (ar > ab * 1.15)
    )

    # -------------------------------------------------
    # Shape analysis
    # -------------------------------------------------

    ys, xs = np.where(changed)

    height = ys.max() - ys.min() + 1
    width = xs.max() - xs.min() + 1

    aspect_ratio = max(width, height) / max(
        min(width, height), 1
    )

    density = changed.sum() / (
        max(height * width, 1)
    )

    # -------------------------------------------------
    # Semantic rules
    # -------------------------------------------------

    # Water
    if blue_after > 0.35:
        return {
            "category": "Lake / River / Water-body change",
            "confidence": min(0.95, 0.60 + blue_after * 0.35),
            "description":
                "The changed region has strong water-like spectral characteristics."
        }

    # Vegetation
    if green_after > 0.45 and green_change > 0.03:
        return {
            "category": "Vegetation / Forest change",
            "confidence": min(0.93, 0.60 + green_after * 0.30),
            "description":
                "The changed region shows vegetation-like spectral characteristics."
        }

    # Road / linear infrastructure
    if aspect_ratio > 5.0 and density < 0.45:
        return {
            "category": "Road / Linear infrastructure change",
            "confidence": 0.72,
            "description":
                "The detected change has an elongated spatial structure consistent with a road or other linear infrastructure."
        }

    # Built-up
    if bright_after > 0.30 or dark_after > 0.30:
        return {
            "category": "Built-up / Construction change",
            "confidence": 0.68,
            "description":
                "The changed region has urban/construction-like brightness characteristics."
        }

    # Bare soil
    if red_brown_after > 0.30:
        return {
            "category": "Bare land / Soil change",
            "confidence": 0.67,
            "description":
                "The changed region has soil/bare-land-like spectral characteristics."
        }

    # Agriculture
    if green_after > 0.20:
        return {
            "category": "Agricultural land change",
            "confidence": 0.60,
            "description":
                "The changed region has characteristics consistent with vegetation or agricultural land."
        }

    return {
        "category": "Land-use change",
        "confidence": 0.55,
        "description":
            "A spatial change was detected, but its land-use type could not be determined confidently."
    }