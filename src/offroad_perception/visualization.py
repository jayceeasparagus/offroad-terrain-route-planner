"""Small Pillow-based visualizations for terrain predictions."""

from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


TERRAIN_COLORS: dict[str, tuple[int, int, int]] = {
    "traversable_ground": (50, 205, 50),
    "soft_or_risky": (255, 165, 0),
    "vegetation": (34, 139, 34),
    "obstacle": (220, 20, 60),
    "unknown": (90, 90, 90),
}


def colorize_classes(class_ids: np.ndarray, class_names: Sequence[str]) -> Image.Image:
    """Render a terrain-ID image with one stable color per class."""
    if class_ids.ndim != 2:
        raise ValueError("class_ids must have shape [height, width]")
    if class_ids.size and (
        class_ids.min() < 0 or class_ids.max() >= len(class_names)
    ):
        raise ValueError("class_ids contains an ID outside class_names")

    palette = np.asarray(
        [TERRAIN_COLORS.get(name, (255, 255, 255)) for name in class_names],
        dtype=np.uint8,
    )
    return Image.fromarray(palette[class_ids], mode="RGB")


def confidence_heatmap(confidence: np.ndarray) -> Image.Image:
    """Render confidence in a dark-blue to yellow heatmap without matplotlib."""
    if confidence.ndim != 2:
        raise ValueError("confidence must have shape [height, width]")

    stops = np.asarray(
        [
            (20, 25, 95),
            (24, 120, 165),
            (45, 190, 160),
            (255, 225, 40),
        ],
        dtype=np.float32,
    )
    values = np.clip(confidence.astype(np.float32), 0.0, 1.0)
    scaled = values * (len(stops) - 1)
    lower = np.floor(scaled).astype(np.intp)
    upper = np.minimum(lower + 1, len(stops) - 1)
    fraction = (scaled - lower)[..., None]
    colors = stops[lower] * (1.0 - fraction) + stops[upper] * fraction
    return Image.fromarray(colors.astype(np.uint8), mode="RGB")


def semantic_overlay(
    image: Image.Image,
    semantic_mask: Image.Image,
    alpha: float = 0.55,
) -> Image.Image:
    """Blend a class-color mask onto the corresponding camera frame."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between zero and one")
    camera = image.convert("RGB")
    mask = semantic_mask.convert("RGB").resize(camera.size, Image.Resampling.NEAREST)
    return Image.blend(camera, mask, alpha)


def render_dashboard(
    image: Image.Image,
    semantic_mask: Image.Image,
    overlay: Image.Image,
    confidence: Image.Image,
    panel_size: tuple[int, int] = (512, 320),
) -> Image.Image:
    """Build a labeled 2-by-2 frame suitable for a demo GIF."""
    panel_width, panel_height = panel_size
    label_height = 22
    panels = [
        ("Camera image", image),
        ("Semantic mask", semantic_mask),
        ("Semantic overlay", overlay),
        ("Prediction confidence", confidence),
    ]
    dashboard = Image.new(
        "RGB",
        (panel_width * 2, (panel_height + label_height) * 2),
        "black",
    )
    draw = ImageDraw.Draw(dashboard)
    font = ImageFont.load_default()

    for index, (title, panel) in enumerate(panels):
        column = index % 2
        row = index // 2
        x = column * panel_width
        y = row * (panel_height + label_height)
        draw.text((x + 6, y + 5), title, fill="white", font=font)
        resized = panel.convert("RGB").resize(
            (panel_width, panel_height), Image.Resampling.BILINEAR
        )
        dashboard.paste(resized, (x, y + label_height))

    return dashboard
