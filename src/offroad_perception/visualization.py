"""Small Pillow-based visualizations for terrain predictions and routes."""

from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .routes import RouteCandidate


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


def traversability_heatmap(route_cost: np.ndarray, blocked: np.ndarray) -> Image.Image:
    """Render low route cost in green and high route cost in red."""
    if route_cost.ndim != 2 or blocked.shape != route_cost.shape:
        raise ValueError("route_cost and blocked must share a two-dimensional shape")
    values = np.clip(route_cost.astype(np.float32), 0.0, 1.0)
    low = np.asarray((25, 200, 65), dtype=np.float32)
    medium = np.asarray((255, 215, 40), dtype=np.float32)
    high = np.asarray((220, 45, 45), dtype=np.float32)
    lower_half = values[..., None] * 2.0
    upper_half = (values[..., None] - 0.5) * 2.0
    colors = np.where(
        values[..., None] <= 0.5,
        low * (1.0 - lower_half) + medium * lower_half,
        medium * (1.0 - upper_half) + high * upper_half,
    )
    colors[blocked] = (190, 40, 220)
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


def draw_candidate_routes(
    image: Image.Image,
    candidates: Sequence[RouteCandidate],
    best_route: RouteCandidate,
) -> Image.Image:
    """Draw blue alternatives and the selected candidate in bright green."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for candidate in candidates:
        points = [tuple(point) for point in candidate.points]
        if candidate.index == best_route.index:
            draw.line(points, fill=(45, 255, 100), width=5)
            end_x, end_y = points[-1]
            radius = 6
            draw.ellipse(
                (end_x - radius, end_y - radius, end_x + radius, end_y + radius),
                fill=(45, 255, 100),
            )
        else:
            draw.line(points, fill=(50, 135, 255), width=2)
    return canvas


def _render_panels(
    panels: Sequence[tuple[str, Image.Image]],
    panel_size: tuple[int, int],
) -> Image.Image:
    panel_width, panel_height = panel_size
    label_height = 22
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


def render_dashboard(
    image: Image.Image,
    semantic_mask: Image.Image,
    overlay: Image.Image,
    confidence: Image.Image,
    panel_size: tuple[int, int] = (512, 320),
) -> Image.Image:
    """Build a labeled 2-by-2 frame suitable for semantic-only playback."""
    return _render_panels(
        [
            ("Camera image", image),
            ("Semantic mask", semantic_mask),
            ("Semantic overlay", overlay),
            ("Prediction confidence", confidence),
        ],
        panel_size,
    )


def render_route_dashboard(
    route_overlay: Image.Image,
    semantic_overlay_image: Image.Image,
    traversability: Image.Image,
    confidence: Image.Image,
    panel_size: tuple[int, int] = (512, 320),
) -> Image.Image:
    """Build a route-planning dashboard from one camera-frame prediction."""
    return _render_panels(
        [
            ("Candidate routes (green = selected)", route_overlay),
            ("Semantic overlay", semantic_overlay_image),
            ("Traversability cost (purple = blocked)", traversability),
            ("Prediction confidence", confidence),
        ],
        panel_size,
    )
