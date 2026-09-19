"""End-to-end RGB/LiDAR semantic route playback."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .bev_grid import GridSpec, build_traversability_grid
from .calibration import CameraCalibration, LidarToCameraTransform, project_lidar_to_image
from .inference import SegmentationPredictor
from .planner import PlannerConfig, plan_route, project_route_to_image
from .semantic_painting import paint_projected_points
from .visualization import (
    colorize_classes,
    confidence_heatmap,
    draw_semantic_points,
    semantic_overlay,
)


@dataclass(frozen=True)
class FusionPair:
    image_path: Path
    lidar_path: Path
    frame_id: str


@dataclass(frozen=True)
class FusionPlaybackSummary:
    frame_count: int
    mean_load_ms: float
    mean_inference_ms: float
    mean_fusion_ms: float
    mean_planning_ms: float
    mean_total_ms: float
    mean_fps: float
    gif_path: str


def discover_extracted_pairs(root: str | Path) -> list[FusionPair]:
    """Read image/LiDAR pairs from an extractor output directory."""
    root = Path(root)
    manifest_path = root / "frames.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Extractor manifest does not exist: {manifest_path}")
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    pairs: list[FusionPair] = []
    for record in records:
        image_path = root / record["image"]
        lidar_path = root / record["lidar"]
        if not image_path.is_file() or not lidar_path.is_file():
            raise FileNotFoundError(
                f"Manifest pair is missing files: {image_path}, {lidar_path}"
            )
        pairs.append(FusionPair(image_path, lidar_path, str(record["frame_id"])))
    if not pairs:
        raise ValueError(f"No frame pairs found in: {manifest_path}")
    return pairs


def _render_panels(
    panels: Sequence[tuple[str, Image.Image]],
    panel_size: tuple[int, int],
) -> Image.Image:
    panel_width, panel_height = panel_size
    label_height = 22
    dashboard = Image.new(
        "RGB", (panel_width * 2, (panel_height + label_height) * 2), "black"
    )
    draw = ImageDraw.Draw(dashboard)
    font = ImageFont.load_default()
    for index, (title, panel) in enumerate(panels):
        column = index % 2
        row = index // 2
        x = column * panel_width
        y = row * (panel_height + label_height)
        draw.text((x + 6, y + 5), title, fill="white", font=font)
        dashboard.paste(
            panel.convert("RGB").resize(
                (panel_width, panel_height), Image.Resampling.BILINEAR
            ),
            (x, y + label_height),
        )
    return dashboard


def _draw_route(image: Image.Image, pixels: np.ndarray, goal: tuple[float, float] | None) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    if len(pixels) >= 2:
        points = [tuple(pixel) for pixel in pixels]
        draw.line(points, fill=(45, 255, 100), width=8)
        end_x, end_y = points[-1]
        draw.ellipse((end_x - 10, end_y - 10, end_x + 10, end_y + 10), fill=(45, 255, 100))
    else:
        draw.text((12, 12), "No valid projected route", fill=(255, 80, 80))
    if goal is not None:
        draw.text((12, 36), f"Goal: {goal[0]:.1f}m ahead", fill=(45, 255, 100))
    return canvas


def run_fusion_playback(
    predictor: SegmentationPredictor,
    pairs: list[FusionPair],
    camera: CameraCalibration,
    lidar_to_camera: LidarToCameraTransform,
    class_names: tuple[str, ...],
    class_costs: tuple[float, ...],
    output_dir: str | Path,
    *,
    obstacle_class_id: int,
    grid_spec: GridSpec = GridSpec(),
    planner_config: PlannerConfig = PlannerConfig(),
    fps: float = 5.0,
    panel_size: tuple[int, int] = (512, 320),
) -> FusionPlaybackSummary:
    """Run the complete RGB/LiDAR route pipeline and write an annotated GIF."""
    if not pairs:
        raise ValueError("pairs must not be empty")
    if fps <= 0:
        raise ValueError("fps must be positive")

    destination = Path(output_dir)
    frame_dir = destination / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    dashboards: list[Image.Image] = []
    timings = {name: [] for name in ("load", "inference", "fusion", "planning", "total")}
    route_records: list[dict[str, object]] = []

    for index, pair in enumerate(pairs, start=1):
        total_start = time.perf_counter()
        load_start = time.perf_counter()
        with Image.open(pair.image_path) as loaded:
            image = loaded.convert("RGB")
        points = np.load(pair.lidar_path)["points"]
        timings["load"].append(time.perf_counter() - load_start)

        inference_start = time.perf_counter()
        prediction = predictor.predict_image(image)
        timings["inference"].append(time.perf_counter() - inference_start)

        fusion_start = time.perf_counter()
        projection = project_lidar_to_image(points, camera, lidar_to_camera)
        painted = paint_projected_points(
            points,
            projection,
            prediction,
            class_costs,
            obstacle_class_id=obstacle_class_id,
        )
        timings["fusion"].append(time.perf_counter() - fusion_start)

        planning_start = time.perf_counter()
        grid = build_traversability_grid(
            painted.points_lidar,
            painted.route_risk,
            painted.blocked,
            lidar_to_camera,
            spec=grid_spec,
        )
        route = None
        route_projection = np.empty((0, 2), dtype=np.float64)
        try:
            route = plan_route(grid, config=planner_config)
            _, projected_route = project_route_to_image(
                grid, route, camera, lidar_to_camera
            )
            route_projection = projected_route.pixels
        except RuntimeError:
            pass
        timings["planning"].append(time.perf_counter() - planning_start)

        mask = colorize_classes(prediction.class_ids, class_names)
        semantic_image = semantic_overlay(image, mask)
        painted_image = draw_semantic_points(
            image,
            painted.pixels,
            painted.class_ids,
            class_names,
            painted.confidence,
        )
        goal = None if route is None else tuple(route.coordinates[-1])
        route_image = _draw_route(image, route_projection, goal)
        dashboard = _render_panels(
            [
                ("A* route (green)", route_image),
                ("Semantic overlay", semantic_image),
                ("Painted LiDAR points", painted_image),
                ("Prediction confidence", confidence_heatmap(prediction.confidence)),
            ],
            panel_size,
        )
        dashboard.save(frame_dir / f"frame_{index:04d}.png")
        dashboards.append(dashboard)
        timings["total"].append(time.perf_counter() - total_start)
        route_records.append(
            {
                "frame_id": pair.frame_id,
                "painted_points": int(len(painted.class_ids)),
                "blocked_points": int(painted.blocked.sum()),
                "route_found": route is not None,
                "goal_forward_m": None if route is None else float(route.coordinates[-1, 0]),
                "goal_right_m": None if route is None else float(route.coordinates[-1, 1]),
                "expanded_nodes": None if route is None else route.expanded_nodes,
            }
        )
        print(f"[{index}/{len(pairs)}] {pair.frame_id}: {timings['total'][-1] * 1000:.1f} ms")

    gif_path = destination / "semantic_route_playback.gif"
    dashboards[0].save(
        gif_path,
        save_all=True,
        append_images=dashboards[1:],
        duration=max(1, round(1000 / fps)),
        loop=0,
        optimize=False,
    )
    dashboards[0].save(destination / "preview.png")

    means = {name: 1000 * float(np.mean(values)) for name, values in timings.items()}
    summary = FusionPlaybackSummary(
        frame_count=len(pairs),
        mean_load_ms=means["load"],
        mean_inference_ms=means["inference"],
        mean_fusion_ms=means["fusion"],
        mean_planning_ms=means["planning"],
        mean_total_ms=means["total"],
        mean_fps=1.0 / (means["total"] / 1000.0),
        gif_path=str(gif_path),
    )
    (destination / "benchmark.json").write_text(
        json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8"
    )
    (destination / "route_summary.json").write_text(
        json.dumps(route_records, indent=2) + "\n", encoding="utf-8"
    )
    return summary
