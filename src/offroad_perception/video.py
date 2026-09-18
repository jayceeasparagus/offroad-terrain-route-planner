"""Helpers for running perception and route scoring over RGB frame sequences."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

from PIL import Image

from .inference import SegmentationPredictor
from .routes import generate_candidate_routes, score_candidate_routes, select_best_route
from .traversability import build_traversability_map
from .visualization import (
    colorize_classes,
    confidence_heatmap,
    draw_candidate_routes,
    render_route_dashboard,
    semantic_overlay,
    traversability_heatmap,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class PlaybackSummary:
    frame_count: int
    mean_load_ms: float
    mean_inference_ms: float
    mean_planning_ms: float
    mean_total_ms: float
    mean_fps: float
    gif_path: str


def discover_image_paths(input_dir: str | Path) -> list[Path]:
    """Return direct image children in filename order."""
    directory = Path(input_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    paths = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"No supported image files found in: {directory}")
    return paths


def run_playback(
    predictor: SegmentationPredictor,
    image_paths: list[Path],
    class_names: tuple[str, ...],
    class_costs: tuple[float, ...],
    output_dir: str | Path,
    *,
    obstacle_class_id: int,
    route_count: int = 7,
    fps: float = 8.0,
    panel_size: tuple[int, int] = (512, 320),
) -> PlaybackSummary:
    """Infer, score visual route candidates, and write an annotated GIF."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not image_paths:
        raise ValueError("image_paths must not be empty")

    destination = Path(output_dir)
    frame_dir = destination / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    dashboards: list[Image.Image] = []
    load_times: list[float] = []
    inference_times: list[float] = []
    planning_times: list[float] = []
    total_times: list[float] = []
    route_records: list[dict[str, float | int | str]] = []

    for index, image_path in enumerate(image_paths, start=1):
        total_start = time.perf_counter()
        load_start = time.perf_counter()
        with Image.open(image_path) as loaded:
            image = loaded.convert("RGB")
        load_times.append(time.perf_counter() - load_start)

        inference_start = time.perf_counter()
        prediction = predictor.predict_image(image)
        inference_times.append(time.perf_counter() - inference_start)

        planning_start = time.perf_counter()
        traversability = build_traversability_map(
            prediction.probabilities,
            class_costs,
            obstacle_class_id=obstacle_class_id,
        )
        candidates = generate_candidate_routes(prediction.class_ids.shape, count=route_count)
        scored_routes = score_candidate_routes(candidates, traversability)
        best_route = select_best_route(scored_routes)
        planning_times.append(time.perf_counter() - planning_start)

        mask = colorize_classes(prediction.class_ids, class_names)
        terrain_overlay = semantic_overlay(image, mask)
        route_overlay = draw_candidate_routes(image, scored_routes, best_route)
        confidence = confidence_heatmap(prediction.confidence)
        cost_map = traversability_heatmap(
            traversability.route_cost,
            traversability.blocked,
        )
        dashboard = render_route_dashboard(
            route_overlay,
            terrain_overlay,
            cost_map,
            confidence,
            panel_size=panel_size,
        )
        dashboard.save(frame_dir / f"frame_{index:04d}.png")
        dashboards.append(dashboard)
        total_times.append(time.perf_counter() - total_start)
        route_records.append(
            {
                "frame": image_path.name,
                "candidate_index": best_route.index,
                "lateral_offset_px": best_route.lateral_offset_px,
                "mean_cost": best_route.mean_cost,
                "obstacle_fraction": best_route.obstacle_fraction,
                "score": best_route.score,
            }
        )
        print(f"[{index}/{len(image_paths)}] {image_path.name}: {total_times[-1] * 1000:.1f} ms")

    gif_path = destination / "terrain_route_playback.gif"
    duration_ms = max(1, round(1000 / fps))
    dashboards[0].save(
        gif_path,
        save_all=True,
        append_images=dashboards[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    dashboards[0].save(destination / "preview.png")

    mean_total = sum(total_times) / len(total_times)
    summary = PlaybackSummary(
        frame_count=len(image_paths),
        mean_load_ms=1000 * sum(load_times) / len(load_times),
        mean_inference_ms=1000 * sum(inference_times) / len(inference_times),
        mean_planning_ms=1000 * sum(planning_times) / len(planning_times),
        mean_total_ms=1000 * mean_total,
        mean_fps=1 / mean_total,
        gif_path=str(gif_path),
    )
    with (destination / "benchmark.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(summary), handle, indent=2)
    with (destination / "route_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(route_records, handle, indent=2)
    return summary
