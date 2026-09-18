"""Helpers for running perception over an ordered directory of RGB frames."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

from PIL import Image

from .inference import SegmentationPredictor
from .visualization import (
    colorize_classes,
    confidence_heatmap,
    render_dashboard,
    semantic_overlay,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class PlaybackSummary:
    frame_count: int
    mean_load_ms: float
    mean_inference_ms: float
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
    output_dir: str | Path,
    *,
    fps: float = 8.0,
    panel_size: tuple[int, int] = (512, 320),
) -> PlaybackSummary:
    """Infer a sequence, write dashboard PNGs, and combine them into a GIF."""
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
    total_times: list[float] = []

    for index, image_path in enumerate(image_paths, start=1):
        total_start = time.perf_counter()
        load_start = time.perf_counter()
        with Image.open(image_path) as loaded:
            image = loaded.convert("RGB")
        load_times.append(time.perf_counter() - load_start)

        inference_start = time.perf_counter()
        prediction = predictor.predict_image(image)
        inference_times.append(time.perf_counter() - inference_start)

        mask = colorize_classes(prediction.class_ids, class_names)
        overlay = semantic_overlay(image, mask)
        confidence = confidence_heatmap(prediction.confidence)
        dashboard = render_dashboard(
            image,
            mask,
            overlay,
            confidence,
            panel_size=panel_size,
        )
        dashboard.save(frame_dir / f"frame_{index:04d}.png")
        dashboards.append(dashboard)
        total_times.append(time.perf_counter() - total_start)
        print(f"[{index}/{len(image_paths)}] {image_path.name}: {total_times[-1] * 1000:.1f} ms")

    gif_path = destination / "terrain_playback.gif"
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
        mean_total_ms=1000 * mean_total,
        mean_fps=1 / mean_total,
        gif_path=str(gif_path),
    )
    with (destination / "benchmark.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(summary), handle, indent=2)
    return summary
