"""Timing helpers for single-image terrain segmentation benchmarks."""

from __future__ import annotations

from collections.abc import Callable
import time
from typing import Any, TypeVar

import numpy as np
from PIL import Image
import torch
from torch import nn


T = TypeVar("T")


def summarize_timings(samples_ms: list[float]) -> dict[str, float]:
    """Return stable latency statistics for one pipeline stage."""

    if not samples_ms:
        raise ValueError("at least one timing sample is required")

    values = np.asarray(samples_ms, dtype=np.float64)
    return {
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "minimum_ms": float(values.min()),
        "maximum_ms": float(values.max()),
    }


def make_reference_image(width: int = 1920, height: int = 1200) -> Image.Image:
    """Create a repeatable RGB image without including disk I/O in timing."""

    x_values = np.linspace(0, 255, width, dtype=np.uint8)
    y_values = np.linspace(0, 255, height, dtype=np.uint8)
    red = np.broadcast_to(x_values, (height, width))
    green = np.broadcast_to(y_values[:, None], (height, width))
    blue = ((red.astype(np.uint16) + green.astype(np.uint16)) // 2).astype(np.uint8)
    return Image.fromarray(np.stack((red, green, blue), axis=-1), mode="RGB")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_stage(
    callback: Callable[[], T],
    device: torch.device,
) -> tuple[T, float]:
    _synchronize(device)
    start = time.perf_counter()
    result = callback()
    _synchronize(device)
    return result, (time.perf_counter() - start) * 1000.0


def _prepare_image(image: Image.Image, image_size: tuple[int, int]) -> torch.Tensor:
    """Match the RGB scaling used by the existing RELLIS predictor."""

    height, width = image_size
    resized = image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def benchmark_rgb_to_mask(
    model: nn.Module,
    device: torch.device,
    *,
    image_size: tuple[int, int] = (320, 512),
    source_image: Image.Image | None = None,
    warmup_iterations: int = 50,
    benchmark_iterations: int = 300,
) -> dict[str, Any]:
    """Measure a serialized batch-one RGB-to-mask inference path.

    Disk I/O is intentionally excluded. The reported total uses CUDA
    synchronization, so it represents completed single-frame latency rather
    than asynchronous kernel launch time.
    """

    if warmup_iterations < 0:
        raise ValueError("warmup_iterations cannot be negative")
    if benchmark_iterations < 1:
        raise ValueError("benchmark_iterations must be at least one")

    source = source_image or make_reference_image()
    model.eval()

    def run_once() -> dict[str, float]:
        _synchronize(device)
        total_start = time.perf_counter()

        cpu_tensor, preprocessing_ms = _time_stage(
            lambda: _prepare_image(source, image_size),
            device,
        )
        tensor, transfer_ms = _time_stage(
            lambda: cpu_tensor.to(device, non_blocking=device.type == "cuda"),
            device,
        )

        with torch.inference_mode():
            logits, inference_ms = _time_stage(lambda: model(tensor), device)

            def postprocess() -> tuple[torch.Tensor, torch.Tensor]:
                probabilities = torch.softmax(logits, dim=1)
                confidence, class_ids = probabilities.max(dim=1)
                return class_ids.cpu(), confidence.cpu()

            _, postprocessing_ms = _time_stage(postprocess, device)

        _synchronize(device)
        total_ms = (time.perf_counter() - total_start) * 1000.0
        return {
            "preprocessing": preprocessing_ms,
            "cpu_to_device": transfer_ms,
            "model_inference": inference_ms,
            "mask_postprocessing": postprocessing_ms,
            "rgb_to_mask": total_ms,
        }

    for _ in range(warmup_iterations):
        run_once()

    samples: dict[str, list[float]] = {
        "preprocessing": [],
        "cpu_to_device": [],
        "model_inference": [],
        "mask_postprocessing": [],
        "rgb_to_mask": [],
    }
    for _ in range(benchmark_iterations):
        timings = run_once()
        for name, value in timings.items():
            samples[name].append(value)

    latency = {name: summarize_timings(values) for name, values in samples.items()}
    return {
        "benchmark_configuration": {
            "batch_size": 1,
            "input_shape": [1, 3, image_size[0], image_size[1]],
            "source_image_shape": [source.height, source.width, 3],
            "warmup_iterations": warmup_iterations,
            "benchmark_iterations": benchmark_iterations,
            "disk_io_included": False,
        },
        "latency": latency,
        "throughput": {
            "model_only_fps": 1000.0 / latency["model_inference"]["mean_ms"],
            "rgb_to_mask_fps": 1000.0 / latency["rgb_to_mask"]["mean_ms"],
        },
    }
