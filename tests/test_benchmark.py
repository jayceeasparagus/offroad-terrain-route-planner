from PIL import Image
import numpy as np
import torch

from offroad_perception.benchmark import (
    benchmark_rgb_to_mask,
    summarize_timings,
)
from offroad_perception.models import CompactUNet


def test_timing_summary_reports_expected_statistics() -> None:
    summary = summarize_timings([1.0, 2.0, 3.0])

    assert summary["mean_ms"] == 2.0
    assert summary["median_ms"] == 2.0
    assert summary["minimum_ms"] == 1.0
    assert summary["maximum_ms"] == 3.0


def test_cpu_rgb_to_mask_benchmark_reports_all_stages() -> None:
    model = CompactUNet(base_channels=4, num_classes=5)
    image = Image.fromarray(np.full((40, 64, 3), 120, dtype=np.uint8))

    result = benchmark_rgb_to_mask(
        model,
        torch.device("cpu"),
        image_size=(32, 48),
        source_image=image,
        warmup_iterations=0,
        benchmark_iterations=2,
    )

    assert result["benchmark_configuration"]["disk_io_included"] is False
    assert result["benchmark_configuration"]["input_shape"] == [1, 3, 32, 48]
    assert result["throughput"]["rgb_to_mask_fps"] > 0.0
    for stage in result["latency"].values():
        assert stage["mean_ms"] >= 0.0
