"""Attach camera-segmentation terrain probabilities to projected LiDAR points."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from .calibration import ProjectionResult
from .inference import SegmentationPrediction


@dataclass(frozen=True)
class SemanticPointCloud:
    """Projected LiDAR observations enriched with terrain semantics and risk."""

    source_indices: np.ndarray
    points_lidar: np.ndarray
    points_camera: np.ndarray
    pixels: np.ndarray
    probabilities: np.ndarray
    class_ids: np.ndarray
    confidence: np.ndarray
    expected_cost: np.ndarray
    route_risk: np.ndarray
    blocked: np.ndarray


def sample_probabilities_at_pixels(
    probabilities: np.ndarray, pixels: np.ndarray
) -> np.ndarray:
    """Sample class probabilities at nearest pixel centers.

    The segmentation predictor already returns probabilities resized to the
    original camera resolution, so nearest-pixel sampling preserves the model's
    output while keeping point painting easy to inspect and reproduce.
    """
    if probabilities.ndim != 3:
        raise ValueError("probabilities must have shape [classes, height, width]")
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("pixels must have shape (N, 2)")

    _, height, width = probabilities.shape
    pixel_indices = np.rint(pixels).astype(np.intp)
    x = np.clip(pixel_indices[:, 0], 0, width - 1)
    y = np.clip(pixel_indices[:, 1], 0, height - 1)
    return probabilities[:, y, x].T.astype(np.float32, copy=False)


def paint_projected_points(
    points_lidar: np.ndarray,
    projection: ProjectionResult,
    prediction: SegmentationPrediction,
    class_costs: Sequence[float],
    *,
    obstacle_class_id: int,
    confidence_penalty: float = 0.20,
    obstacle_block_threshold: float = 0.35,
) -> SemanticPointCloud:
    """Give each in-image LiDAR point semantic probabilities and route risk."""
    points_lidar = np.asarray(points_lidar, dtype=np.float32)
    if points_lidar.ndim != 2 or points_lidar.shape[1] != 3:
        raise ValueError("points_lidar must have shape (N, 3)")
    if len(projection.source_indices) != len(projection.pixels):
        raise ValueError("projection source indices and pixels must have equal length")
    if len(projection.source_indices) != len(projection.camera_points):
        raise ValueError("projection source indices and camera points must have equal length")

    costs = np.asarray(class_costs, dtype=np.float32)
    if prediction.probabilities.shape[0] != len(costs):
        raise ValueError("class_costs must contain one value for every probability channel")
    if not 0 <= obstacle_class_id < len(costs):
        raise ValueError("obstacle_class_id is outside the probability channels")
    if not 0.0 <= confidence_penalty <= 1.0:
        raise ValueError("confidence_penalty must be between zero and one")
    if not 0.0 <= obstacle_block_threshold <= 1.0:
        raise ValueError("obstacle_block_threshold must be between zero and one")

    sampled = sample_probabilities_at_pixels(prediction.probabilities, projection.pixels)
    class_ids = sampled.argmax(axis=1).astype(np.uint8)
    confidence = sampled.max(axis=1).astype(np.float32)
    expected_cost = sampled @ costs
    normalized_cost = (expected_cost - costs.min()) / max(float(np.ptp(costs)), 1e-6)
    route_risk = np.clip(
        normalized_cost + confidence_penalty * (1.0 - confidence), 0.0, 1.0
    ).astype(np.float32)
    blocked = sampled[:, obstacle_class_id] >= obstacle_block_threshold

    return SemanticPointCloud(
        source_indices=projection.source_indices.astype(np.int64, copy=False),
        points_lidar=points_lidar[projection.source_indices],
        points_camera=projection.camera_points.astype(np.float32, copy=False),
        pixels=projection.pixels.astype(np.float32, copy=False),
        probabilities=sampled,
        class_ids=class_ids,
        confidence=confidence,
        expected_cost=expected_cost.astype(np.float32),
        route_risk=route_risk,
        blocked=blocked.astype(bool),
    )
