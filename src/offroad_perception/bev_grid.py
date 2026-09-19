"""Build a small local metric traversability grid from painted LiDAR points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import LidarToCameraTransform


@dataclass(frozen=True)
class GridSpec:
    """Forward/lateral bounds and resolution for a local vehicle grid."""

    forward_min_m: float = 0.0
    forward_max_m: float = 20.0
    lateral_min_m: float = -10.0
    lateral_max_m: float = 10.0
    resolution_m: float = 0.20

    def __post_init__(self) -> None:
        if self.forward_max_m <= self.forward_min_m:
            raise ValueError("forward_max_m must be greater than forward_min_m")
        if self.lateral_max_m <= self.lateral_min_m:
            raise ValueError("lateral_max_m must be greater than lateral_min_m")
        if self.resolution_m <= 0:
            raise ValueError("resolution_m must be positive")

    @property
    def forward_cells(self) -> int:
        return int(np.ceil((self.forward_max_m - self.forward_min_m) / self.resolution_m))

    @property
    def lateral_cells(self) -> int:
        return int(np.ceil((self.lateral_max_m - self.lateral_min_m) / self.resolution_m))


@dataclass(frozen=True)
class TraversabilityGrid:
    """Metric grid used by the local route planner."""

    route_cost: np.ndarray
    blocked: np.ndarray
    observed: np.ndarray
    point_count: np.ndarray
    height_range_m: np.ndarray
    forward_axis: np.ndarray
    right_axis: np.ndarray
    spec: GridSpec

    @property
    def shape(self) -> tuple[int, int]:
        return self.route_cost.shape

    @property
    def start_cell(self) -> tuple[int, int]:
        forward_index = int((-self.spec.forward_min_m) / self.spec.resolution_m)
        lateral_index = int((-self.spec.lateral_min_m) / self.spec.resolution_m)
        return (
            int(np.clip(forward_index, 0, self.shape[0] - 1)),
            int(np.clip(lateral_index, 0, self.shape[1] - 1)),
        )

    def cell_center(self, cell: tuple[int, int]) -> tuple[float, float]:
        """Return a cell center as (forward meters, right meters)."""
        forward_index, lateral_index = cell
        return (
            self.spec.forward_min_m + (forward_index + 0.5) * self.spec.resolution_m,
            self.spec.lateral_min_m + (lateral_index + 0.5) * self.spec.resolution_m,
        )


def vehicle_axes_from_calibration(
    lidar_to_camera: LidarToCameraTransform,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive forward/right axes in LiDAR coordinates from camera axes."""
    camera_forward = np.asarray([0.0, 0.0, 1.0])
    camera_right = np.asarray([1.0, 0.0, 0.0])
    forward = lidar_to_camera.rotation.T @ camera_forward
    right = lidar_to_camera.rotation.T @ camera_right
    return forward / np.linalg.norm(forward), right / np.linalg.norm(right)


def build_traversability_grid(
    points_lidar: np.ndarray,
    route_risk: np.ndarray,
    blocked: np.ndarray,
    lidar_to_camera: LidarToCameraTransform,
    *,
    spec: GridSpec = GridSpec(),
    roughness_scale_m: float = 0.50,
    roughness_weight: float = 0.20,
) -> TraversabilityGrid:
    """Rasterize painted points into a conservative local metric cost grid.

    Empty cells remain high-cost but traversable so the planner can cross sparse
    LiDAR regions. Any obstacle-labeled point blocks its cell. Height variation
    adds a roughness penalty to observed cells.
    """
    points_lidar = np.asarray(points_lidar, dtype=np.float32)
    route_risk = np.asarray(route_risk, dtype=np.float32)
    blocked = np.asarray(blocked, dtype=bool)
    if points_lidar.ndim != 2 or points_lidar.shape[1] != 3:
        raise ValueError("points_lidar must have shape (N, 3)")
    if len(route_risk) != len(points_lidar) or len(blocked) != len(points_lidar):
        raise ValueError("points_lidar, route_risk, and blocked must have equal lengths")
    if roughness_scale_m <= 0:
        raise ValueError("roughness_scale_m must be positive")
    if not 0.0 <= roughness_weight <= 1.0:
        raise ValueError("roughness_weight must be between zero and one")

    forward_axis, right_axis = vehicle_axes_from_calibration(lidar_to_camera)
    forward = points_lidar @ forward_axis
    lateral = points_lidar @ right_axis
    forward_index = np.floor(
        (forward - spec.forward_min_m) / spec.resolution_m
    ).astype(np.intp)
    lateral_index = np.floor(
        (lateral - spec.lateral_min_m) / spec.resolution_m
    ).astype(np.intp)
    in_bounds = (
        (forward_index >= 0)
        & (forward_index < spec.forward_cells)
        & (lateral_index >= 0)
        & (lateral_index < spec.lateral_cells)
        & np.isfinite(route_risk)
    )
    forward_index = forward_index[in_bounds]
    lateral_index = lateral_index[in_bounds]
    point_z = points_lidar[in_bounds, 2]
    point_risk = np.clip(route_risk[in_bounds], 0.0, 1.0)
    point_blocked = blocked[in_bounds]

    shape = (spec.forward_cells, spec.lateral_cells)
    point_count = np.zeros(shape, dtype=np.int32)
    np.add.at(point_count, (forward_index, lateral_index), 1)
    observed = point_count > 0

    route_cost = np.ones(shape, dtype=np.float32)
    np.minimum.at(route_cost, (forward_index, lateral_index), point_risk)
    blocked_grid = np.zeros(shape, dtype=bool)
    np.logical_or.at(blocked_grid, (forward_index, lateral_index), point_blocked)

    minimum_z = np.full(shape, np.inf, dtype=np.float32)
    maximum_z = np.full(shape, -np.inf, dtype=np.float32)
    np.minimum.at(minimum_z, (forward_index, lateral_index), point_z)
    np.maximum.at(maximum_z, (forward_index, lateral_index), point_z)
    height_range = np.where(observed, maximum_z - minimum_z, 0.0).astype(np.float32)
    roughness = np.clip(height_range / roughness_scale_m, 0.0, 1.0)
    route_cost = np.clip(route_cost + roughness_weight * roughness, 0.0, 1.0)
    route_cost[blocked_grid] = 1.0

    return TraversabilityGrid(
        route_cost=route_cost,
        blocked=blocked_grid,
        observed=observed,
        point_count=point_count,
        height_range_m=height_range,
        forward_axis=forward_axis.astype(np.float32),
        right_axis=right_axis.astype(np.float32),
        spec=spec,
    )


def save_traversability_grid(grid: TraversabilityGrid, path: Path) -> None:
    """Save a grid in a portable NPZ format for planning and inspection."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        route_cost=grid.route_cost,
        blocked=grid.blocked,
        observed=grid.observed,
        point_count=grid.point_count,
        height_range_m=grid.height_range_m,
        forward_axis=grid.forward_axis,
        right_axis=grid.right_axis,
        forward_min_m=grid.spec.forward_min_m,
        forward_max_m=grid.spec.forward_max_m,
        lateral_min_m=grid.spec.lateral_min_m,
        lateral_max_m=grid.spec.lateral_max_m,
        resolution_m=grid.spec.resolution_m,
    )
