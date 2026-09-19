"""A* local planning over a vehicle-aligned traversability grid."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np

from .bev_grid import TraversabilityGrid
from .calibration import CameraCalibration, LidarToCameraTransform, ProjectionResult, project_lidar_to_image


@dataclass(frozen=True)
class PlannerConfig:
    """Small set of interpretable local-planner parameters."""

    goal_min_forward_m: float = 8.0
    goal_max_forward_m: float = 18.0
    terrain_weight: float = 4.0
    lateral_goal_penalty: float = 0.03
    max_expansions: int = 30_000


@dataclass(frozen=True)
class PlannedRoute:
    """A successful grid route and its metric coordinates."""

    cells: tuple[tuple[int, int], ...]
    coordinates: np.ndarray
    start_cell: tuple[int, int]
    goal_cell: tuple[int, int]
    total_cost: float
    expanded_nodes: int


def _neighbors(cell: tuple[int, int], shape: tuple[int, int]) -> Iterable[tuple[tuple[int, int], float]]:
    row, column = cell
    for row_delta in (-1, 0, 1):
        for column_delta in (-1, 0, 1):
            if row_delta == 0 and column_delta == 0:
                continue
            neighbor = (row + row_delta, column + column_delta)
            if 0 <= neighbor[0] < shape[0] and 0 <= neighbor[1] < shape[1]:
                yield neighbor, float(np.hypot(row_delta, column_delta))


def _nearest_unblocked_cell(
    grid: TraversabilityGrid, preferred: tuple[int, int], max_radius: int = 8
) -> tuple[int, int]:
    if not grid.blocked[preferred]:
        return preferred
    for radius in range(1, max_radius + 1):
        candidates = [
            (preferred[0] + dr, preferred[1] + dc)
            for dr in range(-radius, radius + 1)
            for dc in range(-radius, radius + 1)
            if max(abs(dr), abs(dc)) == radius
        ]
        for candidate in candidates:
            if (
                0 <= candidate[0] < grid.shape[0]
                and 0 <= candidate[1] < grid.shape[1]
                and not grid.blocked[candidate]
            ):
                return candidate
    raise RuntimeError("No unblocked start cell exists near the vehicle")


def choose_goal_cell(
    grid: TraversabilityGrid,
    *,
    config: PlannerConfig = PlannerConfig(),
) -> tuple[int, int]:
    """Choose the safest reachable-looking cell in a forward look-ahead band."""
    if config.goal_min_forward_m < 0 or config.goal_max_forward_m <= config.goal_min_forward_m:
        raise ValueError("goal forward bounds are invalid")
    forward_centers = grid.spec.forward_min_m + (
        np.arange(grid.shape[0]) + 0.5
    ) * grid.spec.resolution_m
    lateral_centers = grid.spec.lateral_min_m + (
        np.arange(grid.shape[1]) + 0.5
    ) * grid.spec.resolution_m
    forward_mask = (forward_centers >= config.goal_min_forward_m) & (
        forward_centers <= config.goal_max_forward_m
    )
    candidates = np.argwhere(forward_mask[:, None] & ~grid.blocked)
    if len(candidates) == 0:
        raise RuntimeError("No unblocked goal cell exists in the look-ahead band")

    rows = candidates[:, 0]
    columns = candidates[:, 1]
    # Prefer low risk, then centrality, then farther look-ahead progress.
    target_forward = (config.goal_min_forward_m + config.goal_max_forward_m) / 2.0
    score = (
        grid.route_cost[rows, columns]
        + config.lateral_goal_penalty * np.abs(lateral_centers[columns])
        + 0.01 * np.abs(forward_centers[rows] - target_forward)
    )
    best = int(np.argmin(score))
    return int(rows[best]), int(columns[best])


def _has_clear_diagonal(grid: TraversabilityGrid, current: tuple[int, int], neighbor: tuple[int, int]) -> bool:
    row_delta = neighbor[0] - current[0]
    column_delta = neighbor[1] - current[1]
    if abs(row_delta) != 1 or abs(column_delta) != 1:
        return True
    return not (
        grid.blocked[current[0] + row_delta, current[1]]
        or grid.blocked[current[0], current[1] + column_delta]
    )


def _astar(
    grid: TraversabilityGrid,
    start: tuple[int, int],
    goal: tuple[int, int],
    config: PlannerConfig,
) -> tuple[list[tuple[int, int]], float, int] | None:
    open_set: list[tuple[float, tuple[int, int]]] = []
    heapq.heappush(open_set, (0.0, start))
    g_score: dict[tuple[int, int], float] = {start: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    expanded = 0

    def heuristic(cell: tuple[int, int]) -> float:
        return float(np.hypot(goal[0] - cell[0], goal[1] - cell[1]))

    while open_set and expanded < config.max_expansions:
        _, current = heapq.heappop(open_set)
        expanded += 1
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, g_score[goal], expanded

        for neighbor, step_cells in _neighbors(current, grid.shape):
            if grid.blocked[neighbor] or not _has_clear_diagonal(grid, current, neighbor):
                continue
            terrain_cost = 0.5 * (
                float(grid.route_cost[current]) + float(grid.route_cost[neighbor])
            )
            step_cost = step_cells * (1.0 + config.terrain_weight * terrain_cost)
            tentative = g_score[current] + step_cost
            if tentative >= g_score.get(neighbor, float("inf")):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative
            heapq.heappush(open_set, (tentative + heuristic(neighbor), neighbor))
    return None


def _line_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    steps = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
    if steps == 0:
        return [start]
    rows = np.rint(np.linspace(start[0], end[0], steps + 1)).astype(int)
    columns = np.rint(np.linspace(start[1], end[1], steps + 1)).astype(int)
    return list(dict.fromkeys(zip(rows.tolist(), columns.tolist())))


def simplify_route(
    grid: TraversabilityGrid,
    cells: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Remove unnecessary zig-zags while preserving obstacle clearance."""
    if len(cells) <= 2:
        return cells
    simplified = [cells[0]]
    anchor_index = 0
    while anchor_index < len(cells) - 1:
        chosen_index = anchor_index + 1
        for candidate_index in range(anchor_index + 1, len(cells)):
            segment = _line_cells(cells[anchor_index], cells[candidate_index])
            if all(not grid.blocked[cell] for cell in segment):
                chosen_index = candidate_index
        simplified.append(cells[chosen_index])
        anchor_index = chosen_index
    return simplified


def plan_route(
    grid: TraversabilityGrid,
    *,
    config: PlannerConfig = PlannerConfig(),
) -> PlannedRoute:
    """Plan the lowest-risk collision-free route to a forward goal."""
    start = _nearest_unblocked_cell(grid, grid.start_cell)
    goal = choose_goal_cell(grid, config=config)
    result = _astar(grid, start, goal, config)
    if result is None:
        raise RuntimeError("A* could not find a route to the selected goal")
    cells, total_cost, expanded = result
    cells = simplify_route(grid, cells)
    coordinates = np.asarray([grid.cell_center(cell) for cell in cells], dtype=np.float32)
    return PlannedRoute(
        cells=tuple(cells),
        coordinates=coordinates,
        start_cell=start,
        goal_cell=goal,
        total_cost=float(total_cost),
        expanded_nodes=expanded,
    )


def route_to_lidar_points(
    grid: TraversabilityGrid,
    route: PlannedRoute,
    *,
    spacing_m: float = 0.50,
) -> np.ndarray:
    """Convert route (forward, right) coordinates into LiDAR-frame points."""
    if spacing_m <= 0:
        raise ValueError("spacing_m must be positive")
    route_heights = np.asarray(
        [grid.surface_height_m[cell] for cell in route.cells], dtype=np.float32
    )
    dense_coordinates = [route.coordinates[0]]
    dense_heights = [route_heights[0]]
    for index in range(1, len(route.coordinates)):
        previous = route.coordinates[index - 1]
        current = route.coordinates[index]
        distance = float(np.linalg.norm(current - previous))
        steps = max(int(np.ceil(distance / spacing_m)), 1)
        for fraction in np.linspace(0.0, 1.0, steps + 1)[1:]:
            dense_coordinates.append(previous * (1.0 - fraction) + current * fraction)
            dense_heights.append(
                route_heights[index - 1] * (1.0 - fraction)
                + route_heights[index] * fraction
            )
    coordinates = np.asarray(dense_coordinates, dtype=np.float32)
    heights = np.asarray(dense_heights, dtype=np.float32)
    return (
        coordinates[:, 0, None] * grid.forward_axis[None, :]
        + coordinates[:, 1, None] * grid.right_axis[None, :]
        + heights[:, None] * grid.down_axis[None, :]
    ).astype(np.float32)


def project_route_to_image(
    grid: TraversabilityGrid,
    route: PlannedRoute,
    camera: CameraCalibration,
    lidar_to_camera: LidarToCameraTransform,
) -> tuple[np.ndarray, ProjectionResult]:
    """Project the metric route back into the camera image."""
    lidar_points = route_to_lidar_points(grid, route)
    return lidar_points, project_lidar_to_image(lidar_points, camera, lidar_to_camera)
