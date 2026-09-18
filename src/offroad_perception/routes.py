"""Simple image-space candidate-route generation and scoring."""

from dataclasses import dataclass

import numpy as np

from .traversability import TraversabilityMap


@dataclass(frozen=True)
class RouteCandidate:
    """One curved image-space route and the cost used to rank it."""

    index: int
    lateral_offset_px: float
    points: np.ndarray
    mean_cost: float = 0.0
    obstacle_fraction: float = 0.0
    score: float = 0.0


def generate_candidate_routes(
    image_shape: tuple[int, int],
    *,
    count: int = 7,
    lookahead_fraction: float = 0.45,
    points_per_route: int = 80,
) -> list[RouteCandidate]:
    """Generate smooth paths from the lower image center toward the horizon.

    These are visual, image-space route candidates. They are not metric vehicle
    trajectories because this prototype does not estimate camera geometry.
    """
    height, width = image_shape
    if count < 3 or count % 2 == 0:
        raise ValueError("count must be an odd number of at least three")
    if not 0.15 <= lookahead_fraction <= 0.85:
        raise ValueError("lookahead_fraction must be between 0.15 and 0.85")
    if points_per_route < 2:
        raise ValueError("points_per_route must be at least two")

    center_x = (width - 1) / 2
    start = np.asarray((center_x, height - 1), dtype=np.float32)
    end_y = height * (1.0 - lookahead_fraction)
    offsets = np.linspace(-0.32 * width, 0.32 * width, count)
    sample_times = np.linspace(0.0, 1.0, points_per_route, dtype=np.float32)[:, None]
    candidates: list[RouteCandidate] = []

    for index, offset in enumerate(offsets):
        end = np.asarray((center_x + offset, end_y), dtype=np.float32)
        control = np.asarray((center_x + 0.20 * offset, height * 0.78), dtype=np.float32)
        curve = (
            (1.0 - sample_times) ** 2 * start
            + 2.0 * (1.0 - sample_times) * sample_times * control
            + sample_times**2 * end
        )
        curve[:, 0] = np.clip(curve[:, 0], 0, width - 1)
        curve[:, 1] = np.clip(curve[:, 1], 0, height - 1)
        candidates.append(
            RouteCandidate(
                index=index,
                lateral_offset_px=float(offset),
                points=curve,
            )
        )
    return candidates


def score_candidate_routes(
    candidates: list[RouteCandidate],
    traversability: TraversabilityMap,
    *,
    ribbon_radius_px: int = 4,
    curvature_penalty: float = 0.06,
) -> list[RouteCandidate]:
    """Score each route using a narrow ribbon of terrain and obstacle costs."""
    if ribbon_radius_px < 0:
        raise ValueError("ribbon_radius_px must be non-negative")
    height, width = traversability.route_cost.shape
    scored: list[RouteCandidate] = []

    for candidate in candidates:
        x = np.rint(candidate.points[:, 0]).astype(np.intp)
        y = np.rint(candidate.points[:, 1]).astype(np.intp)
        sampled_costs: list[np.ndarray] = []
        sampled_blocks: list[np.ndarray] = []

        for horizontal_offset in range(-ribbon_radius_px, ribbon_radius_px + 1):
            ribbon_x = np.clip(x + horizontal_offset, 0, width - 1)
            sampled_costs.append(traversability.route_cost[y, ribbon_x])
            sampled_blocks.append(traversability.blocked[y, ribbon_x])

        mean_cost = float(np.concatenate(sampled_costs).mean())
        obstacle_fraction = float(np.concatenate(sampled_blocks).mean())
        normalized_curvature = abs(candidate.lateral_offset_px) / max(width * 0.32, 1.0)
        score = mean_cost + 1.5 * obstacle_fraction + curvature_penalty * normalized_curvature
        scored.append(
            RouteCandidate(
                index=candidate.index,
                lateral_offset_px=candidate.lateral_offset_px,
                points=candidate.points,
                mean_cost=mean_cost,
                obstacle_fraction=obstacle_fraction,
                score=float(score),
            )
        )
    return scored


def select_best_route(candidates: list[RouteCandidate]) -> RouteCandidate:
    """Return the least-cost candidate, preferring a straighter route on ties."""
    if not candidates:
        raise ValueError("At least one route candidate is required")
    return min(
        candidates,
        key=lambda candidate: (candidate.score, abs(candidate.lateral_offset_px)),
    )
