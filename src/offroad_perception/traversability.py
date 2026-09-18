"""Convert semantic probabilities into an image-space traversability cost."""

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np


@dataclass(frozen=True)
class TraversabilityMap:
    """Per-pixel route cost derived from class probabilities and confidence."""

    expected_cost: np.ndarray
    route_cost: np.ndarray
    obstacle_probability: np.ndarray
    uncertainty: np.ndarray
    blocked: np.ndarray


def build_traversability_map(
    probabilities: np.ndarray,
    class_costs: Sequence[float],
    *,
    obstacle_class_id: int,
    confidence_penalty: float = 0.20,
    obstacle_block_threshold: float = 0.35,
) -> TraversabilityMap:
    """Build a normalized route-cost image from model probabilities.

    The expected semantic cost keeps uncertain terrain from being treated as
    fully traversable. A pixel is additionally marked blocked when the model
    assigns enough probability to the obstacle class.
    """
    if probabilities.ndim != 3:
        raise ValueError("probabilities must have shape [classes, height, width]")
    if probabilities.shape[0] != len(class_costs):
        raise ValueError("class_costs must contain one value for each class")
    if not 0 <= obstacle_class_id < probabilities.shape[0]:
        raise ValueError("obstacle_class_id is outside the probability channels")
    if not 0.0 <= confidence_penalty <= 1.0:
        raise ValueError("confidence_penalty must be between zero and one")
    if not 0.0 <= obstacle_block_threshold <= 1.0:
        raise ValueError("obstacle_block_threshold must be between zero and one")

    cost_array = np.asarray(class_costs, dtype=np.float32)
    expected_cost = np.tensordot(cost_array, probabilities, axes=(0, 0))
    minimum = float(cost_array.min())
    maximum = float(cost_array.max())
    normalized_cost = (expected_cost - minimum) / max(maximum - minimum, 1e-6)
    confidence = probabilities.max(axis=0)
    uncertainty = 1.0 - confidence
    route_cost = np.clip(
        normalized_cost + confidence_penalty * uncertainty,
        0.0,
        1.0,
    ).astype(np.float32)
    obstacle_probability = probabilities[obstacle_class_id].astype(np.float32)
    blocked = obstacle_probability >= obstacle_block_threshold

    return TraversabilityMap(
        expected_cost=expected_cost.astype(np.float32),
        route_cost=route_cost,
        obstacle_probability=obstacle_probability,
        uncertainty=uncertainty.astype(np.float32),
        blocked=blocked,
    )
