import numpy as np

from offroad_perception.routes import (
    generate_candidate_routes,
    score_candidate_routes,
    select_best_route,
)
from offroad_perception.traversability import TraversabilityMap


def test_route_scorer_avoids_central_blocked_region() -> None:
    height, width = 64, 96
    cost = np.zeros((height, width), dtype=np.float32)
    blocked = np.zeros((height, width), dtype=bool)
    blocked[18:48, 42:55] = True
    cost[blocked] = 1.0
    traversability = TraversabilityMap(
        expected_cost=cost,
        route_cost=cost,
        obstacle_probability=blocked.astype(np.float32),
        uncertainty=np.zeros_like(cost),
        blocked=blocked,
    )

    candidates = generate_candidate_routes((height, width), count=5)
    scored = score_candidate_routes(candidates, traversability)
    best = select_best_route(scored)

    assert best.index != 2
    assert best.obstacle_fraction < scored[2].obstacle_fraction
