import numpy as np

from offroad_perception.traversability import build_traversability_map


def test_obstacle_probability_creates_high_cost_and_blocked_pixel() -> None:
    probabilities = np.zeros((5, 2, 3), dtype=np.float32)
    probabilities[0] = 1.0
    probabilities[:, 0, 1] = 0.0
    probabilities[3, 0, 1] = 0.8
    probabilities[0, 0, 1] = 0.2

    result = build_traversability_map(
        probabilities,
        (1.0, 4.0, 6.0, 100.0, 20.0),
        obstacle_class_id=3,
    )

    assert result.blocked[0, 1]
    assert result.route_cost[0, 1] > result.route_cost[1, 1]
    assert result.obstacle_probability[0, 1] == 0.8
