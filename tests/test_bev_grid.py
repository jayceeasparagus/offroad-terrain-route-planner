import numpy as np

from offroad_perception.bev_grid import (
    GridSpec,
    build_traversability_grid,
    vehicle_axes_from_calibration,
)
from offroad_perception.calibration import LidarToCameraTransform


def test_vehicle_axes_follow_camera_forward_and_right() -> None:
    transform = LidarToCameraTransform(np.eye(3), np.zeros(3))

    forward, right = vehicle_axes_from_calibration(transform)

    np.testing.assert_allclose(forward, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(right, [1.0, 0.0, 0.0])


def test_grid_marks_obstacle_and_penalizes_height_variation() -> None:
    # With identity axes, LiDAR z is camera forward and y is camera down.
    transform = LidarToCameraTransform(np.eye(3), np.zeros(3))
    points = np.array(
        [
            [1.0, 0.0, 1.0],
            [1.0, 0.3, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    grid = build_traversability_grid(
        points,
        route_risk=np.array([0.1, 0.2, 0.4]),
        blocked=np.array([False, True, False]),
        lidar_to_camera=transform,
        spec=GridSpec(forward_min_m=0, forward_max_m=4, lateral_min_m=-2, lateral_max_m=2, resolution_m=1),
    )

    assert grid.observed.sum() == 2
    assert grid.blocked[1, 3]
    np.testing.assert_allclose(grid.height_range_m[1, 3], 0.3, atol=1e-6)
    assert grid.route_cost[1, 3] == 1.0
