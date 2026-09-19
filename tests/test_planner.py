import numpy as np

from offroad_perception.bev_grid import GridSpec, build_traversability_grid
from offroad_perception.calibration import LidarToCameraTransform
from offroad_perception.planner import PlannerConfig, plan_route


def make_grid(route_cost: np.ndarray, blocked: np.ndarray):
    transform = LidarToCameraTransform(np.eye(3), np.zeros(3))
    rows, columns = np.where(np.ones_like(route_cost, dtype=bool))
    # Identity calibration maps LiDAR z to forward and x to right.
    points = np.column_stack((columns.astype(float), np.zeros(len(rows)), rows + 0.5))
    return build_traversability_grid(
        points,
        route_risk=route_cost[rows, columns],
        blocked=blocked[rows, columns],
        lidar_to_camera=transform,
        spec=GridSpec(
            forward_min_m=0,
            forward_max_m=route_cost.shape[0],
            lateral_min_m=-2,
            lateral_max_m=route_cost.shape[1] - 2,
            resolution_m=1,
        ),
    )


def test_astar_routes_around_blocked_column() -> None:
    costs = np.zeros((8, 5), dtype=np.float32)
    blocked = np.zeros_like(costs, dtype=bool)
    blocked[1:7, 2] = True
    grid = make_grid(costs, blocked)

    route = plan_route(
        grid,
        config=PlannerConfig(
            goal_min_forward_m=6,
            goal_max_forward_m=7,
        ),
    )

    assert route.coordinates[-1, 0] >= 6.0
    assert all(not grid.blocked[cell] for cell in route.cells)
    assert any(cell[1] != 4 for cell in route.cells)


def test_astar_prefers_lower_risk_open_lane() -> None:
    costs = np.zeros((8, 5), dtype=np.float32)
    costs[:, 1] = 0.9
    blocked = np.zeros_like(costs, dtype=bool)
    grid = make_grid(costs, blocked)

    route = plan_route(
        grid,
        config=PlannerConfig(
            goal_min_forward_m=6,
            goal_max_forward_m=7,
        ),
    )

    assert all(cell[1] != 3 for cell in route.cells[2:-1])
