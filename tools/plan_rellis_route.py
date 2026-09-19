#!/usr/bin/env python3
"""Run A* on a painted RELLIS traversability grid and overlay the route."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from offroad_perception.bev_grid import load_traversability_grid
from offroad_perception.calibration import CameraCalibration, load_rellis_lidar_to_camera_transform
from offroad_perception.planner import PlannerConfig, plan_route, project_route_to_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--transform", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route-output", type=Path, required=True)
    parser.add_argument("--goal-min-forward-m", type=float, default=8.0)
    parser.add_argument("--goal-max-forward-m", type=float, default=18.0)
    args = parser.parse_args()

    grid = load_traversability_grid(args.grid)
    route = plan_route(
        grid,
        config=PlannerConfig(
            goal_min_forward_m=args.goal_min_forward_m,
            goal_max_forward_m=args.goal_max_forward_m,
        ),
    )
    _, projection = project_route_to_image(
        grid,
        route,
        CameraCalibration.from_json(args.intrinsics),
        load_rellis_lidar_to_camera_transform(args.transform),
    )

    image = Image.open(args.image).convert("RGB")
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    if len(projection.pixels) >= 2:
        points = [tuple(pixel) for pixel in projection.pixels]
        draw.line(points, fill=(45, 255, 100), width=8)
        end_x, end_y = points[-1]
        draw.ellipse((end_x - 10, end_y - 10, end_x + 10, end_y + 10), fill=(45, 255, 100))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(args.output)
    route_points = np.asarray(route.coordinates, dtype=np.float32)
    np.savez_compressed(
        args.route_output,
        cells=np.asarray(route.cells, dtype=np.int32),
        coordinates=route_points,
        pixels=projection.pixels,
        source_indices=projection.source_indices,
    )
    summary = {
        "start_cell": list(route.start_cell),
        "goal_cell": list(route.goal_cell),
        "route_points": int(len(route.cells)),
        "projected_points": int(len(projection.pixels)),
        "total_cost": route.total_cost,
        "expanded_nodes": route.expanded_nodes,
        "goal_forward_m": float(route.coordinates[-1, 0]),
        "goal_right_m": float(route.coordinates[-1, 1]),
    }
    args.route_output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Route cells: {summary['route_points']}")
    print(f"Goal: {summary['goal_forward_m']:.2f} m forward, {summary['goal_right_m']:.2f} m right")
    print(f"Expanded nodes: {summary['expanded_nodes']}")
    print(f"Projected route points: {summary['projected_points']}")
    print(f"Saved overlay: {args.output}")
    print(f"Saved route: {args.route_output}")


if __name__ == "__main__":
    main()
