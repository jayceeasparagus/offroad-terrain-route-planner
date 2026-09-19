#!/usr/bin/env python3
"""Build a hidden local metric traversability grid from painted points."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from offroad_perception.calibration import load_rellis_lidar_to_camera_transform
from offroad_perception.bev_grid import (
    GridSpec,
    build_traversability_grid,
    save_traversability_grid,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--painted", type=Path, required=True)
    parser.add_argument("--transform", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--forward-max-m", type=float, default=20.0)
    parser.add_argument("--lateral-half-width-m", type=float, default=10.0)
    parser.add_argument("--resolution-m", type=float, default=0.20)
    args = parser.parse_args()

    painted = np.load(args.painted)
    grid = build_traversability_grid(
        painted["points_lidar"],
        painted["route_risk"],
        painted["blocked"],
        load_rellis_lidar_to_camera_transform(args.transform),
        spec=GridSpec(
            forward_max_m=args.forward_max_m,
            lateral_min_m=-args.lateral_half_width_m,
            lateral_max_m=args.lateral_half_width_m,
            resolution_m=args.resolution_m,
        ),
    )
    save_traversability_grid(grid, args.output)
    print(f"Grid shape: {grid.shape}")
    print(f"Resolution: {grid.spec.resolution_m:.2f} m")
    print(f"Observed cells: {int(grid.observed.sum())} / {grid.observed.size}")
    print(f"Blocked cells: {int(grid.blocked.sum())}")
    print(f"Start cell: {grid.start_cell}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
