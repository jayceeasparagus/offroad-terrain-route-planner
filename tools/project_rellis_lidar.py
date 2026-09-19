#!/usr/bin/env python3
"""Create a diagnostic image of calibrated RELLIS LiDAR point projection."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from offroad_perception.calibration import (
    CameraCalibration,
    load_rellis_lidar_to_camera_transform,
    project_lidar_to_image,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--lidar", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--transform", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draw-stride", type=int, default=4)
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    points = np.load(args.lidar)["points"]
    camera = CameraCalibration.from_json(args.intrinsics)
    if image.size != (camera.width, camera.height):
        raise ValueError(
            f"Image size {image.size} does not match intrinsics "
            f"{(camera.width, camera.height)}"
        )

    result = project_lidar_to_image(
        points, camera, load_rellis_lidar_to_camera_transform(args.transform)
    )
    output = image.copy()
    draw = ImageDraw.Draw(output)
    distances = np.linalg.norm(result.camera_points, axis=1)
    max_distance = max(float(distances.max()), 1.0) if len(distances) else 1.0
    for pixel, distance in zip(result.pixels[:: args.draw_stride], distances[:: args.draw_stride]):
        ratio = min(float(distance) / max_distance, 1.0)
        color = (255, int(255 * ratio), 0)
        x, y = pixel
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.save(args.output)
    print(f"Input points: {len(points)}")
    print(f"Projected in-bounds points: {len(result.source_indices)}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
