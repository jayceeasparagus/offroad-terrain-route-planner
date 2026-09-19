#!/usr/bin/env python3
"""Paint a synchronized RELLIS LiDAR scan using Compact U-Net terrain predictions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from offroad_perception.calibration import (
    CameraCalibration,
    load_rellis_lidar_to_camera_transform,
    project_lidar_to_image,
)
from offroad_perception.data.taxonomy import load_taxonomy
from offroad_perception.inference import (
    SegmentationPredictor,
    load_compact_unet_checkpoint,
    resolve_device,
)
from offroad_perception.semantic_painting import paint_projected_points
from offroad_perception.visualization import draw_semantic_points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--lidar", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--transform", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, default=Path("configs/taxonomy.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visualization", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--draw-stride", type=int, default=3)
    args = parser.parse_args()

    device = resolve_device(args.device)
    taxonomy = load_taxonomy(args.taxonomy)
    model, checkpoint = load_compact_unet_checkpoint(
        args.checkpoint, device, num_classes=taxonomy.num_classes
    )
    predictor = SegmentationPredictor(model, device)
    prediction = predictor.predict_path(args.image)

    image = Image.open(args.image).convert("RGB")
    points = np.load(args.lidar)["points"]
    camera = CameraCalibration.from_json(args.intrinsics)
    projection = project_lidar_to_image(
        points, camera, load_rellis_lidar_to_camera_transform(args.transform)
    )
    obstacle_class_id = taxonomy.names.index("obstacle")
    painted = paint_projected_points(
        points,
        projection,
        prediction,
        taxonomy.costs,
        obstacle_class_id=obstacle_class_id,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        source_indices=painted.source_indices,
        points_lidar=painted.points_lidar,
        points_camera=painted.points_camera,
        pixels=painted.pixels,
        probabilities=painted.probabilities,
        class_ids=painted.class_ids,
        confidence=painted.confidence,
        expected_cost=painted.expected_cost,
        route_risk=painted.route_risk,
        blocked=painted.blocked,
    )
    args.visualization.parent.mkdir(parents=True, exist_ok=True)
    draw_semantic_points(
        image,
        painted.pixels,
        painted.class_ids,
        taxonomy.names,
        painted.confidence,
        draw_stride=args.draw_stride,
    ).save(args.visualization)

    class_counts = Counter(painted.class_ids.tolist())
    summary = {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "input_points": int(len(points)),
        "painted_points": int(len(painted.class_ids)),
        "blocked_points": int(painted.blocked.sum()),
        "mean_confidence": float(painted.confidence.mean()),
        "mean_route_risk": float(painted.route_risk.mean()),
        "class_counts": {
            taxonomy.names[class_id]: int(class_counts[class_id])
            for class_id in range(taxonomy.num_classes)
        },
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Checkpoint epoch: {summary['checkpoint_epoch']}")
    print(f"Painted points: {summary['painted_points']} / {summary['input_points']}")
    print(f"Blocked points: {summary['blocked_points']}")
    print(f"Mean confidence: {summary['mean_confidence']:.3f}")
    print(f"Saved point cloud: {args.output}")
    print(f"Saved overlay: {args.visualization}")


if __name__ == "__main__":
    main()
