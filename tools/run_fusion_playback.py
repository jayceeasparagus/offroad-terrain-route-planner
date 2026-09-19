#!/usr/bin/env python3
"""Run the complete RGB/LiDAR semantic route planner over an extracted sample."""

from __future__ import annotations

import argparse
from pathlib import Path

from offroad_perception.bev_grid import GridSpec
from offroad_perception.calibration import CameraCalibration, load_rellis_lidar_to_camera_transform
from offroad_perception.data import load_taxonomy
from offroad_perception.fusion_playback import discover_extracted_pairs, run_fusion_playback
from offroad_perception.inference import SegmentationPredictor, load_compact_unet_checkpoint, resolve_device
from offroad_perception.planner import PlannerConfig


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--transform", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "outputs/checkpoints/rellis-epoch8-baseline/best_compact_unet.pt")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/fusion_playback")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--goal-min-forward-m", type=float, default=8.0)
    parser.add_argument("--goal-max-forward-m", type=float, default=18.0)
    args = parser.parse_args()
    if args.max_frames <= 0 or args.stride <= 0:
        raise ValueError("max-frames and stride must be positive")

    taxonomy = load_taxonomy(REPO_ROOT / "configs/taxonomy.yaml")
    device = resolve_device(args.device)
    model, checkpoint = load_compact_unet_checkpoint(
        args.checkpoint,
        device,
        num_classes=taxonomy.num_classes,
    )
    predictor = SegmentationPredictor(model, device)
    pairs = discover_extracted_pairs(args.data_root)
    selected = pairs[::args.stride][:args.max_frames]
    camera = CameraCalibration.from_json(args.data_root / "camera_intrinsics.json")
    transform = load_rellis_lidar_to_camera_transform(args.transform)
    print(f"Device: {device}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Frames selected: {len(selected)}/{len(pairs)}")

    summary = run_fusion_playback(
        predictor,
        selected,
        camera,
        transform,
        taxonomy.names,
        taxonomy.costs,
        args.output_dir,
        obstacle_class_id=taxonomy.names.index("obstacle"),
        grid_spec=GridSpec(),
        planner_config=PlannerConfig(
            goal_min_forward_m=args.goal_min_forward_m,
            goal_max_forward_m=args.goal_max_forward_m,
        ),
        fps=args.fps,
    )
    print(f"Mean inference: {summary.mean_inference_ms:.1f} ms/frame")
    print(f"Mean fusion: {summary.mean_fusion_ms:.1f} ms/frame")
    print(f"Mean planning: {summary.mean_planning_ms:.1f} ms/frame")
    print(f"Mean end-to-end rate: {summary.mean_fps:.2f} FPS")
    print(f"Saved playback: {summary.gif_path}")


if __name__ == "__main__":
    main()
