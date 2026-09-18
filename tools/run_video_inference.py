#!/usr/bin/env python3
"""Create a semantic-terrain playback from an ordered image directory."""

import argparse
from pathlib import Path

from offroad_perception.data import load_taxonomy
from offroad_perception.inference import (
    SegmentationPredictor,
    load_compact_unet_checkpoint,
    resolve_device,
)
from offroad_perception.video import discover_image_paths, run_playback


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / "outputs/checkpoints/rellis-epoch8-baseline/best_compact_unet.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/video_inference",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-frames", type=int, default=40)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--image-height", type=int, default=320)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_frames <= 0:
        raise ValueError("max-frames must be positive")
    if args.stride <= 0:
        raise ValueError("stride must be positive")

    taxonomy = load_taxonomy(REPO_ROOT / "configs/taxonomy.yaml")
    device = resolve_device(args.device)
    model, checkpoint = load_compact_unet_checkpoint(
        args.checkpoint,
        device,
        num_classes=taxonomy.num_classes,
        base_channels=args.base_channels,
    )
    predictor = SegmentationPredictor(
        model,
        device,
        image_size=(args.image_height, args.image_width),
    )

    paths = discover_image_paths(args.input_dir)
    selected = paths[::args.stride][:args.max_frames]
    print(f"Device: {device}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Frames selected: {len(selected)}/{len(paths)}")

    summary = run_playback(
        predictor,
        selected,
        taxonomy.names,
        args.output_dir,
        fps=args.fps,
    )
    print(f"Mean inference: {summary.mean_inference_ms:.1f} ms/frame")
    print(f"Mean end-to-end rate: {summary.mean_fps:.2f} FPS")
    print(f"Saved playback: {summary.gif_path}")
    print(f"Saved benchmark: {args.output_dir / 'benchmark.json'}")


if __name__ == "__main__":
    main()
