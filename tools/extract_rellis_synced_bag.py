#!/usr/bin/env python3
"""Export a synchronized RELLIS camera/LiDAR sample from a ROS bag."""

from __future__ import annotations

import argparse
from pathlib import Path

from offroad_perception.rellis_bag import extract_synced_rellis_bag


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True, help="Input RELLIS .bag file")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=60,
        help="Number of synchronized pairs to export; use 0 for the whole bag",
    )
    parser.add_argument("--pair-tolerance-ms", type=float, default=50.0)
    args = parser.parse_args()

    max_frames = None if args.max_frames == 0 else args.max_frames
    summary = extract_synced_rellis_bag(
        args.bag,
        args.output_dir,
        max_frames=max_frames,
        pair_tolerance_ms=args.pair_tolerance_ms,
    )
    print(f"Extracted {summary['frames']} synchronized camera/LiDAR pairs")
    print(f"Maximum image-to-LiDAR offset: {summary['max_pair_offset_ms']:.2f} ms")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
