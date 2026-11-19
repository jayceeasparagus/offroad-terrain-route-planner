#!/usr/bin/env python3
"""Benchmark a RELLIS Compact U-Net from RGB preprocessing through mask output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from offroad_perception.benchmark import benchmark_rgb_to_mask
from offroad_perception.inference import (
    load_compact_unet_checkpoint,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=50)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_metadata(checkpoint: dict[str, object], key: str) -> object | None:
    value = checkpoint.get(key)
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return None


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    device = resolve_device(args.device)
    model, checkpoint = load_compact_unet_checkpoint(args.checkpoint, device)
    base_channels = int(model.enc1.block[0].weight.shape[0])
    result = benchmark_rgb_to_mask(
        model,
        device,
        warmup_iterations=args.warmup,
        benchmark_iterations=args.iterations,
    )

    result.update(
        {
            "benchmark_name": "rellis_compact_unet_fp32_baseline",
            "hardware": {
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None,
                "pytorch": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "model": {
                "architecture": "CompactUNet",
                "base_channels": base_channels,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": file_sha256(args.checkpoint),
                "checkpoint_epoch": scalar_metadata(checkpoint, "epoch"),
                "best_validation_miou": scalar_metadata(checkpoint, "best_miou"),
            },
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
