#!/usr/bin/env python3
"""Train a wider RELLIS Compact U-Net teacher from random initialization."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import shutil
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from offroad_perception.data import (
    RellisSegmentationDataset,
    load_split_frames,
    load_taxonomy,
)
from offroad_perception.inference import resolve_device
from offroad_perception.models import CompactUNet
from offroad_perception.training import evaluate, train_one_epoch_amp
from offroad_perception.visualization import colorize_classes, confidence_heatmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--label-root", type=Path)
    parser.add_argument("--taxonomy", type=Path, default=Path("configs/taxonomy.yaml"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/checkpoints/rellis_teacher_large64"),
    )
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--preview-count", type=int, default=4)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    return value


def make_loader(
    dataset: RellisSegmentationDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def result_payload(result: Any, names: tuple[str, ...]) -> dict[str, object]:
    payload = asdict(result)
    payload["per_class_iou"] = dict(zip(names, result.per_class_iou, strict=True))
    return payload


def save_preview(
    model: CompactUNet,
    dataset: RellisSegmentationDataset,
    device: torch.device,
    class_names: tuple[str, ...],
    output_path: Path,
    count: int,
) -> None:
    """Write RGB, target, prediction, and confidence rows from the test set."""

    indices = np.linspace(0, len(dataset) - 1, min(max(1, count), len(dataset)), dtype=int)
    rows: list[Image.Image] = []
    model.eval()
    with torch.inference_mode():
        for index in indices:
            sample = dataset[int(index)]
            logits = model(sample["image"].unsqueeze(0).to(device))
            probabilities = torch.softmax(logits, dim=1)[0]
            confidence, prediction = probabilities.max(dim=0)

            rgb = sample["image"].permute(1, 2, 0).numpy()
            rgb_image = Image.fromarray(np.clip(rgb * 255, 0, 255).astype(np.uint8))
            target = sample["mask"].numpy()
            target = np.where(target == 255, len(class_names) - 1, target).astype(np.uint8)
            panels = [
                rgb_image,
                colorize_classes(target, class_names),
                colorize_classes(prediction.cpu().numpy().astype(np.uint8), class_names),
                confidence_heatmap(confidence.cpu().numpy()),
            ]
            row = Image.new("RGB", (sum(panel.width for panel in panels), panels[0].height))
            offset = 0
            for panel in panels:
                row.paste(panel, (offset, 0))
                offset += panel.width
            rows.append(row)

    canvas = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)))
    offset = 0
    for row in rows:
        canvas.paste(row, (0, offset))
        offset += row.height
    canvas.save(output_path)


def checkpoint_payload(
    *,
    model: CompactUNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_miou: float,
    history: list[dict[str, object]],
    class_names: tuple[str, ...],
    arguments: argparse.Namespace,
) -> dict[str, object]:
    return {
        "model_config": {
            "architecture": "CompactUNet",
            "in_channels": 3,
            "num_classes": len(class_names),
            "base_channels": model.enc1.block[0].weight.shape[0],
        },
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "best_miou": best_miou,
        "history": history,
        "taxonomy": class_names,
        "training_arguments": json_ready(vars(arguments)),
    }


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("epochs and batch_size must be positive; num_workers cannot be negative")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    args.output_dir.mkdir(parents=True, exist_ok=True)

    taxonomy = load_taxonomy(args.taxonomy)
    split_root = args.split_root or args.data_root
    split_paths = {
        "train": split_root / "train.lst",
        "validation": split_root / "val.lst",
        "test": split_root / "test.lst",
    }
    split_frames = {
        name: load_split_frames(
            args.data_root,
            split_path,
            image_root=args.image_root,
            label_root=args.label_root,
        )
        for name, split_path in split_paths.items()
    }
    print("Frame counts:", {name: len(frames) for name, frames in split_frames.items()})

    train_dataset = RellisSegmentationDataset(
        split_frames["train"], taxonomy, image_size=(320, 512), augment=True
    )
    validation_dataset = RellisSegmentationDataset(
        split_frames["validation"], taxonomy, image_size=(320, 512)
    )
    test_dataset = RellisSegmentationDataset(
        split_frames["test"], taxonomy, image_size=(320, 512)
    )
    train_loader = make_loader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, device=device, seed=args.seed)
    validation_loader = make_loader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, device=device, seed=args.seed + 1)
    test_loader = make_loader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, device=device, seed=args.seed + 2)

    model = CompactUNet(
        in_channels=3,
        num_classes=taxonomy.num_classes,
        base_channels=args.base_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    start_epoch = 1
    best_miou = -1.0
    history: list[dict[str, object]] = []
    epochs_without_improvement = 0

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        model_config = checkpoint.get("model_config", {})
        if model_config.get("base_channels") != args.base_channels:
            raise ValueError("resume checkpoint width does not match --base-channels")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_miou = float(checkpoint["best_miou"])
        history = list(checkpoint["history"])
        print(f"Resuming from epoch {start_epoch - 1}; best validation mIoU {best_miou:.4f}")

    latest_path = args.output_dir / "latest_rellis_teacher.pt"
    best_path = args.output_dir / "best_rellis_teacher.pt"
    for epoch in range(start_epoch, args.epochs + 1):
        train_result = train_one_epoch_amp(
            model, train_loader, optimizer, device, taxonomy.num_classes,
            ignore_index=taxonomy.ignore_index, use_amp=use_amp, scaler=scaler,
        )
        validation_result = evaluate(
            model, validation_loader, device, taxonomy.num_classes, taxonomy.ignore_index
        )
        scheduler.step(validation_result.miou)
        record = {
            "epoch": epoch,
            "train": result_payload(train_result, taxonomy.names),
            "validation": result_payload(validation_result, taxonomy.names),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        improved = validation_result.miou > best_miou
        if improved:
            best_miou = validation_result.miou
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        payload = checkpoint_payload(
            model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            epoch=epoch, best_miou=best_miou, history=history,
            class_names=taxonomy.names, arguments=args,
        )
        torch.save(payload, latest_path)
        if improved:
            torch.save(payload, best_path)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train {train_result.loss:.4f} | "
            f"val {validation_result.loss:.4f} | val mIoU {validation_result.miou:.4f} | "
            f"best {best_miou:.4f}"
        )
        if epochs_without_improvement >= args.early_stopping_patience:
            print("Early stopping: validation mIoU has not improved recently.")
            break

    best_checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_result = evaluate(
        model, test_loader, device, taxonomy.num_classes, taxonomy.ignore_index
    )
    test_metrics = result_payload(test_result, taxonomy.names)
    (args.output_dir / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
    (args.output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2) + "\n")
    save_preview(
        model, test_dataset, device, taxonomy.names,
        args.output_dir / "test_predictions.png", args.preview_count,
    )
    archive = shutil.make_archive(
        str(args.output_dir.parent / f"{args.output_dir.name}_artifacts"),
        "zip",
        root_dir=args.output_dir,
    )
    print("Test results:", json.dumps(test_metrics, indent=2))
    print("Best checkpoint:", best_path)
    print("Artifacts:", archive)


if __name__ == "__main__":
    main()
