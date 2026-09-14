"""Small training/evaluation loops used by the training notebook."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import confusion_matrix, mean_iou, segmentation_loss


@dataclass(frozen=True)
class EpochResult:
    loss: float
    miou: float
    per_class_iou: tuple[float, ...]


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    optimizer: torch.optim.Optimizer | None,
) -> EpochResult:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_items = 0
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["mask"].to(device, non_blocking=True)
            logits = model(images)
            loss = segmentation_loss(logits, targets, num_classes, ignore_index)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = images.shape[0]
            total_loss += loss.detach().item() * batch_size
            total_items += batch_size
            matrix += confusion_matrix(logits.argmax(dim=1), targets, num_classes, ignore_index).to(device)

    ious = matrix.cpu()
    true_positive = torch.diag(ious).float()
    union = ious.sum(dim=1).float() + ious.sum(dim=0).float() - true_positive
    per_class = torch.where(union > 0, true_positive / union, torch.zeros_like(union))
    return EpochResult(
        loss=total_loss / max(total_items, 1),
        miou=float(mean_iou(ious).item()),
        per_class_iou=tuple(float(value) for value in per_class),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    ignore_index: int = 255,
) -> EpochResult:
    return _run_epoch(model, loader, device, num_classes, ignore_index, optimizer)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: int = 255,
) -> EpochResult:
    return _run_epoch(model, loader, device, num_classes, ignore_index, None)
