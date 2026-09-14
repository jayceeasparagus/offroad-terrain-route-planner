"""Metrics and losses for semantic segmentation."""

import torch
from torch import Tensor
from torch.nn import functional as F


def confusion_matrix(
    prediction: Tensor,
    target: Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> Tensor:
    """Return a ``num_classes x num_classes`` target/prediction matrix."""
    valid = target != ignore_index
    target = target[valid].to(torch.int64)
    prediction = prediction[valid].to(torch.int64)
    encoded = target * num_classes + prediction
    matrix = torch.bincount(encoded, minlength=num_classes**2)
    return matrix.reshape(num_classes, num_classes)


def per_class_iou(matrix: Tensor) -> Tensor:
    """Calculate IoU for each class, using NaN for absent classes."""
    true_positive = torch.diag(matrix).float()
    union = matrix.sum(dim=1).float() + matrix.sum(dim=0).float() - true_positive
    return torch.where(union > 0, true_positive / union, torch.nan)


def mean_iou(matrix: Tensor) -> Tensor:
    """Calculate mean IoU over classes present in the target or prediction."""
    return torch.nanmean(per_class_iou(matrix))


def dice_loss(logits: Tensor, target: Tensor, num_classes: int, ignore_index: int = 255) -> Tensor:
    """Soft multi-class Dice loss that ignores unlabeled pixels."""
    valid = target != ignore_index
    probabilities = logits.softmax(dim=1)
    safe_target = target.masked_fill(~valid, 0)
    one_hot = F.one_hot(safe_target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    valid_mask = valid.unsqueeze(1)
    probabilities = probabilities * valid_mask
    one_hot = one_hot * valid_mask
    intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    return 1.0 - dice.mean()


def segmentation_loss(
    logits: Tensor,
    target: Tensor,
    num_classes: int,
    ignore_index: int = 255,
    dice_weight: float = 0.5,
) -> Tensor:
    """Cross-entropy plus a small Dice term for class imbalance."""
    cross_entropy = F.cross_entropy(logits, target, ignore_index=ignore_index)
    return cross_entropy + dice_weight * dice_loss(logits, target, num_classes, ignore_index)
