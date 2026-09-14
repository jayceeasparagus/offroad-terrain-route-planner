"""Navigation-oriented terrain taxonomy and label remapping."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class TerrainClass:
    """One model output class and its planning cost."""

    id: int
    name: str
    planning_cost: float
    source_label_ids: tuple[int, ...]


@dataclass(frozen=True)
class Taxonomy:
    """Validated collection of terrain classes."""

    classes: tuple[TerrainClass, ...]
    ignore_index: int = 255

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.classes)

    @property
    def costs(self) -> tuple[float, ...]:
        return tuple(item.planning_cost for item in self.classes)

    @property
    def source_to_training_id(self) -> dict[int, int]:
        """Return the raw-dataset ID to model-ID lookup table."""
        return {
            source_id: terrain_class.id
            for terrain_class in self.classes
            for source_id in terrain_class.source_label_ids
        }

    def validate(self) -> None:
        ids = [item.id for item in self.classes]
        if ids != list(range(len(ids))):
            raise ValueError("Terrain class IDs must be contiguous starting at zero")
        if len(set(self.names)) != len(self.names):
            raise ValueError("Terrain class names must be unique")
        if any(item.planning_cost < 0 for item in self.classes):
            raise ValueError("Planning costs must be non-negative")

        source_ids = [source_id for item in self.classes for source_id in item.source_label_ids]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("Source label IDs may only belong to one training class")


def load_taxonomy(path: str | Path) -> Taxonomy:
    """Load and validate a YAML taxonomy file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    classes = tuple(
        TerrainClass(
            id=int(item["id"]),
            name=str(item["name"]),
            planning_cost=float(item["planning_cost"]),
            source_label_ids=tuple(int(source_id) for source_id in item["source_label_ids"]),
        )
        for item in raw["classes"]
    )
    taxonomy = Taxonomy(classes=classes, ignore_index=int(raw.get("ignore_index", 255)))
    taxonomy.validate()
    return taxonomy


def remap_label_ids(label: np.ndarray, taxonomy: Taxonomy) -> np.ndarray:
    """Convert raw dataset IDs into contiguous training IDs.

    Values not declared in the taxonomy become ``ignore_index`` so an unknown
    dataset label cannot silently become a valid training target.
    """
    if label.ndim != 2:
        raise ValueError(f"Expected a 2-D label image, got shape {label.shape}")

    remapped = np.full(label.shape, taxonomy.ignore_index, dtype=np.uint8)
    for source_id, training_id in taxonomy.source_to_training_id.items():
        remapped[label == source_id] = training_id
    return remapped
