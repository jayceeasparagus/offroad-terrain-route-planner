"""Navigation-oriented terrain taxonomy."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TerrainClass:
    """One model output class and its planning cost."""

    id: int
    name: str
    planning_cost: float


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

    def validate(self) -> None:
        ids = [item.id for item in self.classes]
        if ids != list(range(len(ids))):
            raise ValueError("Terrain class IDs must be contiguous starting at zero")
        if len(set(self.names)) != len(self.names):
            raise ValueError("Terrain class names must be unique")
        if any(item.planning_cost < 0 for item in self.classes):
            raise ValueError("Planning costs must be non-negative")


def load_taxonomy(path: str | Path) -> Taxonomy:
    """Load and validate a YAML taxonomy file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    classes = tuple(
        TerrainClass(
            id=int(item["id"]),
            name=str(item["name"]),
            planning_cost=float(item["planning_cost"]),
        )
        for item in raw["classes"]
    )
    taxonomy = Taxonomy(classes=classes, ignore_index=int(raw.get("ignore_index", 255)))
    taxonomy.validate()
    return taxonomy

