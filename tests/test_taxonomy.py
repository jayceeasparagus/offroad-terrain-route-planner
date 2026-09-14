from pathlib import Path

from offroad_perception.data.taxonomy import load_taxonomy


def test_default_taxonomy_is_valid() -> None:
    taxonomy = load_taxonomy(Path("configs/taxonomy.yaml"))
    assert taxonomy.num_classes == 5
    assert taxonomy.names == ("traversable_ground", "soft_or_risky", "vegetation", "obstacle", "unknown")
    assert taxonomy.ignore_index == 255


def test_planning_costs_are_ordered_by_risk() -> None:
    taxonomy = load_taxonomy(Path("configs/taxonomy.yaml"))
    assert taxonomy.costs[0] < taxonomy.costs[1] < taxonomy.costs[2] < taxonomy.costs[3]

