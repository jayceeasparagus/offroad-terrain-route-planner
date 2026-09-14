from pathlib import Path

import numpy as np
import pytest

from offroad_perception.data.dataset import RellisSegmentationDataset
from offroad_perception.data.rellis import discover_camera_frames, discover_lidar_scans
from offroad_perception.data.taxonomy import load_taxonomy, remap_label_ids


ROOT = Path("data/raw/rellis3d")


def test_rellis_label_mapping_covers_declared_source_ids() -> None:
    taxonomy = load_taxonomy(Path("configs/taxonomy.yaml"))
    source_ids = sorted(taxonomy.source_to_training_id)
    labels = np.asarray(source_ids, dtype=np.uint8).reshape(4, 5)
    remapped = remap_label_ids(labels, taxonomy)
    assert set(remapped.flat) == set(range(taxonomy.num_classes))


def test_example_files_can_be_discovered() -> None:
    if not ROOT.exists():
        pytest.skip("RELLIS example data is not available")

    frames = discover_camera_frames(ROOT)
    scans = discover_lidar_scans(ROOT)
    assert len(frames) == 4
    assert len(scans) >= 8
    assert all(frame.label_path.exists() for frame in frames)


def test_example_dataset_returns_tensors() -> None:
    if not ROOT.exists():
        pytest.skip("RELLIS example data is not available")

    taxonomy = load_taxonomy(Path("configs/taxonomy.yaml"))
    frames = discover_camera_frames(ROOT)
    sample = RellisSegmentationDataset(frames, taxonomy, image_size=(64, 96))[0]
    assert sample["image"].shape == (3, 64, 96)
    assert sample["mask"].shape == (64, 96)
    assert str(sample["mask"].dtype) == "torch.int64"
