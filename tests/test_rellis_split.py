from pathlib import Path

import pytest

from offroad_perception.data.rellis import load_split_frames


FULL_ROOT = Path("data/raw/rellis3d/full")


def test_official_rellis_splits_load() -> None:
    if not FULL_ROOT.exists():
        pytest.skip("full RELLIS dataset is not available")

    train = load_split_frames(FULL_ROOT, "train.lst")
    validation = load_split_frames(FULL_ROOT, "val.lst")
    test = load_split_frames(FULL_ROOT, "test.lst")

    assert len(train) == 3302
    assert len(validation) == 983
    assert len(test) == 1672
    assert all(frame.image_path.exists() and frame.label_path.exists() for frame in train[:10])
