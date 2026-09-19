import json
from pathlib import Path

import pytest

from offroad_perception.fusion_playback import discover_extracted_pairs


def test_discover_extracted_pairs_reads_manifest(tmp_path: Path) -> None:
    (tmp_path / "camera").mkdir()
    (tmp_path / "lidar").mkdir()
    (tmp_path / "camera/frame_000000.png").touch()
    (tmp_path / "lidar/frame_000000.npz").touch()
    (tmp_path / "frames.json").write_text(
        json.dumps(
            [
                {
                    "frame_id": "frame_000000",
                    "image": "camera/frame_000000.png",
                    "lidar": "lidar/frame_000000.npz",
                }
            ]
        ),
        encoding="utf-8",
    )

    pairs = discover_extracted_pairs(tmp_path)

    assert len(pairs) == 1
    assert pairs[0].frame_id == "frame_000000"


def test_discover_extracted_pairs_rejects_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_extracted_pairs(tmp_path)
