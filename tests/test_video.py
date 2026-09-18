from pathlib import Path

from PIL import Image

from offroad_perception.video import discover_image_paths


def test_discover_image_paths_sorts_supported_files(tmp_path: Path) -> None:
    Image.new("RGB", (4, 4)).save(tmp_path / "frame_002.jpg")
    Image.new("RGB", (4, 4)).save(tmp_path / "frame_001.png")
    (tmp_path / "notes.txt").write_text("not an image", encoding="utf-8")

    paths = discover_image_paths(tmp_path)

    assert [path.name for path in paths] == ["frame_001.png", "frame_002.jpg"]
