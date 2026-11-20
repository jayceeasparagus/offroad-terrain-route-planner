from pathlib import Path

from PIL import Image

from offroad_perception.data.rellis import load_split_frames


def test_split_loader_supports_separate_image_and_label_roots(tmp_path: Path) -> None:
    images = tmp_path / "images" / "Rellis-3D"
    labels = tmp_path / "labels" / "Rellis-3D"
    image_path = images / "00000/pylon_camera_node/frame000000-1581624652_750.jpg"
    label_path = labels / "00000/pylon_camera_node_label_id/frame000000-1581624652_750.png"
    image_path.parent.mkdir(parents=True)
    label_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 6)).save(image_path)
    Image.new("L", (8, 6)).save(label_path)
    split_path = tmp_path / "train.lst"
    split_path.write_text(
        "00000/pylon_camera_node/frame000000-1581624652_750.jpg "
        "00000/pylon_camera_node_label_id/frame000000-1581624652_750.png\n"
    )

    frames = load_split_frames(
        images,
        split_path,
        image_root=images,
        label_root=labels,
    )

    assert len(frames) == 1
    assert frames[0].image_path == image_path
    assert frames[0].label_path == label_path
