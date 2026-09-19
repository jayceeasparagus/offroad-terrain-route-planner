"""Extract synchronized RELLIS camera and LiDAR frames from a ROS bag.

The project uses ROS bags only as an offline dataset container.  The exported
PNG and NPZ files are ordinary files consumed by the rest of the pipeline, so
ROS is not a runtime dependency of the route planner.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IMAGE_TOPIC = "/pylon_camera_node/image_raw"
CAMERA_INFO_TOPIC = "/pylon_camera_node/camera_info"
POINTS_TOPIC = "/os1_cloud_node/points"


def decode_pointcloud_xyz(message: Any) -> np.ndarray:
    """Return finite XYZ coordinates from a ``sensor_msgs/PointCloud2`` message."""
    fields = {field.name: field for field in message.fields}
    missing = {"x", "y", "z"}.difference(fields)
    if missing:
        raise ValueError(f"Point cloud is missing XYZ fields: {sorted(missing)}")

    for name in ("x", "y", "z"):
        field = fields[name]
        # sensor_msgs/PointField.FLOAT32 is 7.
        if field.datatype != 7 or field.count != 1:
            raise ValueError(f"Expected {name} to be one float32 field")

    byte_order = ">" if message.is_bigendian else "<"
    dtype = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": [byte_order + "f4"] * 3,
            "offsets": [fields[name].offset for name in ("x", "y", "z")],
            "itemsize": message.point_step,
        }
    )
    count = int(message.width) * int(message.height)
    packed = np.frombuffer(message.data, dtype=dtype, count=count)
    xyz = np.column_stack((packed["x"], packed["y"], packed["z"])).astype(
        np.float32, copy=False
    )
    return xyz[np.isfinite(xyz).all(axis=1)]


def camera_info_to_dict(message: Any) -> dict[str, Any]:
    """Convert a ROS CameraInfo message into JSON-friendly calibration data."""
    return {
        "width": int(message.width),
        "height": int(message.height),
        "distortion_model": str(message.distortion_model),
        "K": [float(value) for value in message.K],
        "D": [float(value) for value in message.D],
        "R": [float(value) for value in message.R],
        "P": [float(value) for value in message.P],
    }


def decode_camera_image(message: Any) -> Image.Image:
    """Decode common RELLIS RGB/Bayer camera encodings into an RGB PIL image."""
    encoding = str(message.encoding).lower()
    height, width, step = int(message.height), int(message.width), int(message.step)
    raw = np.frombuffer(message.data, dtype=np.uint8).reshape(height, step)

    if encoding == "bayer_rggb8":
        try:
            import cv2
        except ImportError as error:  # pragma: no cover - dependency error path
            raise RuntimeError(
                "OpenCV is required to decode the RELLIS Bayer camera stream. "
                "Install project dependencies with: python -m pip install -e ."
            ) from error
        bayer = raw[:, :width]
        rgb = cv2.cvtColor(bayer, cv2.COLOR_BAYER_RG2RGB)
        return Image.fromarray(rgb, mode="RGB")

    if encoding == "rgb8":
        rgb = raw[:, : width * 3].reshape(height, width, 3)
        return Image.fromarray(rgb, mode="RGB")

    if encoding == "bgr8":
        bgr = raw[:, : width * 3].reshape(height, width, 3)
        return Image.fromarray(bgr[:, :, ::-1], mode="RGB")

    raise ValueError(f"Unsupported camera encoding: {message.encoding}")


def extract_synced_rellis_bag(
    bag_path: Path,
    output_dir: Path,
    *,
    max_frames: int | None = 60,
    pair_tolerance_ms: float = 50.0,
) -> dict[str, Any]:
    """Export synchronized RELLIS RGB images, XYZ scans, and intrinsics.

    ``max_frames`` keeps the default extraction compact for local planning
    experiments. Pass ``None`` to export the entire bag.
    """
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as error:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "rosbags is required to read the RELLIS bag. "
            "Install project dependencies with: python -m pip install -e ."
        ) from error

    bag_path = Path(bag_path)
    if not bag_path.is_file():
        raise FileNotFoundError(f"ROS bag not found: {bag_path}")

    image_dir = output_dir / "camera"
    lidar_dir = output_dir / "lidar"
    image_dir.mkdir(parents=True, exist_ok=True)
    lidar_dir.mkdir(parents=True, exist_ok=True)

    tolerance_ns = int(pair_tolerance_ms * 1_000_000)
    camera_info: dict[str, Any] | None = None
    pending_image: tuple[int, Any] | None = None
    records: list[dict[str, Any]] = []

    with AnyReader([bag_path]) as reader:
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic not in {IMAGE_TOPIC, CAMERA_INFO_TOPIC, POINTS_TOPIC}:
                continue
            message = reader.deserialize(rawdata, connection.msgtype)

            if connection.topic == CAMERA_INFO_TOPIC:
                camera_info = camera_info_to_dict(message)
                continue

            if connection.topic == IMAGE_TOPIC:
                pending_image = (timestamp, message)
                continue

            if pending_image is None:
                continue

            image_timestamp, image_message = pending_image
            time_offset_ns = int(timestamp) - int(image_timestamp)
            pending_image = None
            if abs(time_offset_ns) > tolerance_ns:
                continue
            if camera_info is None:
                raise RuntimeError("Received a camera frame before CameraInfo")

            frame_id = f"frame_{len(records):06d}"
            decode_camera_image(image_message).save(image_dir / f"{frame_id}.png")
            np.savez_compressed(lidar_dir / f"{frame_id}.npz", points=decode_pointcloud_xyz(message))
            records.append(
                {
                    "frame_id": frame_id,
                    "image": f"camera/{frame_id}.png",
                    "lidar": f"lidar/{frame_id}.npz",
                    "image_timestamp_ns": int(image_timestamp),
                    "lidar_timestamp_ns": int(timestamp),
                    "time_offset_ms": time_offset_ns / 1_000_000,
                }
            )
            if max_frames is not None and len(records) >= max_frames:
                break

    if not records:
        raise RuntimeError("No synchronized image/LiDAR pairs were extracted")

    (output_dir / "camera_intrinsics.json").write_text(
        json.dumps(camera_info, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "frames.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "bag": str(bag_path),
        "frames": len(records),
        "max_pair_offset_ms": max(abs(record["time_offset_ms"]) for record in records),
        "camera_intrinsics": "camera_intrinsics.json",
        "frames_manifest": "frames.json",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
