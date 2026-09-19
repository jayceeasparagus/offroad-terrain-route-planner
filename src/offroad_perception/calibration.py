"""Camera/LiDAR calibration loading and geometric point projection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class CameraCalibration:
    """Pinhole camera intrinsics and lens-distortion coefficients."""

    matrix: np.ndarray
    distortion: np.ndarray
    width: int
    height: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraCalibration":
        matrix = np.asarray(data["K"], dtype=np.float64).reshape(3, 3)
        distortion = np.asarray(data.get("D", []), dtype=np.float64)
        return cls(
            matrix=matrix,
            distortion=distortion,
            width=int(data["width"]),
            height=int(data["height"]),
        )

    @classmethod
    def from_json(cls, path: Path) -> "CameraCalibration":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class LidarToCameraTransform:
    """Rigid transform that maps a LiDAR point into camera coordinates."""

    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Expected LiDAR points with shape (N, 3)")
        return points @ self.rotation.T + self.translation


@dataclass(frozen=True)
class ProjectionResult:
    """LiDAR points that land inside an image, with their original indices."""

    source_indices: np.ndarray
    pixels: np.ndarray
    camera_points: np.ndarray


def quaternion_to_rotation_matrix(
    *, w: float, x: float, y: float, z: float
) -> np.ndarray:
    """Return a 3x3 rotation matrix from a normalized scalar-first quaternion."""
    quaternion = np.asarray([w, x, y, z], dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if norm == 0:
        raise ValueError("Calibration quaternion cannot be zero")
    w, x, y, z = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_rellis_lidar_to_camera_transform(path: Path) -> LidarToCameraTransform:
    """Load and invert RELLIS's camera-to-Ouster TF transform for projection."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    key = "os1_cloud_node-pylon_camera_node"
    if key not in data:
        raise KeyError(f"Expected {key!r} in RELLIS calibration file: {path}")
    transform = data[key]
    quaternion = transform["q"]
    translation = transform["t"]
    # In ROS TF notation this record gives the child camera pose in the parent
    # Ouster frame: p_lidar = R_camera_to_lidar * p_camera + t.  Projection
    # needs the opposite mapping, p_camera = R.T * (p_lidar - t).
    camera_to_lidar_rotation = quaternion_to_rotation_matrix(**quaternion)
    camera_to_lidar_translation = np.asarray(
        [translation["x"], translation["y"], translation["z"]], dtype=np.float64
    )
    lidar_to_camera_rotation = camera_to_lidar_rotation.T
    return LidarToCameraTransform(
        rotation=lidar_to_camera_rotation,
        translation=-lidar_to_camera_rotation @ camera_to_lidar_translation,
    )


def project_lidar_to_image(
    points: np.ndarray,
    camera: CameraCalibration,
    lidar_to_camera: LidarToCameraTransform,
    *,
    min_depth_m: float = 0.2,
) -> ProjectionResult:
    """Project finite, forward LiDAR points onto the calibrated camera image."""
    camera_points = lidar_to_camera.apply(points)
    finite = np.isfinite(camera_points).all(axis=1)
    forward = camera_points[:, 2] > min_depth_m
    source_indices = np.flatnonzero(finite & forward)
    forward_points = camera_points[source_indices]

    if not len(forward_points):
        return ProjectionResult(
            source_indices=np.empty(0, dtype=np.int64),
            pixels=np.empty((0, 2), dtype=np.float64),
            camera_points=np.empty((0, 3), dtype=np.float64),
        )

    try:
        import cv2
    except ImportError as error:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "OpenCV is required for distortion-aware point projection. "
            "Install project dependencies with: python -m pip install -e ."
        ) from error

    pixels, _ = cv2.projectPoints(
        forward_points.reshape(-1, 1, 3),
        np.zeros(3),
        np.zeros(3),
        camera.matrix,
        camera.distortion,
    )
    pixels = pixels.reshape(-1, 2)
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < camera.width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < camera.height)
    )
    return ProjectionResult(
        source_indices=source_indices[inside],
        pixels=pixels[inside],
        camera_points=forward_points[inside],
    )
