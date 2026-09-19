from pathlib import Path

import numpy as np

from offroad_perception.calibration import (
    CameraCalibration,
    LidarToCameraTransform,
    load_rellis_lidar_to_camera_transform,
    project_lidar_to_image,
    quaternion_to_rotation_matrix,
)


def test_project_lidar_points_filters_behind_and_out_of_bounds() -> None:
    camera = CameraCalibration(
        matrix=np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
        distortion=np.empty(0),
        width=101,
        height=101,
    )
    transform = LidarToCameraTransform(np.eye(3), np.zeros(3))
    points = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 0.0, -1.0], [2.0, 0.0, 1.0]])

    result = project_lidar_to_image(points, camera, transform)

    assert result.source_indices.tolist() == [0, 1]
    np.testing.assert_allclose(result.pixels, [[50.0, 50.0], [100.0, 50.0]])


def test_load_rellis_transform_maps_translation_and_rotation(tmp_path: Path) -> None:
    calibration = tmp_path / "transforms.yaml"
    calibration.write_text(
        """os1_cloud_node-pylon_camera_node:
  q: {w: 1.0, x: 0.0, y: 0.0, z: 0.0}
  t: {x: 1.0, y: 2.0, z: 3.0}
""",
        encoding="utf-8",
    )

    transform = load_rellis_lidar_to_camera_transform(calibration)

    np.testing.assert_allclose(transform.apply(np.array([[4.0, 5.0, 6.0]])), [[3.0, 3.0, 3.0]])
    np.testing.assert_allclose(quaternion_to_rotation_matrix(w=1, x=0, y=0, z=0), np.eye(3))
