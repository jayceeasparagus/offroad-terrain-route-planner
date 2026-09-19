from types import SimpleNamespace

import numpy as np
import pytest

from offroad_perception.rellis_bag import camera_info_to_dict, decode_pointcloud_xyz


def test_decode_pointcloud_xyz_filters_nonfinite_points() -> None:
    dtype = np.dtype(
        {"names": ["x", "y", "z"], "formats": ["<f4"] * 3, "offsets": [0, 4, 8], "itemsize": 16}
    )
    packed = np.zeros(3, dtype=dtype)
    packed["x"] = [1.0, np.nan, 7.0]
    packed["y"] = [2.0, 3.0, 8.0]
    packed["z"] = [3.0, 4.0, np.inf]
    fields = [
        SimpleNamespace(name="x", offset=0, datatype=7, count=1),
        SimpleNamespace(name="y", offset=4, datatype=7, count=1),
        SimpleNamespace(name="z", offset=8, datatype=7, count=1),
    ]
    message = SimpleNamespace(
        fields=fields,
        is_bigendian=False,
        point_step=16,
        width=3,
        height=1,
        data=packed.tobytes(),
    )

    xyz = decode_pointcloud_xyz(message)

    assert xyz.shape == (1, 3)
    np.testing.assert_allclose(xyz, [[1.0, 2.0, 3.0]])


def test_camera_info_to_dict_serializes_numbers() -> None:
    message = SimpleNamespace(
        width=1920,
        height=1200,
        distortion_model="plumb_bob",
        K=np.arange(9),
        D=np.arange(5),
        R=np.arange(9),
        P=np.arange(12),
    )

    data = camera_info_to_dict(message)

    assert data["width"] == 1920
    assert data["K"] == list(range(9))
    assert data["P"] == list(range(12))
