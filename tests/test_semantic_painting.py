import numpy as np

from offroad_perception.calibration import ProjectionResult
from offroad_perception.inference import SegmentationPrediction
from offroad_perception.semantic_painting import (
    paint_projected_points,
    sample_probabilities_at_pixels,
)


def test_sample_probabilities_uses_nearest_pixel_centers() -> None:
    probabilities = np.array(
        [
            [[0.9, 0.1], [0.2, 0.3]],
            [[0.1, 0.9], [0.8, 0.7]],
        ],
        dtype=np.float32,
    )

    sampled = sample_probabilities_at_pixels(
        probabilities, np.array([[0.2, 0.2], [1.0, 1.0]])
    )

    np.testing.assert_allclose(sampled, [[0.9, 0.1], [0.3, 0.7]])


def test_paint_projected_points_preserves_semantics_and_risk() -> None:
    prediction = SegmentationPrediction(
        class_ids=np.array([[0, 1], [0, 1]], dtype=np.uint8),
        confidence=np.array([[0.8, 0.9], [0.6, 0.7]], dtype=np.float32),
        probabilities=np.array(
            [
                [[0.8, 0.1], [0.6, 0.3]],
                [[0.2, 0.9], [0.4, 0.7]],
            ],
            dtype=np.float32,
        ),
    )
    projection = ProjectionResult(
        source_indices=np.array([0, 2]),
        pixels=np.array([[0.0, 0.0], [1.0, 1.0]]),
        camera_points=np.array([[0.0, 0.0, 3.0], [1.0, 0.0, 4.0]]),
    )
    points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)

    painted = paint_projected_points(
        points,
        projection,
        prediction,
        class_costs=[1.0, 100.0],
        obstacle_class_id=1,
        obstacle_block_threshold=0.5,
    )

    assert painted.points_lidar.tolist() == [[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]]
    assert painted.class_ids.tolist() == [0, 1]
    assert painted.blocked.tolist() == [False, True]
    np.testing.assert_allclose(painted.confidence, [0.8, 0.7])
    assert painted.route_risk[0] < painted.route_risk[1]
