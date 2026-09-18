import numpy as np
from PIL import Image

from offroad_perception.visualization import (
    colorize_classes,
    confidence_heatmap,
    render_dashboard,
    semantic_overlay,
)


CLASS_NAMES = (
    "traversable_ground",
    "soft_or_risky",
    "vegetation",
    "obstacle",
    "unknown",
)


def test_visualization_builds_labeled_dashboard() -> None:
    class_ids = np.asarray([[0, 1, 2], [3, 4, 0]], dtype=np.uint8)
    camera = Image.new("RGB", (30, 20), "black")
    mask = colorize_classes(class_ids, CLASS_NAMES)
    confidence = confidence_heatmap(np.full((2, 3), 0.5, dtype=np.float32))
    overlay = semantic_overlay(camera, mask)
    dashboard = render_dashboard(camera, mask, overlay, confidence, panel_size=(30, 20))

    assert mask.size == (3, 2)
    assert confidence.size == (3, 2)
    assert dashboard.size == (60, 84)


def test_colorize_rejects_out_of_range_class_id() -> None:
    class_ids = np.asarray([[5]], dtype=np.uint8)
    try:
        colorize_classes(class_ids, CLASS_NAMES)
    except ValueError as error:
        assert "outside" in str(error)
    else:
        raise AssertionError("Invalid class ID should raise ValueError")
