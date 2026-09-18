# Off-Road Terrain Perception and Route Planning

A small autonomy prototype that turns recorded off-road RGB video into a five-class terrain map, a per-pixel confidence map, and a short image-space route suggestion.

## Project flow

```text
RGB image sequence or video
        -> compact U-Net segmentation
        -> terrain class + confidence per pixel
        -> traversability cost image
        -> score candidate visual routes
        -> annotated playback
```

The first version intentionally uses one compact U-Net and five navigation-oriented classes. It uses a classical, confidence-aware candidate-route scorer rather than a second learned model. The route is an image-space look-ahead suggestion; without camera calibration or vehicle state, it is not a metric GPS path. ROS 2, SLAM, LiDAR fusion, and multiple-model comparisons are outside the initial scope.

## Implemented milestones

1. Dataset manifest and navigation-oriented terrain taxonomy
2. Compact U-Net training and held-out sequence evaluation
3. Video inference with terrain masks and confidence visualization
4. Probability-based traversability cost and candidate-route scoring

## Run the local route demo

After copying a trained checkpoint to `outputs/checkpoints/rellis-epoch8-baseline/`, create an annotated playback from one RELLIS camera sequence:

```bash
PYTHONPATH=src python tools/run_video_inference.py \
  --input-dir data/raw/rellis3d/full/Rellis-3D/00000/pylon_camera_node \
  --device cpu \
  --max-frames 20 \
  --output-dir outputs/route_playback
```

The command writes `terrain_route_playback.gif`, a `preview.png`, per-frame dashboard PNGs, `benchmark.json`, and `route_summary.json`.

The dashboard shows candidate routes in blue, the lowest-cost selected route in green, a semantic overlay, a terrain-cost image, and the model confidence. Purple cost-map cells are blocked because the model assigned at least 35% obstacle probability.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

Dataset files, model checkpoints, and generated outputs are intentionally excluded from Git.
