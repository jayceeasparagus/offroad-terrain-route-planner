# Off-Road Terrain Perception and Route Planning

A small autonomy prototype that turns recorded off-road RGB video into a five-class terrain map, a per-pixel confidence map, and eventually a short local route.

## Project flow

```text
RGB image sequence or video
        -> compact U-Net segmentation
        -> terrain class + confidence per pixel
        -> traversability scoring
        -> candidate local route
        -> annotated playback
```

The first version intentionally uses one compact U-Net and five navigation-oriented classes. It uses a classical, confidence-aware candidate-route scorer rather than a second learned model. ROS 2, SLAM, LiDAR fusion, and multiple-model comparisons are outside the initial scope.

## Planned milestones

1. Dataset manifest and taxonomy
2. Segmentation training and held-out sequence evaluation
3. Video inference, confidence visualization, and playback
4. Confidence-aware traversability scoring
5. Candidate-route generation and scoring
6. Playback benchmarks and failure analysis

## Run local video inference

After copying a trained checkpoint to `outputs/checkpoints/rellis-epoch8-baseline/`, create a semantic playback from one RELLIS camera sequence:

```bash
PYTHONPATH=src python tools/run_video_inference.py \
  --input-dir data/raw/rellis3d/full/Rellis-3D/00000/pylon_camera_node \
  --device cpu \
  --max-frames 20 \
  --output-dir outputs/video_inference
```

The command writes `terrain_playback.gif`, a `preview.png`, per-frame dashboard PNGs, and `benchmark.json`. The four dashboard panels are the camera image, semantic mask, color overlay, and model confidence.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

Dataset files, model checkpoints, and generated outputs are intentionally excluded from Git.
