# Off-Road Terrain Perception and Route Planning

A small autonomy prototype that turns recorded off-road RGB video into a five-class terrain map, a per-pixel confidence map, and a short local route suggestion.

## Project flow

```text
RGB image sequence + synchronized LiDAR scan
        -> compact U-Net segmentation
        -> terrain class + confidence per pixel
        -> project LiDAR into the calibrated camera view
        -> metric traversability cost grid
        -> A* local route
        -> annotated playback
```

The initial RGB-only baseline uses a compact U-Net and five navigation-oriented classes. Its confidence-aware candidate-route scorer is intentionally simple and image-space only. The enhanced pipeline will use LiDAR for real distances and ground geometry while retaining one compact segmentation model.

## Implemented milestones

1. Dataset manifest and navigation-oriented terrain taxonomy
2. Compact U-Net training and held-out sequence evaluation
3. Video inference with terrain masks and confidence visualization
4. Probability-based traversability cost and RGB-only candidate-route scoring
5. Offline extraction of synchronized RELLIS camera and Ouster LiDAR samples

## Run the local RGB-only route demo

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

## Extract the synchronized RELLIS LiDAR sample

The synchronized RELLIS bag is used only as an offline dataset container. The extractor writes ordinary PNG and NPZ files, so ROS is not required to run inference or planning.

```bash
PYTHONPATH=src python tools/extract_rellis_synced_bag.py \
  --bag data/raw/rellis3d/synced_sample/example_synced.bag \
  --output-dir data/raw/rellis3d/extracted_sample \
  --max-frames 60
```

The output contains `camera/`, `lidar/`, `camera_intrinsics.json`, and `frames.json`. Every exported LiDAR file contains finite XYZ coordinates paired to one camera image within the configured timestamp tolerance.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

Dataset files, model checkpoints, and generated outputs are intentionally excluded from Git.
