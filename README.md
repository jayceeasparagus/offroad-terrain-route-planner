# Off-Road Terrain Perception and Route Planning

A small, end-to-end autonomy prototype that turns recorded RGB and LiDAR data into a short, local route through off-road terrain.

## Project flow

```text
RGB + LiDAR + calibration + odometry
        -> semantic segmentation
        -> semantic LiDAR fusion
        -> local traversability grid
        -> A* route planning
        -> annotated playback
```

The first version intentionally uses one compact U-Net, five navigation-oriented classes, provided poses, and a classical A* planner. ROS 2, SLAM, learned route prediction, and multiple-model comparisons are outside the initial scope.

## Planned milestones

1. Dataset manifest and taxonomy
2. Segmentation training and held-out sequence evaluation
3. Camera-LiDAR projection and semantic point painting
4. Local traversability mapping
5. A* route planning
6. Video playback, benchmarks, and failure analysis

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

Dataset files, model checkpoints, and generated outputs are intentionally excluded from Git.

