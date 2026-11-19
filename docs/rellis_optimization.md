# RELLIS model optimization workflow

This workflow starts with a measured baseline for the final RELLIS Compact
U-Net. It isolates the deployment-relevant image path:

```text
RGB image -> resize and tensor conversion -> Compact U-Net -> class mask + confidence
```

The benchmark is intentionally separate from LiDAR fusion and route planning.
Those are useful system demonstrations, but they would hide where segmentation
latency comes from.

## Kaggle handoff: baseline benchmark

### 1. Create the checkpoint input

On the local machine, the final checkpoint is already at:

```text
outputs/checkpoints/rellis-final-40/best_compact_unet.pt
```

Create a private Kaggle dataset named `rellis-final-artifacts` and upload the
existing `offroad_terrain_artifacts.zip` download, or upload that checkpoint
file directly. Attach both that dataset and the existing `rellis_zip` dataset
to a new Kaggle notebook. RELLIS itself is not read during this baseline run,
but attaching it prepares the notebook for the teacher-training step.

### 2. Configure the notebook

Set the accelerator to a Tesla T4 and enable Internet for the initial clone.
Run this cell:

```python
from pathlib import Path
import subprocess

REPO_ROOT = Path("/kaggle/working/offroad-terrain-route-planner")

if not REPO_ROOT.exists():
    subprocess.run(
        [
            "git",
            "clone",
            "https://github.com/jayceeasparagus/offroad-terrain-route-planner.git",
            str(REPO_ROOT),
        ],
        check=True,
    )
else:
    subprocess.run(["git", "pull", "--ff-only"], cwd=REPO_ROOT, check=True)

subprocess.run(
    ["python", "-m", "pip", "install", "-e", ".", "--no-deps"],
    cwd=REPO_ROOT,
    check=True,
)
```

### 3. Find the attached checkpoint

```python
from pathlib import Path

checkpoint_candidates = list(Path("/kaggle/input").rglob("best_compact_unet.pt"))
assert checkpoint_candidates, "Attach the dataset containing best_compact_unet.pt"

CHECKPOINT = checkpoint_candidates[0]
print("Using checkpoint:", CHECKPOINT)
```

### 4. Run the baseline

```python
import subprocess

OUTPUT_JSON = Path("/kaggle/working/rellis_compact_unet_fp32_baseline.json")

subprocess.run(
    [
        "python",
        "tools/benchmark_compact_unet.py",
        "--checkpoint", str(CHECKPOINT),
        "--device", "cuda",
        "--warmup", "50",
        "--iterations", "300",
        "--output", str(OUTPUT_JSON),
    ],
    cwd=REPO_ROOT,
    check=True,
)
```

The output reports model-only latency, complete in-memory RGB-to-mask latency,
and FPS for batch-one FP32 inference. It excludes image disk I/O. Use **Save
Version** after the run; the JSON will appear under that version's Output tab.

## What comes after the baseline

Keep this JSON unchanged. The next milestone trains a larger RELLIS teacher
from random initialization, evaluates it on the held-out sequences, and runs
the same benchmark command. Only then do we compare FP16, channels-last, and
compilation optimizations against both models.
