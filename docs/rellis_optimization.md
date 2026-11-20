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

## Kaggle handoff: overnight teacher training

The teacher is the same understandable U-Net design widened from 32 to 64 base
channels. It has about 7.7M parameters, starts from random initialization, and
uses light image augmentation. The existing official RELLIS train, validation,
and test splits stay unchanged, so its held-out test result is comparable to
the compact baseline.

Attach these two existing Kaggle inputs to a fresh GPU notebook:

- `rellis_zip`: the full RGB-image and ID-mask archives.
- `RELLIS`: the small dataset containing `train.lst`, `val.lst`, and `test.lst`.

Use the same clone/setup cell from the baseline section. Then run this cell to
locate the two read-only RELLIS roots and verify the official split:

```python
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
IMAGE_ROOT = next(path for path in INPUT_ROOT.rglob("rellis_images_full") if path.is_dir())
LABEL_ROOT = next(path for path in INPUT_ROOT.rglob("rellis_masks_id") if path.is_dir())
SPLIT_ROOT = next(path.parent for path in INPUT_ROOT.rglob("train.lst"))

print("Images:", IMAGE_ROOT)
print("Labels:", LABEL_ROOT)
print("Splits:", SPLIT_ROOT)
```

Start the run with a Tesla T4 accelerator and Internet enabled:

```python
import subprocess

OUTPUT_DIR = REPO_ROOT / "outputs/checkpoints/rellis_teacher_large64"

subprocess.run(
    [
        "python",
        "tools/train_rellis_teacher.py",
        "--data-root", str(IMAGE_ROOT),
        "--image-root", str(IMAGE_ROOT),
        "--label-root", str(LABEL_ROOT),
        "--split-root", str(SPLIT_ROOT),
        "--output-dir", str(OUTPUT_DIR),
        "--base-channels", "64",
        "--epochs", "60",
        "--batch-size", "4",
        "--num-workers", "2",
        "--device", "cuda",
    ],
    cwd=REPO_ROOT,
    check=True,
)
```

The trainer writes `latest_rellis_teacher.pt` at the end of every epoch and
preserves `best_rellis_teacher.pt` whenever validation mIoU improves. It also
writes final held-out `test_metrics.json`, `test_predictions.png`,
`training_history.json`, and `rellis_teacher_large64_artifacts.zip`.

Use **Save Version** then **Save & Run All** before leaving Kaggle. After the
run finishes, download the artifact ZIP from the version's Output tab. To
continue an intentionally stopped local/draft run, add:

```text
--resume outputs/checkpoints/rellis_teacher_large64/latest_rellis_teacher.pt
```

Keep the baseline JSON unchanged. After the teacher finishes, benchmark it
with the same benchmark command and then compare FP16, channels-last, and
compilation optimizations against both models.
