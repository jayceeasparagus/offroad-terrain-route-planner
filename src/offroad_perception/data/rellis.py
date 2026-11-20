"""RELLIS-3D image, label, split, and LiDAR discovery utilities."""

from dataclasses import dataclass
from pathlib import Path
import re


_FRAME_PATTERN = re.compile(r"frame(?P<index>\d+)-(?P<seconds>\d+)_(?P<millis>\d+)")


@dataclass(frozen=True)
class RellisFrame:
    """One camera/label record with an optional time-matched LiDAR scan."""

    frame_index: int
    timestamp: float
    image_path: Path
    label_path: Path
    lidar_path: Path | None = None
    lidar_timestamp_delta_s: float | None = None


@dataclass(frozen=True)
class RellisLidarScan:
    """A LiDAR scan identified by its frame index and timestamp."""

    frame_index: int
    timestamp: float
    path: Path


def _parse_frame_name(path: Path) -> tuple[int, float]:
    match = _FRAME_PATTERN.search(path.stem)
    if match is None:
        raise ValueError(f"Could not parse RELLIS frame name: {path.name}")
    seconds = int(match.group("seconds"))
    millis = int(match.group("millis").ljust(3, "0")[:3])
    return int(match.group("index")), seconds + millis / 1000.0


def discover_camera_frames(dataset_root: str | Path) -> list[RellisFrame]:
    """Find RGB images with matching ID masks under a RELLIS root."""
    root = Path(dataset_root)
    label_by_stem = {
        path.stem: path
        for path in root.rglob("pylon_camera_node_label_id/*")
        if path.suffix.lower() == ".png"
    }

    frames: list[RellisFrame] = []
    for image_path in sorted(root.rglob("pylon_camera_node/*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label_path = label_by_stem.get(image_path.stem)
        if label_path is None:
            continue
        frame_index, timestamp = _parse_frame_name(image_path)
        frames.append(RellisFrame(frame_index, timestamp, image_path, label_path))
    return frames


def _resolve_split_path(root: Path, relative_path: str) -> Path:
    """Resolve a split path against either a RELLIS root or ``Rellis-3D``."""

    with_container = root / "Rellis-3D" / relative_path
    without_container = root / relative_path
    if with_container.exists():
        return with_container
    return without_container


def load_split_frames(
    dataset_root: str | Path,
    split_file: str | Path,
    *,
    image_root: str | Path | None = None,
    label_root: str | Path | None = None,
) -> list[RellisFrame]:
    """Load official RELLIS splits, optionally using separate image/mask roots.

    The public RELLIS camera and label archives are often mounted as separate
    Kaggle inputs. ``image_root`` and ``label_root`` let the trainer read those
    inputs directly instead of copying their multi-gigabyte contents.
    """
    root = Path(dataset_root)
    split_path = Path(split_file)
    if not split_path.is_absolute():
        split_path = root / split_path

    image_data_root = Path(image_root) if image_root is not None else root
    label_data_root = Path(label_root) if label_root is not None else root

    frames: list[RellisFrame] = []
    with split_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.strip().split()
            if not fields or fields[0].startswith("#"):
                continue
            if len(fields) < 2:
                raise ValueError(f"Invalid split entry at {split_path}:{line_number}")

            image_path = _resolve_split_path(image_data_root, fields[0])
            label_path = _resolve_split_path(label_data_root, fields[1])
            if not image_path.exists() or not label_path.exists():
                raise FileNotFoundError(
                    f"Split entry points to missing files at {split_path}:{line_number}"
                )
            frame_index, timestamp = _parse_frame_name(image_path)
            frames.append(RellisFrame(frame_index, timestamp, image_path, label_path))
    return frames


def discover_lidar_scans(dataset_root: str | Path) -> list[RellisLidarScan]:
    """Find example color PLY LiDAR scans under a RELLIS root."""
    root = Path(dataset_root)
    scans: list[RellisLidarScan] = []
    for lidar_path in sorted(root.rglob("os1_cloud_node_color_ply/*.ply")):
        frame_index, timestamp = _parse_frame_name(lidar_path)
        scans.append(RellisLidarScan(frame_index, timestamp, lidar_path))
    return scans


def attach_nearest_lidar(
    frames: list[RellisFrame],
    scans: list[RellisLidarScan],
    max_delta_s: float = 0.10,
) -> list[RellisFrame]:
    """Attach the nearest LiDAR scan when it is close enough in time."""
    result: list[RellisFrame] = []
    for frame in frames:
        candidates = [scan for scan in scans if scan.frame_index == frame.frame_index]
        if not candidates:
            result.append(frame)
            continue
        nearest = min(candidates, key=lambda scan: abs(scan.timestamp - frame.timestamp))
        delta = abs(nearest.timestamp - frame.timestamp)
        if delta <= max_delta_s:
            result.append(
                RellisFrame(
                    frame_index=frame.frame_index,
                    timestamp=frame.timestamp,
                    image_path=frame.image_path,
                    label_path=frame.label_path,
                    lidar_path=nearest.path,
                    lidar_timestamp_delta_s=delta,
                )
            )
        else:
            result.append(frame)
    return result
