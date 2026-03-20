import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.nuscenes_analysis import collect_trajectories_by_category, load_nuscenes


Trajectory = Sequence[Sequence[float]]

DEFAULT_PAST_STEPS = 4
DEFAULT_FUTURE_STEPS = 6


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_trajectories_from_json(path: Path) -> List[np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [np.asarray(trajectory, dtype=np.float32) for trajectory in payload]


def extract_trajectories_from_nuscenes(
    dataroot: Path,
    version: str = "v1.0-mini",
    include_cyclists: bool = False,
) -> List[np.ndarray]:
    nusc = load_nuscenes(dataroot=dataroot, version=version)
    grouped = collect_trajectories_by_category(nusc, include_cyclists=include_cyclists)
    trajectories: List[np.ndarray] = []

    for payload in grouped.values():
        trajectory = payload["trajectory"]
        xy = np.asarray([[point.x, point.y] for point in trajectory], dtype=np.float32)
        trajectories.append(xy)

    return trajectories


def compute_velocities(xy: np.ndarray) -> np.ndarray:
    velocities = np.zeros_like(xy, dtype=np.float32)
    if len(xy) > 1:
        velocities[1:] = xy[1:] - xy[:-1]
    return velocities


def normalize_sequence(
    past_xy: np.ndarray,
    future_xy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    origin = past_xy[0]
    relative_past_xy = past_xy - origin
    relative_future_xy = future_xy - origin
    return relative_past_xy.astype(np.float32), relative_future_xy.astype(np.float32)


def build_window_features(
    past_xy: np.ndarray,
    future_xy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    relative_past_xy, relative_future_xy = normalize_sequence(past_xy, future_xy)
    velocities = compute_velocities(past_xy)
    features = np.concatenate([relative_past_xy, velocities], axis=-1).astype(np.float32)
    targets = relative_future_xy.astype(np.float32)
    return features, targets


def build_sequence_dataset(
    trajectories: Sequence[np.ndarray],
    past_steps: int = DEFAULT_PAST_STEPS,
    future_steps: int = DEFAULT_FUTURE_STEPS,
) -> Tuple[np.ndarray, np.ndarray]:
    inputs: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    total_steps = past_steps + future_steps

    for trajectory in trajectories:
        xy = np.asarray(trajectory, dtype=np.float32)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"Each trajectory must have shape (T, 2), got {xy.shape}.")
        if len(xy) < total_steps:
            continue

        for start_idx in range(len(xy) - total_steps + 1):
            past_xy = xy[start_idx : start_idx + past_steps]
            future_xy = xy[start_idx + past_steps : start_idx + total_steps]
            features, future_targets = build_window_features(past_xy, future_xy)
            inputs.append(features)
            targets.append(future_targets)

    if not inputs:
        return (
            np.empty((0, past_steps, 4), dtype=np.float32),
            np.empty((0, future_steps, 2), dtype=np.float32),
        )

    return np.stack(inputs), np.stack(targets)


def validate_processed_shapes(
    inputs: np.ndarray,
    targets: np.ndarray,
    past_steps: int = DEFAULT_PAST_STEPS,
    future_steps: int = DEFAULT_FUTURE_STEPS,
) -> None:
    if inputs.ndim != 3 or inputs.shape[1:] != (past_steps, 4):
        raise ValueError(f"Expected inputs of shape (N, {past_steps}, 4), got {inputs.shape}.")
    if targets.ndim != 3 or targets.shape[1:] != (future_steps, 2):
        raise ValueError(f"Expected targets of shape (N, {future_steps}, 2), got {targets.shape}.")


def save_processed_dataset(
    inputs: np.ndarray,
    targets: np.ndarray,
    output_path: Path,
) -> None:
    ensure_parent(output_path)
    if output_path.suffix == ".pt":
        torch.save({"X": inputs, "Y": targets}, output_path)
    elif output_path.suffix == ".npz":
        np.savez(output_path, X=inputs, Y=targets)
    else:
        raise ValueError("Output path must end with .pt or .npz")


def build_dataset_from_json(
    trajectories_path: Path,
    past_steps: int = DEFAULT_PAST_STEPS,
    future_steps: int = DEFAULT_FUTURE_STEPS,
) -> Tuple[np.ndarray, np.ndarray]:
    trajectories = load_trajectories_from_json(trajectories_path)
    return build_sequence_dataset(trajectories, past_steps=past_steps, future_steps=future_steps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training-ready nuScenes trajectory sequences.")
    parser.add_argument("--trajectories-json", type=Path, help="Path to JSON file containing trajectories.")
    parser.add_argument("--dataroot", type=Path, help="nuScenes dataroot. Used when trajectories-json is not set.")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--include-cyclists", action="store_true")
    parser.add_argument("--past-steps", type=int, default=DEFAULT_PAST_STEPS)
    parser.add_argument("--future-steps", type=int, default=DEFAULT_FUTURE_STEPS)
    parser.add_argument("--output", type=Path, help="Optional output path (.pt or .npz).")
    args = parser.parse_args()

    if args.trajectories_json is not None:
        trajectories = load_trajectories_from_json(args.trajectories_json)
    elif args.dataroot is not None:
        trajectories = extract_trajectories_from_nuscenes(
            dataroot=args.dataroot,
            version=args.version,
            include_cyclists=args.include_cyclists,
        )
    else:
        raise ValueError("Provide either --trajectories-json or --dataroot.")

    inputs, targets = build_sequence_dataset(
        trajectories=trajectories,
        past_steps=args.past_steps,
        future_steps=args.future_steps,
    )
    validate_processed_shapes(
        inputs=inputs,
        targets=targets,
        past_steps=args.past_steps,
        future_steps=args.future_steps,
    )

    print(f"Trajectories loaded: {len(trajectories)}")
    print(f"Samples built: {len(inputs)}")
    print(f"X shape: {inputs.shape}")
    print(f"Y shape: {targets.shape}")

    if args.output is not None:
        save_processed_dataset(inputs=inputs, targets=targets, output_path=args.output)
        print(f"Saved processed dataset to {args.output}")


if __name__ == "__main__":
    main()
