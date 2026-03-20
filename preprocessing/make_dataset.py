import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from preprocessing.dataset import validate_shapes
from utils.config import DEFAULT_DATASET_PATH, DEFAULT_SCALER_PATH, FUTURE_STEPS, PAST_STEPS
from utils.io import save_json
from utils.seed import set_seed


def generate_agent_sequence(total_steps: int) -> np.ndarray:
    start = np.random.uniform(-8.0, 8.0, size=2)
    velocity = np.random.uniform(-0.6, 0.6, size=2)
    points: List[np.ndarray] = []
    current = start.astype(np.float32)

    for step in range(total_steps):
        acceleration = 0.04 * np.array(
            [np.sin(step / 3.0 + start[0]), np.cos(step / 4.0 + start[1])],
            dtype=np.float32,
        )
        drift = np.random.normal(0.0, 0.03, size=2).astype(np.float32)
        velocity = 0.92 * velocity + acceleration + drift
        current = current + velocity
        points.append(current.copy())

    return np.stack(points)


def build_sequences(num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    inputs = []
    targets = []
    total_steps = PAST_STEPS + FUTURE_STEPS

    for _ in range(num_samples):
        coords = generate_agent_sequence(total_steps)
        velocities = np.zeros_like(coords)
        velocities[1:] = coords[1:] - coords[:-1]

        past_xy = coords[:PAST_STEPS]
        future_xy = coords[PAST_STEPS:]
        past_v = velocities[:PAST_STEPS]

        anchor = past_xy[-1]
        relative_past_xy = past_xy - anchor
        relative_future_xy = future_xy - anchor

        features = np.concatenate([relative_past_xy, past_v], axis=-1)
        inputs.append(features.astype(np.float32))
        targets.append(relative_future_xy.astype(np.float32))

    return np.stack(inputs), np.stack(targets)


def normalize_splits(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    train_y: np.ndarray,
    val_y: np.ndarray,
    test_y: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[float]]]:
    input_mean = train_x.reshape(-1, train_x.shape[-1]).mean(axis=0)
    input_std = train_x.reshape(-1, train_x.shape[-1]).std(axis=0) + 1e-6
    target_mean = train_y.reshape(-1, train_y.shape[-1]).mean(axis=0)
    target_std = train_y.reshape(-1, train_y.shape[-1]).std(axis=0) + 1e-6

    arrays = {
        "X_train": ((train_x - input_mean) / input_std).astype(np.float32),
        "X_val": ((val_x - input_mean) / input_std).astype(np.float32),
        "X_test": ((test_x - input_mean) / input_std).astype(np.float32),
        "Y_train": ((train_y - target_mean) / target_std).astype(np.float32),
        "Y_val": ((val_y - target_mean) / target_std).astype(np.float32),
        "Y_test": ((test_y - target_mean) / target_std).astype(np.float32),
    }
    stats = {
        "input_mean": input_mean.tolist(),
        "input_std": input_std.tolist(),
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
    }
    return arrays, stats


def split_dataset(inputs: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, ...]:
    total = inputs.shape[0]
    train_end = int(total * 0.7)
    val_end = int(total * 0.85)
    return (
        inputs[:train_end],
        inputs[train_end:val_end],
        inputs[val_end:],
        targets[:train_end],
        targets[train_end:val_end],
        targets[val_end:],
    )


def save_dataset(output_path: Path, arrays: Dict[str, np.ndarray]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and preprocess trajectory dataset.")
    parser.add_argument("--samples", type=int, default=1200, help="Number of trajectory sequences to generate.")
    parser.add_argument("--output", type=Path, default=DEFAULT_DATASET_PATH, help="Path to save processed dataset.")
    parser.add_argument("--scaler-output", type=Path, default=DEFAULT_SCALER_PATH, help="Path to save scaler stats.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    inputs, targets = build_sequences(args.samples)
    validate_shapes(inputs, targets)
    split = split_dataset(inputs, targets)
    arrays, stats = normalize_splits(*split)
    save_dataset(args.output, arrays)
    save_json(args.scaler_output, stats)

    print(f"Saved processed dataset to {args.output}")
    print(f"Saved scaler stats to {args.scaler_output}")
    print(f"Train: {arrays['X_train'].shape}, Val: {arrays['X_val'].shape}, Test: {arrays['X_test'].shape}")


if __name__ == "__main__":
    main()
