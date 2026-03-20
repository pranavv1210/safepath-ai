from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from models.trajectory_model import TrajectoryModel
from training.model_io import load_model_checkpoint
from utils.config import DEFAULT_MODEL_PATH, FEATURE_DIM, FUTURE_STEPS, PAST_STEPS


def _resolve_model_path(model_path: Path) -> Path:
    return model_path if model_path.is_absolute() else Path(__file__).resolve().parent.parent / model_path


@lru_cache(maxsize=2)
def _cached_model(model_path_str: str, device_str: str) -> tuple[TrajectoryModel, Dict[str, float]]:
    model_path = Path(model_path_str)
    device = torch.device(device_str)
    model, checkpoint = load_model_checkpoint(model_path, device=device)
    metrics = checkpoint.get("metrics", {})
    return model, metrics


def load_model(
    model_path: Path = DEFAULT_MODEL_PATH,
    device: torch.device | None = None,
) -> tuple[TrajectoryModel, Dict[str, float]]:
    resolved_model_path = _resolve_model_path(model_path)
    if not resolved_model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {resolved_model_path}")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _cached_model(str(resolved_model_path), str(device))


def prepare_input_features(trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if trajectory.shape != (PAST_STEPS, FEATURE_DIM):
        raise ValueError(f"Expected trajectory shape {(PAST_STEPS, FEATURE_DIM)}, got {trajectory.shape}")

    origin = trajectory[0, :2].copy()
    prepared = trajectory.copy()
    prepared[:, :2] = prepared[:, :2] - origin
    return prepared.astype(np.float32), origin.astype(np.float32)


def restore_global_coordinates(relative_predictions: np.ndarray, origin: np.ndarray) -> np.ndarray:
    restored = relative_predictions.copy()
    restored[:, :2] = restored[:, :2] + origin
    return restored


def score_trajectories(paths: np.ndarray) -> np.ndarray:
    final_points = paths[:, -1, :]
    center = final_points.mean(axis=0, keepdims=True)
    dispersion = np.linalg.norm(final_points - center, axis=-1)
    logits = -dispersion
    logits = logits - logits.max()
    probabilities = np.exp(logits) / np.exp(logits).sum()
    return probabilities.astype(np.float32)


def predict(model: TrajectoryModel, input_sequence: np.ndarray, device: torch.device | None = None) -> np.ndarray:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        source = torch.tensor(input_sequence[None, ...], dtype=torch.float32, device=device)
        prediction = model(source).cpu().numpy()[0]
    return prediction


def predict_multimodal(
    trajectory: List[List[float]],
    model_path: Path = DEFAULT_MODEL_PATH,
    num_samples: int = 3,
    noise_std: float = 0.02,
) -> Dict[str, List]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metrics = load_model(model_path=model_path, device=device)
    model.train()

    trajectory_array = np.asarray(trajectory, dtype=np.float32)
    prepared_input, origin = prepare_input_features(trajectory_array)

    outputs = []
    with torch.no_grad():
        for _ in range(num_samples):
            noisy_input = prepared_input.copy()
            noisy_input[:, :2] += np.random.normal(0.0, noise_std, size=noisy_input[:, :2].shape).astype(np.float32)
            source = torch.tensor(noisy_input[None, ...], dtype=torch.float32, device=device)
            relative_prediction = model(source).cpu().numpy()[0]
            outputs.append(restore_global_coordinates(relative_prediction, origin))

    paths = np.stack(outputs)
    probabilities = score_trajectories(paths)
    return {
        "paths": paths.tolist(),
        "probabilities": probabilities.tolist(),
        "meta": {
            "path_count": int(num_samples),
            "future_steps": FUTURE_STEPS,
            "past_steps": PAST_STEPS,
            "ade": float(metrics.get("ade", 0.0)),
            "fde": float(metrics.get("fde", 0.0)),
        },
    }
