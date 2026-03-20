from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.config import FUTURE_STEPS, PAST_STEPS
from utils.io import load_json


@dataclass
class ScalerStats:
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray

    @classmethod
    def from_json(cls, path: Path) -> "ScalerStats":
        payload = load_json(path)
        return cls(
            input_mean=np.asarray(payload["input_mean"], dtype=np.float32),
            input_std=np.asarray(payload["input_std"], dtype=np.float32),
            target_mean=np.asarray(payload["target_mean"], dtype=np.float32),
            target_std=np.asarray(payload["target_std"], dtype=np.float32),
        )


class TrajectoryDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]


def load_processed_dataset(dataset_path: Path) -> Dict[str, np.ndarray]:
    data = np.load(dataset_path)
    return {key: data[key] for key in data.files}


def build_datasets(dataset_path: Path) -> Dict[str, TrajectoryDataset]:
    arrays = load_processed_dataset(dataset_path)
    return {
        "train": TrajectoryDataset(arrays["X_train"], arrays["Y_train"]),
        "val": TrajectoryDataset(arrays["X_val"], arrays["Y_val"]),
        "test": TrajectoryDataset(arrays["X_test"], arrays["Y_test"]),
    }


def validate_shapes(inputs: np.ndarray, targets: np.ndarray) -> None:
    if inputs.ndim != 3 or inputs.shape[1:] != (PAST_STEPS, 4):
        raise ValueError(f"Expected inputs of shape (N, {PAST_STEPS}, 4), got {inputs.shape}")
    if targets.ndim != 3 or targets.shape[1:] != (FUTURE_STEPS, 2):
        raise ValueError(f"Expected targets of shape (N, {FUTURE_STEPS}, 2), got {targets.shape}")
