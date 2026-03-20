from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TrajectoryDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        self.inputs = torch.as_tensor(inputs, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]


def load_processed_dataset(path: Path) -> Dict[str, np.ndarray]:
    if path.suffix == ".pt":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return {
            "X": np.asarray(payload["X"], dtype=np.float32),
            "Y": np.asarray(payload["Y"], dtype=np.float32),
        }

    data = np.load(path)
    return {
        "X": np.asarray(data["X"], dtype=np.float32),
        "Y": np.asarray(data["Y"], dtype=np.float32),
    }


def build_dataset_from_file(path: Path) -> TrajectoryDataset:
    arrays = load_processed_dataset(path)
    return TrajectoryDataset(inputs=arrays["X"], targets=arrays["Y"])


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
