from pathlib import Path
from typing import Any, Dict

import torch

from models.trajectory_model import TrajectoryModel


def save_model_checkpoint(
    model: TrajectoryModel,
    output_path: Path,
    hidden_size: int,
    future_steps: int,
    input_size: int,
    dropout: float,
    metrics: Dict[str, float] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_size": hidden_size,
            "future_steps": future_steps,
            "input_size": input_size,
            "dropout": dropout,
            "metrics": metrics or {},
        },
        output_path,
    )


def load_model_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[TrajectoryModel, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TrajectoryModel(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        future_steps=checkpoint["future_steps"],
        dropout=checkpoint.get("dropout", 0.1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint
