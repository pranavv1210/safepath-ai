import torch


def ade(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    distances = torch.linalg.norm(predictions - targets, dim=-1)
    return distances.mean()


def fde(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    final_distances = torch.linalg.norm(predictions[:, -1] - targets[:, -1], dim=-1)
    return final_distances.mean()


def weighted_trajectory_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    final_step_weight: float = 2.0,
) -> torch.Tensor:
    mse_all_steps = torch.mean((predictions - targets) ** 2)
    mse_final_step = torch.mean((predictions[:, -1] - targets[:, -1]) ** 2)
    return mse_all_steps + final_step_weight * mse_final_step
