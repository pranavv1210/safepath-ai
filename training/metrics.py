import torch


def ade(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    distances = torch.linalg.norm(predictions - targets, dim=-1)
    return distances.mean()


def fde(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    final_distances = torch.linalg.norm(predictions[:, -1] - targets[:, -1], dim=-1)
    return final_distances.mean()
