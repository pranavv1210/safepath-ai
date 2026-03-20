import torch


def weighted_mse_loss(predictions: torch.Tensor, targets: torch.Tensor, final_step_weight: float = 2.0) -> torch.Tensor:
    weights = torch.ones(targets.size(1), device=targets.device)
    weights[-1] = final_step_weight
    squared_error = (predictions - targets) ** 2
    weighted_error = squared_error * weights.view(1, -1, 1)
    return weighted_error.mean()
