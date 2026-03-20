import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.trajectory_model import TrajectoryModel
from preprocessing.trajectory_dataset import build_dataset_from_file, build_dataloader
from training.model_io import load_model_checkpoint, save_model_checkpoint
from training.trajectory_metrics import ade, fde, weighted_trajectory_loss
from utils.config import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, FEATURE_DIM, FUTURE_STEPS


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[Subset, Subset]:
    dataset_size = len(dataset)
    if dataset_size < 2:
        raise ValueError("Need at least 2 samples to make a train/validation split.")

    indices = np.arange(dataset_size)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    train_size = max(1, int(dataset_size * train_ratio))
    val_size = dataset_size - train_size
    if val_size == 0:
        train_size = dataset_size - 1
        val_size = 1

    train_indices = indices[:train_size].tolist()
    val_indices = indices[train_size:].tolist()
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def evaluate(
    model: TrajectoryModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    final_step_weight: float,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            predictions = model(batch_x)
            total_loss += weighted_trajectory_loss(predictions, batch_y, final_step_weight=final_step_weight).item()
            total_ade += ade(predictions, batch_y).item()
            total_fde += fde(predictions, batch_y).item()
            total_batches += 1

    if total_batches == 0:
        return {"loss": 0.0, "ade": 0.0, "fde": 0.0}

    return {
        "loss": total_loss / total_batches,
        "ade": total_ade / total_batches,
        "fde": total_fde / total_batches,
    }


def predict(model: TrajectoryModel, input_sequence: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        if input_sequence.ndim == 2:
            input_sequence = input_sequence.unsqueeze(0)
        input_sequence = input_sequence.to(device)
        prediction = model(input_sequence)
    return prediction.squeeze(0).cpu()


def format_trajectory(points: torch.Tensor) -> list[tuple[float, float]]:
    return [(round(float(x), 4), round(float(y), 4)) for x, y in points]


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = build_dataset_from_file(args.dataset)
    train_dataset, val_dataset = split_dataset(dataset, train_ratio=0.8, seed=args.seed)

    train_loader = build_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = build_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = TrajectoryModel(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        future_steps=args.future_steps,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_ade = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_ade = 0.0
        running_fde = 0.0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = weighted_trajectory_loss(predictions, batch_y, final_step_weight=args.final_step_weight)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_ade += ade(predictions.detach(), batch_y).item()
            running_fde += fde(predictions.detach(), batch_y).item()
            num_batches += 1

        train_metrics = {
            "loss": running_loss / max(num_batches, 1),
            "ade": running_ade / max(num_batches, 1),
            "fde": running_fde / max(num_batches, 1),
        }
        val_metrics = evaluate(model, val_loader, device, final_step_weight=args.final_step_weight)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_ADE={train_metrics['ade']:.4f} | "
            f"train_FDE={train_metrics['fde']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_ADE={val_metrics['ade']:.4f} | "
            f"val_FDE={val_metrics['fde']:.4f}"
        )

        if val_metrics["ade"] < best_val_ade:
            best_val_ade = val_metrics["ade"]
            save_model_checkpoint(
                model=model,
                output_path=args.model_output,
                hidden_size=args.hidden_size,
                future_steps=args.future_steps,
                input_size=args.input_size,
                dropout=args.dropout,
                metrics=val_metrics,
            )

    best_model, checkpoint = load_model_checkpoint(args.model_output, device=device)
    sample_x, sample_y = val_dataset[0]
    predicted = predict(best_model, sample_x, device=device)
    sample_ade = ade(predicted.unsqueeze(0), sample_y.unsqueeze(0)).item()
    sample_fde = fde(predicted.unsqueeze(0), sample_y.unsqueeze(0)).item()

    print()
    print(f"Best checkpoint saved to: {args.model_output}")
    print(f"Best validation metrics: {checkpoint.get('metrics', {})}")
    print("Predicted:")
    print(format_trajectory(predicted))
    print("Ground Truth:")
    print(format_trajectory(sample_y))
    print(f"ADE: {sample_ade:.4f}")
    print(f"FDE: {sample_fde:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LSTM encoder-decoder for trajectory prediction.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--input-size", type=int, default=FEATURE_DIM)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--future-steps", type=int, default=FUTURE_STEPS)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--final-step-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
