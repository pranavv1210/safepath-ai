import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from preprocessing.trajectory_dataset import build_dataset_from_file
from training.model_io import load_model_checkpoint
from training.trajectory_metrics import ade, fde, weighted_trajectory_loss


DATASET_PATH = BASE_DIR / "data" / "processed" / "nuscenes_native_sequences.pt"
MODEL_PATH = BASE_DIR / "models" / "trajectory_model.pth"
OUTPUT_DIR = BASE_DIR / "visualization"


def build_validation_loader(seed: int = 42, batch_size: int = 32) -> tuple[Subset, DataLoader]:
    dataset = build_dataset_from_file(DATASET_PATH)
    indices = np.arange(len(dataset))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    train_size = max(1, int(len(dataset) * 0.8))
    if len(dataset) - train_size == 0:
        train_size = len(dataset) - 1

    val_indices = indices[train_size:].tolist()
    val_dataset = Subset(dataset, val_indices)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return val_dataset, val_loader


def collect_predictions() -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model_checkpoint(MODEL_PATH, device=device)
    model.eval()

    val_dataset, val_loader = build_validation_loader()

    batch_losses = []
    batch_ades = []
    batch_fdes = []
    sample_ades = []
    sample_fdes = []
    sample_records = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            predictions = model(batch_x)

            batch_losses.append(
                weighted_trajectory_loss(predictions, batch_y, final_step_weight=2.0).item()
            )
            batch_ades.append(ade(predictions, batch_y).item())
            batch_fdes.append(fde(predictions, batch_y).item())

            point_distances = torch.linalg.norm(predictions - batch_y, dim=-1)
            per_sample_ade = point_distances.mean(dim=1).cpu().numpy()
            per_sample_fde = point_distances[:, -1].cpu().numpy()

            sample_ades.extend(per_sample_ade.tolist())
            sample_fdes.extend(per_sample_fde.tolist())

            for observed, target, pred, sample_ade_value, sample_fde_value in zip(
                batch_x.cpu().numpy(),
                batch_y.cpu().numpy(),
                predictions.cpu().numpy(),
                per_sample_ade,
                per_sample_fde,
            ):
                sample_records.append(
                    {
                        "observed": observed,
                        "target": target,
                        "predicted": pred,
                        "ade": float(sample_ade_value),
                        "fde": float(sample_fde_value),
                    }
                )

    sample_records.sort(key=lambda item: item["ade"])
    median_record = sample_records[len(sample_records) // 2]

    return {
        "device": str(device),
        "dataset_size": len(val_dataset.dataset),
        "val_size": len(val_dataset),
        "batch_count": len(batch_losses),
        "loss": float(np.mean(batch_losses)),
        "ade": float(np.mean(batch_ades)),
        "fde": float(np.mean(batch_fdes)),
        "sample_ades": np.asarray(sample_ades, dtype=np.float32),
        "sample_fdes": np.asarray(sample_fdes, dtype=np.float32),
        "median_record": median_record,
        "checkpoint_metrics": checkpoint.get("metrics", {}),
    }


def save_summary_figure(results: dict) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("white")

    metrics_ax = axes[0, 0]
    metrics = {
        "Loss": results["loss"],
        "ADE": results["ade"],
        "FDE": results["fde"],
    }
    bars = metrics_ax.bar(metrics.keys(), metrics.values(), color=["#355C7D", "#4ECDC4", "#FF6B6B"])
    metrics_ax.set_title("Validation Metrics")
    metrics_ax.set_ylabel("Value")
    metrics_ax.grid(axis="y", alpha=0.25)
    for bar in bars:
        value = bar.get_height()
        metrics_ax.text(bar.get_x() + bar.get_width() / 2, value + 0.05, f"{value:.4f}", ha="center")

    ade_ax = axes[0, 1]
    ade_ax.hist(results["sample_ades"], bins=24, color="#4ECDC4", edgecolor="white")
    ade_ax.set_title("Per-sample ADE Distribution")
    ade_ax.set_xlabel("ADE")
    ade_ax.set_ylabel("Validation Samples")
    ade_ax.grid(alpha=0.2)

    fde_ax = axes[1, 0]
    fde_ax.hist(results["sample_fdes"], bins=24, color="#FF6B6B", edgecolor="white")
    fde_ax.set_title("Per-sample FDE Distribution")
    fde_ax.set_xlabel("FDE")
    fde_ax.set_ylabel("Validation Samples")
    fde_ax.grid(alpha=0.2)

    text_ax = axes[1, 1]
    text_ax.axis("off")
    summary_lines = [
        "SafePath AI Validation Summary",
        f"Dataset size: {results['dataset_size']}",
        f"Validation size: {results['val_size']}",
        f"Batches: {results['batch_count']}",
        f"Device: {results['device']}",
        f"ADE: {results['ade']:.4f}",
        f"FDE: {results['fde']:.4f}",
        f"Loss: {results['loss']:.4f}",
    ]
    text_ax.text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        fontsize=14,
        linespacing=1.6,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#F7F9FC", "edgecolor": "#D9E2EC"},
    )

    fig.suptitle("Evaluation Metrics Derived from Validation Run", fontsize=18, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = OUTPUT_DIR / "evaluation_metrics_summary.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_trajectory_figure(results: dict) -> Path:
    record = results["median_record"]
    observed = record["observed"][:, :2]
    target = record["target"]
    predicted = record["predicted"]

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("white")

    ax.plot(observed[:, 0], observed[:, 1], marker="o", linewidth=2.5, color="#355C7D", label="Observed trajectory")
    ax.plot(target[:, 0], target[:, 1], marker="o", linewidth=2.5, color="#2A9D8F", label="Ground-truth future")
    ax.plot(predicted[:, 0], predicted[:, 1], marker="o", linewidth=2.5, color="#E76F51", label="Predicted future")

    ax.scatter(observed[-1, 0], observed[-1, 1], s=100, color="#1D3557", zorder=5)
    ax.annotate("Prediction starts here", (observed[-1, 0], observed[-1, 1]), xytext=(8, 8), textcoords="offset points")

    ax.set_title("Sample Validation Trajectory: Prediction vs Ground Truth")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.grid(alpha=0.25)
    ax.legend()
    ax.text(
        0.02,
        0.98,
        f"Sample ADE: {record['ade']:.4f}\nSample FDE: {record['fde']:.4f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#D9E2EC"},
    )

    fig.tight_layout()
    output_path = OUTPUT_DIR / "sample_trajectory_prediction.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = collect_predictions()
    summary_path = save_summary_figure(results)
    trajectory_path = save_trajectory_figure(results)

    print(f"Saved: {summary_path}")
    print(f"Saved: {trajectory_path}")
    print(f"ADE={results['ade']:.6f}")
    print(f"FDE={results['fde']:.6f}")
    print(f"Loss={results['loss']:.6f}")


if __name__ == "__main__":
    main()
