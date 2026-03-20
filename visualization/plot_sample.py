import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from inference.predict import predict_multimodal
from risk_engine.risk import score_paths


def plot_sample(output_path: Path) -> None:
    past = np.array(
        [
            [-4.0, -1.0, 0.0, 0.0],
            [-3.6, -0.7, 0.4, 0.3],
            [-3.1, -0.5, 0.5, 0.2],
            [-2.5, -0.3, 0.6, 0.2],
            [-1.8, -0.1, 0.7, 0.2],
            [-1.0, 0.1, 0.8, 0.2],
            [-0.2, 0.25, 0.8, 0.15],
            [0.5, 0.35, 0.7, 0.1],
        ],
        dtype=np.float32,
    )
    result = predict_multimodal(past.tolist())
    risks = score_paths(result["paths"], result["probabilities"])

    plt.figure(figsize=(8, 6))
    plt.plot(past[:, 0], past[:, 1], color="blue", marker="o", label="Past")
    colors = {"HIGH": "red", "MEDIUM": "gold", "LOW": "green"}
    for idx, path in enumerate(result["paths"]):
        path_array = np.asarray(path)
        risk = risks[idx]["risk_level"]
        plt.plot(
            path_array[:, 0],
            path_array[:, 1],
            color=colors[risk],
            linestyle="--",
            marker="x",
            label=f"Pred {idx + 1} ({risk})",
        )

    plt.scatter([0], [0], color="black", s=80, label="Vehicle")
    plt.title("SafePath AI Sample Prediction")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a sample SafePath AI visualization.")
    parser.add_argument("--output", type=Path, default=Path("static/sample_plot.png"))
    args = parser.parse_args()
    plot_sample(args.output)
