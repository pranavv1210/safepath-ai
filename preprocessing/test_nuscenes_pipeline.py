import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.nuscenes_preprocess import (
    build_sequence_dataset,
    extract_trajectories_from_nuscenes,
    load_trajectories_from_json,
    save_processed_dataset,
    validate_processed_shapes,
)
from preprocessing.trajectory_dataset import TrajectoryDataset, build_dataloader


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the nuScenes trajectory preprocessing pipeline.")
    parser.add_argument("--trajectories-json", type=Path, help="Path to JSON file containing trajectories.")
    parser.add_argument("--dataroot", type=Path, help="nuScenes dataroot. Used when trajectories-json is not set.")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--include-cyclists", action="store_true")
    parser.add_argument("--past-steps", type=int, default=4)
    parser.add_argument("--future-steps", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, help="Optional output path for saving processed tensors.")
    args = parser.parse_args()

    if args.trajectories_json is not None:
        trajectories = load_trajectories_from_json(args.trajectories_json)
    elif args.dataroot is not None:
        trajectories = extract_trajectories_from_nuscenes(
            dataroot=args.dataroot,
            version=args.version,
            include_cyclists=args.include_cyclists,
        )
    else:
        raise ValueError("Provide either --trajectories-json or --dataroot.")

    inputs, targets = build_sequence_dataset(
        trajectories=trajectories,
        past_steps=args.past_steps,
        future_steps=args.future_steps,
    )
    validate_processed_shapes(
        inputs=inputs,
        targets=targets,
        past_steps=args.past_steps,
        future_steps=args.future_steps,
    )

    dataset = TrajectoryDataset(inputs=inputs, targets=targets)
    if len(dataset) == 0:
        raise ValueError("No training samples were created. Check trajectory lengths or step settings.")
    dataloader = build_dataloader(dataset, batch_size=args.batch_size, shuffle=True)
    batch_x, batch_y = next(iter(dataloader))

    print(f"Number of trajectories: {len(trajectories)}")
    print(f"Number of samples: {len(dataset)}")
    print(f"X batch shape: {tuple(batch_x.shape)}")
    print(f"Y batch shape: {tuple(batch_y.shape)}")

    if args.output is not None:
        save_processed_dataset(inputs=inputs, targets=targets, output_path=args.output)
        print(f"Saved processed dataset to {args.output}")


if __name__ == "__main__":
    main()
