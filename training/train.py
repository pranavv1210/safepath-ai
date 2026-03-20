"""
Legacy training entry point kept for compatibility.

The project now uses the unified nuScenes-native training pipeline in
`training/train_model.py`.
"""

from training.train_model import parse_args, train


if __name__ == "__main__":
    train(parse_args())
