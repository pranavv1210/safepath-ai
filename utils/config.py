from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = BASE_DIR / "app" / "checkpoints"
MODEL_DIR = BASE_DIR / "models"

PAST_STEPS = 4
FUTURE_STEPS = 6
FEATURE_DIM = 4
TARGET_DIM = 2
FPS = 2.0
SAMPLING_RATE_HZ = FPS
VEHICLE_SPEED = 1.0
COLLISION_DISTANCE_THRESHOLD = 1.5

DEFAULT_DATASET_PATH = PROCESSED_DATA_DIR / "nuscenes_native_sequences.pt"
DEFAULT_SCALER_PATH = PROCESSED_DATA_DIR / "scaler_stats.json"
DEFAULT_MODEL_PATH = MODEL_DIR / "trajectory_model.pth"
