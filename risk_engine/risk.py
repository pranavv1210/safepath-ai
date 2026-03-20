from typing import Dict, List

import numpy as np

from utils.config import COLLISION_DISTANCE_THRESHOLD, FPS, VEHICLE_SPEED


def vehicle_path(num_steps: int) -> np.ndarray:
    x_positions = np.arange(num_steps, dtype=np.float32) * VEHICLE_SPEED
    y_positions = np.zeros(num_steps, dtype=np.float32)
    return np.stack([x_positions, y_positions], axis=-1)


def classify_risk(min_distance: float, ttc: float | None) -> str:
    if min_distance < 1.0 or (ttc is not None and ttc <= 1.5):
        return "HIGH"
    if min_distance < 2.0 or (ttc is not None and ttc <= 3.0):
        return "MEDIUM"
    return "LOW"


def analyze_trajectory(path: np.ndarray, probability: float) -> Dict[str, float | str | bool]:
    ego_path = vehicle_path(len(path))
    distances = np.linalg.norm(path - ego_path, axis=-1)
    closest_idx = int(np.argmin(distances))
    min_distance = float(distances[closest_idx])
    intersects = bool(min_distance < COLLISION_DISTANCE_THRESHOLD)
    ttc = float(closest_idx / FPS) if intersects else None

    risk_level = classify_risk(min_distance, ttc)
    proximity_score = float(np.clip(1.0 - min_distance / 3.0, 0.0, 1.0))
    urgency_score = 0.0 if ttc is None else float(np.clip(1.0 - ttc / 4.0, 0.0, 1.0))
    collision_probability = float(np.clip(0.25 * probability + 0.5 * proximity_score + 0.25 * urgency_score, 0.0, 1.0))

    return {
        "risk_level": risk_level,
        "collision_probability": round(collision_probability, 4),
        "min_distance": round(min_distance, 4),
        "time_to_collision": None if ttc is None else round(ttc, 2),
        "intersection": intersects,
    }
def score_paths(paths: List[List[List[float]]], probabilities: List[float]) -> List[Dict[str, float | str | bool]]:
    return [
        analyze_trajectory(np.asarray(path, dtype=np.float32), float(probability))
        for path, probability in zip(paths, probabilities)
    ]
