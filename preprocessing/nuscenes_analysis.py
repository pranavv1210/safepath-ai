import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from nuscenes.nuscenes import NuScenes


MICROSECONDS_PER_SECOND = 1_000_000.0


@dataclass
class TrajectoryPoint:
    timestamp_us: int
    sample_token: str
    annotation_token: str
    x: float
    y: float


def load_nuscenes(dataroot: Path, version: str) -> NuScenes:
    return NuScenes(version=version, dataroot=str(dataroot), verbose=False)


def print_dataset_summary(nusc: NuScenes) -> None:
    print("STEP 1 - DATASET SUMMARY")
    print(f"Scenes: {len(nusc.scene)}")
    print(f"Samples: {len(nusc.sample)}")
    print(f"Sample annotations: {len(nusc.sample_annotation)}")
    print(f"Instances: {len(nusc.instance)}")
    print()


def print_table_structure() -> None:
    print("STEP 2 - IMPORTANT TABLES")
    print("scene: A driving sequence. In nuScenes mini, each scene is about 20 seconds long.")
    print("sample: A keyframe in a scene. Samples are linked with prev/next and carry a timestamp.")
    print("sample_annotation: One object annotation in one sample. This contains translation (x, y, z).")
    print("instance: One physical object tracked across time. It links all annotations for the same agent.")
    print("category: Semantic class such as pedestrian or bicycle.")
    print()
    print("How they link:")
    print("scene.first_sample_token -> sample tokens through sample.next")
    print("sample.anns -> list of sample_annotation tokens in that frame")
    print("sample_annotation.instance_token -> the tracked object")
    print("instance.first_annotation_token / last_annotation_token -> full object track")
    print("sample_annotation.category_name -> semantic type for filtering")
    print()


def iter_instance_annotation_tokens(nusc: NuScenes, instance_token: str) -> Iterable[str]:
    instance = nusc.get("instance", instance_token)
    ann_token = instance["first_annotation_token"]
    while ann_token:
        ann = nusc.get("sample_annotation", ann_token)
        yield ann_token
        ann_token = ann["next"]


def get_instance_category_name(nusc: NuScenes, instance_token: str) -> str:
    instance = nusc.get("instance", instance_token)
    first_ann = nusc.get("sample_annotation", instance["first_annotation_token"])
    return first_ann["category_name"]


def build_instance_trajectory(nusc: NuScenes, instance_token: str) -> List[TrajectoryPoint]:
    trajectory: List[TrajectoryPoint] = []
    for ann_token in iter_instance_annotation_tokens(nusc, instance_token):
        ann = nusc.get("sample_annotation", ann_token)
        sample = nusc.get("sample", ann["sample_token"])
        x, y, _ = ann["translation"]
        trajectory.append(
            TrajectoryPoint(
                timestamp_us=sample["timestamp"],
                sample_token=sample["token"],
                annotation_token=ann_token,
                x=float(x),
                y=float(y),
            )
        )
    trajectory.sort(key=lambda point: point.timestamp_us)
    return trajectory


def trajectory_displacement(trajectory: Sequence[TrajectoryPoint]) -> float:
    if len(trajectory) < 2:
        return 0.0
    start = np.asarray([trajectory[0].x, trajectory[0].y], dtype=np.float32)
    end = np.asarray([trajectory[-1].x, trajectory[-1].y], dtype=np.float32)
    return float(np.linalg.norm(end - start))


def find_first_matching_instance(
    nusc: NuScenes,
    category_substring: str,
    min_points: int = 4,
    min_displacement: float = 1.0,
) -> str:
    category_substring = category_substring.lower()
    for instance in nusc.instance:
        category_name = get_instance_category_name(nusc, instance["token"]).lower()
        if category_substring in category_name:
            trajectory = build_instance_trajectory(nusc, instance["token"])
            if len(trajectory) >= min_points and trajectory_displacement(trajectory) >= min_displacement:
                return instance["token"]
    raise ValueError(f"No instance found for category containing '{category_substring}'.")


def print_pedestrian_trace(nusc: NuScenes) -> List[TrajectoryPoint]:
    pedestrian_token = find_first_matching_instance(nusc, "pedestrian")
    category_name = get_instance_category_name(nusc, pedestrian_token)
    trajectory = build_instance_trajectory(nusc, pedestrian_token)

    print("STEP 3 - TRACE ONE PEDESTRIAN")
    print(f"Instance token: {pedestrian_token}")
    print(f"Category: {category_name}")
    print("[(timestamp_us, x, y), ...]")
    preview = [(point.timestamp_us, round(point.x, 3), round(point.y, 3)) for point in trajectory[:10]]
    print(preview)
    print()
    print("Explanation:")
    print("The same pedestrian is tracked because every annotation in the chain has the same instance_token.")
    print("The annotation chain is walked using sample_annotation.next from first_annotation_token to last_annotation_token.")
    print("Positions change because sample_annotation.translation gives the pedestrian center in each keyframe.")
    print()
    return trajectory


def print_position_extraction() -> None:
    print("STEP 4 - EXTRACT POSITIONS")
    print("sample_annotation.translation = (x, y, z)")
    print("For trajectory prediction here, keep x and y only and ignore z.")
    print("These coordinates are in the global nuScenes map frame.")
    print()


def is_relevant_agent(category_name: str, include_cyclists: bool) -> bool:
    lowered = category_name.lower()
    if lowered.startswith("human.pedestrian"):
        return True
    if include_cyclists and lowered.startswith("vehicle.bicycle"):
        return True
    return False


def collect_trajectories_by_category(nusc: NuScenes, include_cyclists: bool) -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}

    for instance in nusc.instance:
        instance_token = instance["token"]
        category_name = get_instance_category_name(nusc, instance_token)
        if not is_relevant_agent(category_name, include_cyclists):
            continue

        trajectory = build_instance_trajectory(nusc, instance_token)
        if len(trajectory) < 2:
            continue

        results[instance_token] = {
            "category_name": category_name,
            "trajectory": trajectory,
        }

    return results


def print_trajectory_building(trajectories: Dict[str, Dict[str, object]]) -> None:
    print("STEP 5 - BUILD TRAJECTORIES")
    print("Grouped by instance_token.")
    if trajectories:
        example_token, example_payload = max(
            trajectories.items(),
            key=lambda item: trajectory_displacement(item[1]["trajectory"]),
        )
        example_traj = example_payload["trajectory"]
        example_xy = [(round(point.x, 3), round(point.y, 3)) for point in example_traj[:8]]
        print(f"Example instance_token: {example_token}")
        print("Ordered trajectory preview:")
        print(example_xy)
    print()


def estimate_keyframe_rate_hz(nusc: NuScenes) -> float:
    deltas_us: List[int] = []
    for scene in nusc.scene:
        sample_token = scene["first_sample_token"]
        previous_timestamp = None
        while sample_token:
            sample = nusc.get("sample", sample_token)
            current_timestamp = sample["timestamp"]
            if previous_timestamp is not None:
                deltas_us.append(current_timestamp - previous_timestamp)
            previous_timestamp = current_timestamp
            sample_token = sample["next"]

    if not deltas_us:
        return 0.0

    mean_delta_s = float(np.mean(deltas_us)) / MICROSECONDS_PER_SECOND
    return 1.0 / mean_delta_s


def print_temporal_understanding(nusc: NuScenes) -> float:
    keyframe_rate_hz = estimate_keyframe_rate_hz(nusc)
    print("STEP 6 - TEMPORAL UNDERSTANDING")
    print("Move frame-to-frame with sample.prev and sample.next.")
    print("For one object, move annotation-to-annotation with sample_annotation.prev and sample_annotation.next.")
    print("Timestamps come from sample.timestamp in microseconds.")
    print(f"Estimated keyframe rate: {keyframe_rate_hz:.3f} Hz")
    print("That is about one annotation every 0.5 seconds.")
    print()
    return keyframe_rate_hz


def print_relevant_counts(trajectories: Dict[str, Dict[str, object]]) -> None:
    print("STEP 7 - FILTER RELEVANT DATA")
    category_counts: Dict[str, int] = {}
    for payload in trajectories.values():
        category_name = str(payload["category_name"])
        category_counts[category_name] = category_counts.get(category_name, 0) + 1

    print(f"Usable tracked trajectories: {len(trajectories)}")
    for category_name, count in sorted(category_counts.items()):
        print(f"{category_name}: {count}")
    print()


def linear_resample_trajectory(
    trajectory: Sequence[TrajectoryPoint],
    target_hz: float,
) -> np.ndarray:
    timestamps_s = np.asarray([point.timestamp_us for point in trajectory], dtype=np.float64) / MICROSECONDS_PER_SECOND
    xy = np.asarray([[point.x, point.y] for point in trajectory], dtype=np.float32)
    if len(timestamps_s) < 2:
        return xy

    step_s = 1.0 / target_hz
    resampled_times = np.arange(timestamps_s[0], timestamps_s[-1] + 1e-9, step_s, dtype=np.float64)
    x_interp = np.interp(resampled_times, timestamps_s, xy[:, 0])
    y_interp = np.interp(resampled_times, timestamps_s, xy[:, 1])
    return np.stack([x_interp, y_interp], axis=-1).astype(np.float32)


def count_sliding_windows(num_points: int, past_steps: int, future_steps: int) -> int:
    total = past_steps + future_steps
    return max(0, num_points - total + 1)


def build_training_examples(
    trajectories: Dict[str, Dict[str, object]],
    past_steps: int,
    future_steps: int,
    target_hz: float,
) -> Dict[str, np.ndarray]:
    inputs: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    for payload in trajectories.values():
        trajectory = payload["trajectory"]
        xy = linear_resample_trajectory(trajectory, target_hz=target_hz)
        window_count = count_sliding_windows(len(xy), past_steps, future_steps)
        for start in range(window_count):
            past_xy = xy[start : start + past_steps]
            future_xy = xy[start + past_steps : start + past_steps + future_steps]
            anchor = past_xy[-1]
            inputs.append((past_xy - anchor).astype(np.float32))
            targets.append((future_xy - anchor).astype(np.float32))

    if not inputs:
        return {
            "inputs": np.empty((0, past_steps, 2), dtype=np.float32),
            "targets": np.empty((0, future_steps, 2), dtype=np.float32),
        }

    return {
        "inputs": np.stack(inputs),
        "targets": np.stack(targets),
    }


def print_model_format_summary(
    trajectories: Dict[str, Dict[str, object]],
    native_hz: float,
    target_hz: float,
    past_seconds: float,
    future_seconds: float,
    past_steps: int,
    future_steps: int,
) -> None:
    print("STEP 8 - CONVERT TO MODEL FORMAT")
    native_past_steps = int(round(past_seconds * native_hz))
    native_future_steps = int(round(future_seconds * native_hz))
    print(f"Native nuScenes keyframe rate is about {native_hz:.1f} Hz.")
    print(f"At native rate, 2 seconds past = {native_past_steps} steps and 3 seconds future = {native_future_steps} steps.")
    print(f"To match your model format, resample trajectories to {target_hz:.1f} Hz.")
    print(f"Then use input shape ({past_steps}, 2) and target shape ({future_steps}, 2).")
    examples = build_training_examples(
        trajectories=trajectories,
        past_steps=past_steps,
        future_steps=future_steps,
        target_hz=target_hz,
    )
    print(f"Sliding-window training examples after {target_hz:.1f} Hz resampling: {len(examples['inputs'])}")
    print()


def print_summary_answers(native_hz: float) -> None:
    print("STEP 9 - SUMMARY")
    print("1. Which table gives (x, y)?")
    print("sample_annotation via translation[0] and translation[1].")
    print("2. How to track the same person across time?")
    print("Use instance_token, or walk an instance from first_annotation_token through sample_annotation.next.")
    print("3. How to build sequences for training?")
    print("Filter pedestrian/cycle instances, sort each trajectory by timestamp, optionally resample, then slide a fixed window.")
    print("4. What should be ignored for this task?")
    print("Images, LiDAR, radar, sample_data sensor files, and the z coordinate.")
    print()
    print(f"Important note: raw nuScenes annotations are about {native_hz:.1f} Hz, so 8/12 steps needs interpolation to 4 Hz.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze nuScenes for pedestrian/cyclist trajectory prediction.")
    parser.add_argument("--dataroot", type=Path, required=True, help="Path to the nuScenes dataroot.")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--past-seconds", type=float, default=2.0)
    parser.add_argument("--future-seconds", type=float, default=3.0)
    parser.add_argument("--target-hz", type=float, default=4.0, help="Target frequency for model windows.")
    parser.add_argument("--include-cyclists", action="store_true")
    args = parser.parse_args()

    nusc = load_nuscenes(dataroot=args.dataroot, version=args.version)

    print_dataset_summary(nusc)
    print_table_structure()
    print_pedestrian_trace(nusc)
    print_position_extraction()

    trajectories = collect_trajectories_by_category(nusc, include_cyclists=args.include_cyclists)
    print_trajectory_building(trajectories)
    native_hz = print_temporal_understanding(nusc)
    print_relevant_counts(trajectories)

    past_steps = int(round(args.past_seconds * args.target_hz))
    future_steps = int(round(args.future_seconds * args.target_hz))
    print_model_format_summary(
        trajectories=trajectories,
        native_hz=native_hz,
        target_hz=args.target_hz,
        past_seconds=args.past_seconds,
        future_seconds=args.future_seconds,
        past_steps=past_steps,
        future_steps=future_steps,
    )
    print_summary_answers(native_hz=native_hz)


if __name__ == "__main__":
    main()
