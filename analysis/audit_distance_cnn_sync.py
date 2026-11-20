"""Audit the distance checkpoint on camera images keyed to CARLA world snapshots.

Run from the repository root with python -m analysis.audit_distance_cnn_sync.
This short diagnostic drives with simulator gaps and varies the target gap.
"""
import argparse
import csv
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from queue import Empty, SimpleQueue

import carla
import numpy as np
import torch

from car_dreamer import load_task_configs
from car_dreamer.toolkit.observer.handlers.sensor_handlers import CameraHandler
from carla_wrapper import CarlaWrapper
from generate_dataset import make_weather
from run_gap_pid import GapPID, load_gap_estimator, steering_command
from training.closed_loop import read_snapshot

TERMINAL_KEYS = (
    "destination_reached", "out_of_lane", "terminal_dist", "time_exceeded", "is_collision"
)


class FrameCamera(CameraHandler):
    """Retain the capture ID alongside the stock camera handler's RGB image."""

    def __init__(self, world, config):
        super().__init__(world, config)
        self.samples = SimpleQueue()

    def _update_data(self, data):
        super()._update_data(data)
        self.samples.put((data.frame, self._data))

    def drain(self):
        while True:
            try:
                yield self.samples.get_nowait()
            except Empty:
                return


def collect_episode(env, camera, estimate, seed, target, steps, dt, output, scenario=None):
    random.seed(seed)
    np.random.seed(seed)
    scenario_name = scenario.name if scenario else "legacy"
    if scenario:
        env.select_scenario(scenario)
    env.reset()
    world = env.ego.get_world()
    pid = GapPID(target, kp=0.5, ki=0.0, kd=0.5)
    captures = {}
    decisions = []
    images = []
    snapshots = {}
    latest = None
    missing_decisions = 0

    def read_camera(decision_frame):
        nonlocal latest
        for frame, image in camera.drain():
            if frame not in snapshots or frame in captures:
                continue
            if not image.any():
                raise ValueError(f"Camera frame {frame} is black.")
            time_s, pose, gap, flags = snapshots[frame]
            prediction = estimate(image)
            if not math.isfinite(prediction):
                raise ValueError(f"Nonfinite CNN prediction on frame {frame}.")
            images.append(image.copy())
            captures[frame] = {
                "seed": seed, "scenario": scenario_name, "target_gap_m": target, "camera_frame": frame,
                "simulation_time_s": time_s, "true_gap_m": gap, "predicted_gap_m": prediction,
                "terminal_flags": ", ".join(flags), "frame_index": len(images) - 1,
            }
            if decision_frame is not None and (latest is None or frame > latest):
                latest = frame

    frame, time_s, pose, gap = read_snapshot(env, dt)
    start_pose = pose.tolist()
    snapshots[frame] = (time_s, start_pose, gap, [])
    ending = "step_limit"
    for step in range(steps):
        read_camera(frame)
        if latest is None:
            missing_decisions += 1
        else:
            capture = captures[latest]
            age = frame - latest
            if age < 0:
                raise RuntimeError("Received a camera frame ahead of the decision snapshot.")
            decisions.append({
                "seed": seed, "scenario": scenario_name, "target_gap_m": target, "decision_frame": frame,
                "camera_frame": latest, "camera_age_ticks": age,
                "camera_age_s": time_s - capture["simulation_time_s"],
                "capture_gap_m": capture["true_gap_m"], "decision_gap_m": gap,
                "predicted_gap_m": capture["predicted_gap_m"],
                "repeated_camera_frame": len(decisions) > 0 and
                decisions[-1]["camera_frame"] == latest,
            })

        command = 0.0 if step < 5 else pid.update(gap, dt)
        steer = env.ego_steer_command() if scenario else steering_command(env)
        action = np.array([command, steer], dtype=np.float32)
        _, _, done, info = env.step(action)
        next_frame, time_s, pose, gap = read_snapshot(env, dt)
        if next_frame != frame + 1:
            raise RuntimeError("Snapshots were not consecutive. Use an exclusive CARLA server.")
        frame = next_frame
        flags = [key for key in TERMINAL_KEYS if bool(info.get(key, False))]
        if env.is_collision() and "is_collision" not in flags:
            flags.append("is_collision")
        snapshots[frame] = (time_s, pose.tolist(), gap, flags)
        if done or flags:
            ending = ", ".join(flags) or "environment_done"
            break

    # Advance the render pipeline after ending so delayed terminal images can arrive.
    # These ticks apply brakes, make no decisions, and are outside the labeled episode.
    brake = carla.VehicleControl(brake=1.0)
    env.ego.apply_control(brake)
    env.nonego.apply_control(brake)
    for _ in range(4):
        world.tick()
        read_camera(None)
    rows = sorted(captures.values(), key=lambda row: row["camera_frame"])
    if rows:
        torch.save({
            "images": torch.from_numpy(np.stack([images[row["frame_index"]] for row in rows])),
            "frame_ids": torch.tensor([row["camera_frame"] for row in rows]),
            "timestamps_s": torch.tensor([row["simulation_time_s"] for row in rows],
                                         dtype=torch.float64),
            "states": torch.tensor([snapshots[row["camera_frame"]][1] for row in rows]),
            "distances_m": torch.tensor([row["true_gap_m"] for row in rows]),
            "terminal_flags": [row["terminal_flags"] for row in rows],
            "scenario": asdict(scenario) if scenario else None,
        }, output / f"seed_{seed}.pth")
    for idx, row in enumerate(rows):
        row["frame_index"] = idx
    report = {
        "seed": seed, "scenario": asdict(scenario) if scenario else None,
        "target_gap_m": target, "ending": ending, "steps": step + 1,
        "start_pose": start_pose, "vehicle_types": [env.ego.type_id, env.nonego.type_id],
        "snapshot_count": len(snapshots), "matched_captures": len(rows),
        "decisions_without_camera": missing_decisions,
        "terminal_camera_captured": bool(snapshots[frame][3]) and frame in captures,
    }
    return rows, decisions, report


def error_stats(prediction, target):
    if not len(target):
        return {"frames": 0, "mae_m": None, "mean_signed_error_m": None, "p95_abs_error_m": None}
    error = prediction - target
    absolute = np.abs(error)
    return {
        "frames": len(target), "mae_m": float(absolute.mean()),
        "mean_signed_error_m": float(error.mean()),
        "p95_abs_error_m": float(np.quantile(absolute, 0.95)),
    }


def values(rows, key):
    return np.asarray([row[key] for row in rows], dtype=float)


def summarize(captures, decisions):
    target = values(captures, "true_gap_m")
    prediction = values(captures, "predicted_gap_m")
    gap_masks = {
        "0–3 m": target < 3,
        "3–6 m": (target >= 3) & (target < 6),
        "6–8 m": (target >= 6) & (target < 8),
        "8–10 m": (target >= 8) & (target <= 10),
        ">10 m": target > 10,
    }
    control_gap = values(decisions, "decision_gap_m")
    capture_gap = values(decisions, "capture_gap_m")
    decision_prediction = values(decisions, "predicted_gap_m")
    age = values(decisions, "camera_age_ticks")
    return {
        "unique_capture_frames": error_stats(prediction, target),
        "capture_bins": {label: error_stats(prediction[mask], target[mask])
                         for label, mask in gap_masks.items()},
        "decision_frames_using_capture_gap": error_stats(decision_prediction, capture_gap),
        "decision_frames_using_current_gap": error_stats(decision_prediction, control_gap),
        "capture_gap_minus_current_gap": error_stats(capture_gap, control_gap),
        "camera_age_ticks_histogram": {str(int(i)): int((age == i).sum()) for i in np.unique(age)},
        "repeated_images_at_decision": sum(row["repeated_camera_frame"] for row in decisions),
        "capture_true_above_10_m": int((target > 10).sum()),
        "capture_true_and_predicted_above_10_m": int(((target > 10) & (prediction > 10)).sum()),
    }


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=105)
    parser.add_argument("--shared-route", action="store_true",
                        help="Use fixed Town03 routes and route-following steering")
    parser.add_argument("--target-gaps", type=float, nargs="+", default=(4, 6, 8, 10, 12),
                        help="One simulator-gap PID target per short audit run")
    parser.add_argument("--audit-cutoff", type=float, default=14.0,
                        help="Audit-only gap limit, including samples beyond 10 m")
    args = parser.parse_args()
    if args.steps < 6 or args.steps > 500:
        parser.error("--steps must be 6–500")
    if (not math.isfinite(args.audit_cutoff) or
            any(not 0 < gap < args.audit_cutoff for gap in args.target_gaps)):
        parser.error("All target gaps must be positive and below a finite audit cutoff")

    estimate = load_gap_estimator(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=False)
    config = load_task_configs("carla_follow").update({
        "env.display.enable": False, "env.world.carla_port": args.port,
        "env.action.discrete": False, "env.terminal.terminal_dist": args.audit_cutoff,
    })
    scenarios = (None,)
    env_type = CarlaWrapper
    if args.shared_route:
        from shared_follow_route import SCENARIOS, SharedRouteFollowing

        env_type = SharedRouteFollowing
        scenarios = SCENARIOS
    env = env_type(config=config.env)
    world = env._world.carla_world
    dt = world.get_settings().fixed_delta_seconds
    camera = FrameCamera(env._world, config.env.observation.camera)
    env._observer._data_handlers["camera"] = camera
    captures, decisions, episodes = [], [], []
    try:
        world.set_weather(make_weather("day", "cloudy"))
        for idx, target in enumerate(args.target_gaps):
            rows, control_rows, report = collect_episode(
                env, camera, estimate, args.seed + idx, target, args.steps, dt, args.output,
                scenario=scenarios[idx % len(scenarios)],
            )
            captures.extend(rows)
            decisions.extend(control_rows)
            episodes.append(report)
            print(f"Seed {report['seed']} | {len(rows)} captures | {report['ending']}", flush=True)
    finally:
        try:
            env._observer.destroy()
            for actor_id in env._world.actor_ids:
                env._world.destroy_actor(actor_id)
        finally:
            env._world._set_synchronous_mode(False)

    write_csv(args.output / "captures.csv", captures)
    write_csv(args.output / "decisions.csv", decisions)
    report = {
        "checkpoint": str(args.checkpoint.resolve()), "map": world.get_map().name,
        "weather": "day_cloudy", "fixed_delta_seconds": dt,
        "audit_cutoff_m": args.audit_cutoff, "shared_route": args.shared_route,
        "episodes": episodes, **summarize(captures, decisions),
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not captures:
        raise RuntimeError("No camera frames matched to a simulator snapshot.")


if __name__ == "__main__":
    main()
