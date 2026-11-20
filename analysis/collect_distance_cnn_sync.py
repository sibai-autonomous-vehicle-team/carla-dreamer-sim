"""Collect camera images labeled by the exact CARLA capture frame."""

import argparse
import json
import math
import random
import re
from dataclasses import asdict
from pathlib import Path

import carla
import numpy as np
import torch

from analysis.audit_distance_cnn_sync import FrameCamera, TERMINAL_KEYS
from car_dreamer import load_task_configs
from generate_dataset import TIME_OF_DAY, WEATHER_PRESETS, make_weather
from run_gap_pid import GapPID
from shared_follow_route import Scenario, SharedRouteFollowing
from training.closed_loop import read_snapshot


PARTITIONS = ("train", "validation", "test")
CONDITIONS = tuple(f"{time}_{weather}" for time in TIME_OF_DAY for weather in WEATHER_PRESETS)
COLLECTION_GAINS = (0.9, 0.05, 0.15)
MIN_CLOSE_GAP_M = 0.30
MIN_MOVING_SPEED_MPS = 0.25
LEAD_SPEED_MPS = 1.0


def horizontal_speed(actor):
    velocity = actor.get_velocity()
    return math.hypot(velocity.x, velocity.y)


def speed_command(speed, target):
    if target <= 0:
        return -3.0
    return float(np.clip(0.7 + 2.3 * (target - speed), -3.0, 3.0))


class CloseGapFollowing(SharedRouteFollowing):
    """Track short gaps while both vehicles move at walking speed."""

    def get_nonego_vehicle_control(self):
        command = speed_command(horizontal_speed(self.nonego), LEAD_SPEED_MPS)
        return carla.VehicleControl(throttle=max(0.0, command / 3), brake=max(0.0, -command / 3),
                                    steer=self.route.steer(self.nonego))

    def ego_close_gap_command(self, gap, target, ego_speed, lead_speed):
        if gap <= MIN_CLOSE_GAP_M:
            return -3.0
        target_speed = np.clip(lead_speed + 0.9 * (gap - target), 0.0, 1.45)
        return speed_command(ego_speed, target_speed)


def load_routes(path):
    groups = json.loads(path.read_text())
    if not isinstance(groups, dict) or set(groups) != set(PARTITIONS):
        raise ValueError("Routes must define train, validation, and test partitions.")
    result = {}
    names, geometry = set(), set()
    for split in PARTITIONS:
        if not isinstance(groups[split], list) or not groups[split]:
            raise ValueError(f"The {split} partition must contain at least one route.")
        result[split] = []
        for route in groups[split]:
            scenario = Scenario(**{**route, "branch_roads": tuple(route.get("branch_roads", ()))})
            if not re.fullmatch(r"[A-Za-z0-9_-]+", scenario.name) or scenario.name in names:
                raise ValueError(f"Route name must be unique and filesystem-safe: {scenario.name}")
            if scenario.length < 10 or not math.isfinite(scenario.finish_m) or scenario.finish_m <= 0:
                raise ValueError(f"Invalid route length or finish: {scenario.name}")
            key = (scenario.road, scenario.lane, scenario.start_s, scenario.branch_roads)
            if key in geometry:
                raise ValueError(f"Duplicate route geometry: {scenario.name}")
            names.add(scenario.name)
            geometry.add(key)
            result[split].append(scenario)
    return result


def collect_episode(env, camera, scenario, seed, target, steps, dt, stride, close_gaps=False):
    random.seed(seed)
    np.random.seed(seed)
    env.select_scenario(scenario)
    if close_gaps:
        # Start above the target and close gently under measured-gap feedback.
        env.lead_start_index = math.ceil(5 + target)
    env.reset()
    world = env.ego.get_world()
    pid = GapPID(target, *COLLECTION_GAINS)
    snapshots, images = {}, {}
    frame, time_s, pose, gap = read_snapshot(env, dt)
    snapshots[frame] = (time_s, pose, gap, "")
    speeds = {}
    if close_gaps:
        speeds[frame] = (horizontal_speed(env.ego), horizontal_speed(env.nonego))

    def save_matching_images():
        for capture_frame, image in camera.drain():
            if capture_frame not in snapshots or capture_frame in images or not image.any():
                continue
            if close_gaps:
                _, _, capture_gap, flags = snapshots[capture_frame]
                if (not MIN_CLOSE_GAP_M <= capture_gap <= 3.25
                        or min(speeds[capture_frame]) < MIN_MOVING_SPEED_MPS
                        or "is_collision" in flags):
                    continue
            images[capture_frame] = image.copy()

    ending = "step_limit"
    for step in range(steps):
        if step < 5:
            command = 0.0
        elif close_gaps:
            command = env.ego_close_gap_command(gap, target, *speeds[frame])
        else:
            command = pid.update(gap, dt)
        action = np.array([command, env.ego_steer_command()], dtype=np.float32)
        _, _, done, info = env.step(action)
        next_frame, time_s, pose, gap = read_snapshot(env, dt)
        if next_frame != frame + 1:
            raise RuntimeError("Snapshots were not consecutive. Use an exclusive CARLA server.")
        frame = next_frame
        flags = [key for key in TERMINAL_KEYS if bool(info.get(key, False))]
        if env.is_collision() and "is_collision" not in flags:
            flags.append("is_collision")
        terminal = ", ".join(flags)
        snapshots[frame] = (time_s, pose, gap, terminal)
        if close_gaps:
            speeds[frame] = (horizontal_speed(env.ego), horizontal_speed(env.nonego))
        save_matching_images()
        if done or terminal:
            ending = terminal or "environment_done"
            break

    # Drain delayed renders without adding frames after the episode to the archive.
    brake = carla.VehicleControl(brake=1.0)
    env.ego.apply_control(brake)
    env.nonego.apply_control(brake)
    for _ in range(4):
        world.tick()
        save_matching_images()

    if not images:
        raise RuntimeError(f"No synchronized camera frames for {scenario.name}, seed {seed}.")
    selected = sorted(images)[::stride]
    if frame in images and ending != "step_limit" and frame not in selected:
        selected.append(frame)
    times, poses, gaps, flags = zip(*(snapshots[frame_id] for frame_id in selected))
    archive = {
        "images": torch.from_numpy(np.stack([images[frame_id] for frame_id in selected])),
        "distances_m": torch.tensor(gaps, dtype=torch.float32),
        "frame_ids": torch.tensor(selected, dtype=torch.int64),
        "states": torch.stack(poses),
        "timestamps_s": torch.tensor(times, dtype=torch.float64),
        "terminal_flags": list(flags),
        "scenario": asdict(scenario),
        "target_gap_m": target,
        "seed": seed,
        "ending": ending,
        "steps": step + 1,
        "matched_frames_before_subsampling": len(images),
        "vehicle_types": [env.ego.type_id, env.nonego.type_id],
    }
    if close_gaps:
        archive.update(mode="moving_close_gaps", lead_start_index=env.lead_start_index,
                       speeds_mps=torch.tensor([speeds[frame_id] for frame_id in selected]))
    return archive


def atomic_write(path, writer):
    temporary = path.with_suffix(".tmp")
    writer(temporary)
    temporary.replace(path)


def write_json(path, payload):
    atomic_write(path, lambda temporary: temporary.write_text(json.dumps(payload, indent=2) + "\n"))


def record_episode(manifest, split, relative, archive, scenario, seed, target):
    gaps = archive["distances_m"]
    frames = archive["frame_ids"]
    if (archive["seed"] != seed or archive["target_gap_m"] != target
            or archive["scenario"] != asdict(scenario)
            or not len(gaps) or len(gaps) != len(archive["images"]) or len(frames) != len(gaps)
            or not torch.isfinite(gaps).all() or (gaps < 0).any() or (frames.diff() <= 0).any()):
        raise ValueError(f"Invalid or conflicting episode archive: {relative}")
    manifest[split].append(str(relative))
    manifest["episodes"].append({
        "path": str(relative), "split": split, "scenario": scenario.name, "seed": seed,
        "target_gap_m": target, "frames": len(gaps), "ending": archive["ending"],
        "gap_m_min": float(gaps.min()), "gap_m_max": float(gaps.max()),
        "frames_0_3_to_1_m": int(((gaps >= 0.3) & (gaps < 1)).sum()),
        "frames_1_to_2_m": int(((gaps >= 1) & (gaps < 2)).sum()),
        "frames_2_to_3_m": int(((gaps >= 2) & (gaps <= 3)).sum()),
        "frames_8_to_10_m": int(((gaps >= 8) & (gaps <= 10)).sum()),
        "frames_above_10_m": int((gaps > 10).sum()),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--condition", choices=CONDITIONS, default="day_cloudy",
                        help="Lighting and weather preset (default: day_cloudy)")
    parser.add_argument("--seed", type=int, default=4200)
    parser.add_argument("--target-gaps", type=float, nargs="+")
    parser.add_argument("--close-gaps", action="store_true",
                        help="Collect moving close gaps using closer starts and a slow lead")
    parser.add_argument("--steps", type=int, default=75)
    parser.add_argument("--capture-every", type=int, default=2)
    parser.add_argument("--cutoff", type=float, default=15.0,
                        help="Collection-only true gap limit in metres")
    args = parser.parse_args()
    if args.target_gaps is None:
        args.target_gaps = (0.65, 1.5, 2.5) if args.close_gaps else (5, 8, 11)
    if not 6 <= args.steps <= 500 or args.capture_every < 1:
        parser.error("Use 6–500 steps and a positive capture stride.")
    if not math.isfinite(args.cutoff) or not all(0 < gap < args.cutoff for gap in args.target_gaps):
        parser.error("Use a finite distance cutoff above every positive target gap.")
    if args.close_gaps and not all(MIN_CLOSE_GAP_M <= gap < 3 for gap in args.target_gaps):
        parser.error("Use close-gap targets from 0.3 m up to 3 m.")
    if not 0 <= args.seed < 2**32:
        parser.error("Seed must fit a NumPy 32-bit unsigned seed.")

    routes = load_routes(args.routes)
    count = sum(len(group) for group in routes.values()) * len(args.target_gaps)
    if args.seed + count > 2**32:
        parser.error("Seed plus number of planned episodes exceeds the NumPy seed range.")
    settings = {
        "routes": {split: [asdict(route) for route in group] for split, group in routes.items()},
        "seed": args.seed, "target_gaps_m": args.target_gaps, "max_steps": args.steps,
        "capture_every": args.capture_every, "collection_cutoff_m": args.cutoff,
        "pid_gains": COLLECTION_GAINS, "weather": args.condition, "map": "Town03",
    }
    if args.close_gaps:
        del settings["pid_gains"]
        settings.update(mode="moving_close_gaps", lead_speed_mps=LEAD_SPEED_MPS,
                        minimum_capture_gap_m=MIN_CLOSE_GAP_M,
                        minimum_capture_speed_mps=MIN_MOVING_SPEED_MPS)
    # Progress is recoverable after any fully written episode. Resume only the same plan.
    settings = json.loads(json.dumps(settings))
    args.output.mkdir(parents=True, exist_ok=True)
    plan = args.output / "collection_plan.json"
    if plan.exists():
        if json.loads(plan.read_text()) != settings:
            raise ValueError(f"Collection settings changed within {args.output}.")
    else:
        if (args.output / "episodes").exists() or (args.output / "split.json").exists():
            raise ValueError(f"Cannot verify existing episode plan in {args.output}.")
        write_json(plan, settings)

    manifest = {split: [] for split in PARTITIONS}
    manifest.update(episodes=[], planned_episodes=count, complete=False)
    config = load_task_configs("carla_follow").update({
        "env.display.enable": False, "env.world.carla_port": args.port,
        "env.action.discrete": False, "env.terminal.terminal_dist": args.cutoff,
        "env.observation.enabled": ["camera", "collision"],
    })
    env = None
    try:
        seed = args.seed
        for split, scenarios in routes.items():
            for scenario in scenarios:
                for target in args.target_gaps:
                    relative = Path("episodes") / split / scenario.name / f"seed_{seed}.pth"
                    path = args.output / relative
                    if path.exists():
                        archive = torch.load(path, map_location="cpu", weights_only=True)
                    else:
                        if env is None:
                            env_type = CloseGapFollowing if args.close_gaps else SharedRouteFollowing
                            env = env_type(config=config.env)
                            world = env._world.carla_world
                            dt = world.get_settings().fixed_delta_seconds
                            world.set_weather(make_weather(*args.condition.split("_", 1)))
                            camera = FrameCamera(env._world, config.env.observation.camera)
                            env._observer._data_handlers["camera"] = camera
                        archive = collect_episode(env, camera, scenario, seed, target, args.steps,
                                                  dt, args.capture_every, close_gaps=args.close_gaps)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        atomic_write(path, lambda temporary: torch.save(archive, temporary))
                    record_episode(manifest, split, relative, archive, scenario, seed, target)
                    row = manifest["episodes"][-1]
                    print(f"[{len(manifest['episodes'])}/{count}] {split} {scenario.name} "
                          f"target {target:g} m, {row['frames']} matched, "
                          f"{row['gap_m_min']:.1f}–{row['gap_m_max']:.1f} m, {row['ending']}",
                          flush=True)
                    write_json(args.output / "split.json", manifest)
                    seed += 1
    finally:
        if env is not None:
            try:
                env._observer.destroy()
                for actor_id in env._world.actor_ids:
                    env._world.destroy_actor(actor_id)
            finally:
                env._world._set_synchronous_mode(False)
    manifest["complete"] = True
    write_json(args.output / "split.json", manifest)


if __name__ == "__main__":
    main()
