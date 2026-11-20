"""Run a longitudinal gap PID in the CARLA car-following environment."""

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path


PID_DEFAULTS = {
    "episodes": 3,
    "steps": 200,
    "warmup_steps": 0,
    "target_gap": 5.0,
    "kp": 0.5,
    "ki": 0.0,
    "kd": 0.5,
    "seed": 42,
}


class GapPID:
    def __init__(self, target, kp, ki, kd):
        self.target = target
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0.0
        self.previous_error = None

    def update(self, gap, dt):
        error = gap - self.target
        derivative = (
            0.0 if self.previous_error is None else (error - self.previous_error) / dt
        )
        self.previous_error = error
        next_integral = self.integral + error * dt
        command = self.kp * error + self.ki * next_integral + self.kd * derivative
        # Stop the integral from growing in the direction of saturation.
        if self.ki and (abs(command) <= 3.0 or command * error < 0.0):
            self.integral = next_integral
        return max(-3.0, min(3.0, command))


def steering_command(env):
    ego = env.ego.get_transform()
    lead = env.nonego.get_location()
    heading = math.atan2(lead.y - ego.location.y, lead.x - ego.location.x)
    error = (heading - math.radians(ego.rotation.yaw) + math.pi) % (2 * math.pi) - math.pi
    return max(-1.0, min(1.0, -0.75 * error))


def load_gap_estimator(checkpoint_path):
    import torch
    from torch import nn
    from torchvision.models import resnet18

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def estimate(image):
        frame = torch.tensor(image.copy(), device=device, dtype=torch.float32)
        frame = frame.permute(2, 0, 1).unsqueeze(0) / 255.0
        return model((frame - mean) / std).item()

    return estimate



def configure_rollout(parser, args):
    profile = {}
    terminal = {}
    if args.specialist and args.pid_profile:
        parser.error("Choose --specialist or --pid-profile")
    profile_path = args.specialist / "pid.json" if args.specialist else args.pid_profile
    if profile_path:
        try:
            profile = json.loads(profile_path.read_text())
            if not isinstance(profile, dict):
                raise ValueError("expected a JSON object")
            condition = profile["condition"]
            if not isinstance(condition, str):
                raise ValueError("invalid condition")
            if args.condition and args.condition != condition:
                raise ValueError("use --checkpoint with --condition for cross-condition testing")
            args.condition = condition
            if args.specialist:
                args.checkpoint = args.specialist / "cnn.pth"
                if not args.checkpoint.is_file():
                    raise FileNotFoundError(f"No checkpoint in {args.specialist}")
            shared_route = profile.get("shared_route", False)
            if not isinstance(shared_route, bool):
                raise ValueError("shared_route must be a JSON boolean")
            args.shared_route = args.shared_route or shared_route
            if args.route_manifest is None and profile.get("route_manifest"):
                args.route_manifest = profile_path.parent / profile["route_manifest"]
            for field in ("terminal_dist", "out_lane_thres"):
                value = profile[field]
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{field} must be positive and finite")
                terminal[f"env.terminal.{field}"] = value
        except (OSError, KeyError, ValueError, TypeError) as error:
            parser.error(f"Invalid specialist profile: {error}")
    args.condition = args.condition or "day_cloudy"
    for field, default in PID_DEFAULTS.items():
        value = getattr(args, field)
        if value is None:
            value = profile.get(field, default)
        valid_types = (int,) if isinstance(default, int) else (int, float)
        if type(value) not in valid_types or not math.isfinite(value):
            parser.error(f"Invalid value for {field.replace('_', '-')}")
        setattr(args, field, value)
    return terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    checkpoints = parser.add_mutually_exclusive_group()
    checkpoints.add_argument("--checkpoint", type=Path, help="Use CNN gap estimates when provided.")
    checkpoints.add_argument("--diagnostic-checkpoint", type=Path,
                        help="Log CNN estimates while controlling with ground truth.")
    checkpoints.add_argument("--specialist", type=Path,
                             help="Directory with cnn.pth and its condition and PID settings.")
    parser.add_argument("--pid-profile", type=Path,
                        help="Use PID settings independently of the checkpoint.")
    parser.add_argument("--condition", help="Weather and lighting, such as sunset_cloudy.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--shared-route", action="store_true",
                        help="Use fixed Town03 routes with an explicit ego finish.")
    parser.add_argument("--lead-start-index", type=int,
                        help="Place the lead at this shared-route waypoint instead of waypoint 10.")
    parser.add_argument("--route-manifest", type=Path,
                        help="Load fixed route specifications from a grouped JSON manifest.")
    parser.add_argument("--route-split", choices=("train", "validation", "test"),
                        default="test", help="Which manifest routes to run.")
    parser.add_argument("--carla-site", type=Path, help="Site-packages from the CARLA environment.")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--warmup-steps", type=int,
                        help="Apply neutral longitudinal control for this many initial steps.")
    parser.add_argument("--target-gap", type=float)
    parser.add_argument("--kp", type=float)
    parser.add_argument("--ki", type=float)
    parser.add_argument("--kd", type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    terminal_overrides = configure_rollout(parser, args)
    if args.episodes < 1 or args.steps < 1 or args.target_gap <= 0:
        parser.error("Episodes, steps, and target gap must be positive.")
    if not 0 <= args.warmup_steps < args.steps:
        parser.error("--warmup-steps must be between zero and --steps minus one")
    if args.lead_start_index is not None and (args.lead_start_index < 1 or not args.shared_route):
        parser.error("--lead-start-index requires --shared-route and a positive waypoint index")
    if args.route_manifest and not args.shared_route:
        parser.error("--route-manifest requires --shared-route")
    if args.carla_site:
        sys.path.append(str(args.carla_site))

    import numpy as np
    import car_dreamer
    from carla_wrapper import CarlaWrapper
    from generate_dataset import make_weather

    try:
        weather = make_weather(*args.condition.split("_", 1))
    except (TypeError, ValueError) as error:
        parser.error(f"Invalid --condition {args.condition!r}: {error}")

    checkpoint = args.checkpoint or args.diagnostic_checkpoint
    estimate = load_gap_estimator(checkpoint) if checkpoint else None
    config = car_dreamer.load_task_configs("carla_follow")
    config = config.update({
        "env.display.enable": False,
        "env.world.carla_port": args.port,
        "env.action.discrete": False,
        **terminal_overrides,
    })
    env_type = CarlaWrapper
    if args.shared_route:
        from shared_follow_route import SCENARIOS, Scenario, SharedRouteFollowing

        scenarios = SCENARIOS
        if args.route_manifest:
            try:
                specs = json.loads(args.route_manifest.read_text())[args.route_split]
                scenarios = tuple(Scenario(**spec) for spec in specs)
            except (OSError, KeyError, ValueError, TypeError) as error:
                parser.error(f"Invalid route manifest: {error}")
            if not scenarios:
                parser.error(f"Route split {args.route_split!r} is empty")
        env_type = SharedRouteFollowing
    env = env_type(config=config.env)
    if args.lead_start_index is not None:
        env.lead_start_index = args.lead_start_index
    world = env._world.carla_world
    dt = world.get_settings().fixed_delta_seconds
    if dt is None or dt <= 0:
        raise RuntimeError("The CARLA world needs a fixed simulation timestep.")
    world.set_weather(weather)
    print(f"Condition: {args.condition} | sun altitude {weather.sun_altitude_angle:g} degrees")
    terminal = env._config.terminal
    print(f"Active thresholds: true gap > {terminal.terminal_dist:g} m, "
          f"route deviation > {terminal.out_lane_thres:g} m, "
          f"environment steps > {terminal.time_limit}, dt={dt:g} s")
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
    print(f"Gap input: {'CNN' if args.checkpoint else 'CARLA ground truth'} | target {args.target_gap:g} m")

    for episode in range(1, args.episodes + 1):
        random.seed(args.seed + episode - 1)
        np.random.seed(args.seed + episode - 1)
        if args.shared_route:
            scenario = scenarios[(episode - 1) % len(scenarios)]
            env.select_scenario(scenario)
            print(f"Route: {scenario.name} | ego finish {scenario.finish_m:g} m along route")
        observation, state = env.reset()
        pid = GapPID(args.target_gap, args.kp, args.ki, args.kd)
        shadow = GapPID(args.target_gap, args.kp, args.ki, args.kd)
        true_gap = env.get_distance()
        minimum_gap = true_gap
        maximum_gap = true_gap
        total_error = 0.0
        reason = "step_limit"
        rows = []
        initialization = {
            "state": np.asarray(state).tolist(),
            "actor_types": [env.ego.type_id, env.nonego.type_id],
            "velocities": [[v.x, v.y, v.z] for v in
                           (env.ego.get_velocity(), env.nonego.get_velocity())],
            "initial_gap_m": true_gap,
        }
        for step in range(1, args.steps + 1):
            valid_camera = bool(np.any(observation["visual"]))
            predicted = estimate(observation["visual"]) if estimate and valid_camera else None
            alternate = None
            if step <= args.warmup_steps or (not args.warmup_steps and not valid_camera and step <= 20):
                command = 0.0
            elif not valid_camera and estimate:
                reason = "empty_camera"
                break
            else:
                if "first_pid_step" not in initialization:
                    initialization["first_pid_step"] = step
                    initialization["first_pid_gap_m"] = true_gap
                measured_gap = predicted if args.checkpoint else true_gap
                command = pid.update(measured_gap, dt)
                if predicted is not None:
                    alternate = shadow.update(true_gap if args.checkpoint else predicted, dt)

            steer = env.ego_steer_command() if args.shared_route else steering_command(env)
            action = np.array([command, steer], dtype=np.float32)
            observation, _, done, info = env.step(action)
            ego = env.get_ego_vehicle()
            control = ego.get_control()
            next_gap = env.get_distance()
            flags = [key for key in ("destination_reached", "out_of_lane", "terminal_dist",
                                     "time_exceeded", "is_collision")
                     if bool(info.get(key, False))]
            if env.is_collision() and "is_collision" not in flags:
                flags.append("is_collision")
            if flags or done:
                reason = ", ".join(flags) or "environment_done"
            terminal_reason = reason if flags or done or step == args.steps else ""
            rows.append({
                "step": step, "time_s": (step - 1) * dt,
                "true_gap_m": true_gap, "predicted_gap_m": predicted,
                "pid_command": command, "alternate_pid_command": alternate,
                "throttle": control.throttle, "brake": control.brake,
                "post_step_true_gap_m": next_gap, "terminal_reason": terminal_reason,
            })
            if args.shared_route:
                rows[-1].update(env.route_measurements(), steering_command=steer)
            print(f"step={step:03d} | true={true_gap:.3f} m | CNN={predicted} | "
                  f"command={command:+.3f} | next true={next_gap:.3f} m")
            true_gap = next_gap
            minimum_gap = min(minimum_gap, true_gap)
            maximum_gap = max(maximum_gap, true_gap)
            total_error += abs(true_gap - args.target_gap)
            if terminal_reason:
                break

        step = len(rows)
        if not step:
            raise RuntimeError(f"Episode {episode} ended before an action: {reason}")
        if args.output:
            rows[-1]["terminal_reason"] = reason
            stem = f"seed_{args.seed + episode - 1}"
            with (args.output / f"{stem}.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            summary = {
                "seed": args.seed + episode - 1, "initialization": initialization,
                "condition": args.condition, "checkpoint": str(checkpoint) if checkpoint else None,
                "steps": step, "reason": reason, "min_true_gap_m": minimum_gap,
                "max_true_gap_m": maximum_gap, "tracking_mae_m": total_error / step,
                "distance_limit_m": terminal.terminal_dist, "lane_limit_m": terminal.out_lane_thres,
                "target_m": args.target_gap, "dt_s": dt, "gains": [args.kp, args.ki, args.kd],
                "warmup_steps": args.warmup_steps, "max_steps": args.steps,
            }
            if args.shared_route:
                summary.update(
                    scenario=scenario.name, finish_m=scenario.finish_m,
                    lead_start_index=env.lead_start_index,
                    route_centerline_xy=env.route.xy.tolist(),
                    final_route=env.route_measurements(),
                    max_ego_route_error_m=max(row["ego_route_error_m"] for row in rows),
                )
            (args.output / f"{stem}.json").write_text(json.dumps(summary, indent=2) + "\n")

        print(
            f"Episode {episode}: {step} steps | min true gap {minimum_gap:.2f} m | "
            f"mean gap tracking error {total_error / step:.2f} m | end: {reason}"
        )


if __name__ == "__main__":
    main()
