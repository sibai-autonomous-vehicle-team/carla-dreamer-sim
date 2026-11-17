"""Run a longitudinal gap PID in the CARLA car-following environment."""

import argparse
import math
import random
import sys
from pathlib import Path
from car_dreamer.toolkit import get_vehicle_pos

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
        return max(0.0, model((frame - mean) / std).item())

    return estimate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Use CNN gap estimates when provided.")
    parser.add_argument("--carla-site", type=Path, help="Site-packages from the CARLA environment.")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--target-gap", type=float, default=5.0)
    parser.add_argument("--kp", type=float, default=0.5)
    parser.add_argument("--ki", type=float, default=0.0)
    parser.add_argument("--kd", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.episodes < 1 or args.steps < 1 or args.target_gap <= 0:
        parser.error("Episodes, steps, and target gap must be positive.")
    if args.carla_site:
        sys.path.append(str(args.carla_site))

    import numpy as np
    import car_dreamer
    from carla_wrapper import CarlaWrapper
    from generate_dataset import make_weather

    estimate = load_gap_estimator(args.checkpoint) if args.checkpoint else None
    config = car_dreamer.load_task_configs("carla_follow")
    config = config.update({
        "env.display.enable": False,
        "env.world.carla_port": args.port,
        "env.action.discrete": False,
    })
    env = CarlaWrapper(config=config.env)
    world = env._world.carla_world
    dt = world.get_settings().fixed_delta_seconds
    if dt is None or dt <= 0:
        raise RuntimeError("The CARLA world needs a fixed simulation timestep.")
    world.set_weather(make_weather("day", "cloudy"))
    print(f"Gap input: {'CNN' if estimate else 'CARLA ground truth'} | target {args.target_gap:g} m")

    for episode in range(1, args.episodes + 1):
        random.seed(args.seed + episode - 1)
        np.random.seed(args.seed + episode - 1)
        observation, _ = env.reset()
        pid = GapPID(args.target_gap, args.kp, args.ki, args.kd)
        true_gap = env.get_distance()
        minimum_gap = true_gap
        total_error = 0.0
        reason = "step_limit"
        for step in range(1, args.steps + 1):
            # Give the camera time to produce an image after reset in both modes.
            if step <= 20 and not np.any(observation["visual"]):
                command = 0.0
            elif estimate and not np.any(observation["visual"]):
                raise RuntimeError("CARLA returned an empty camera image.")
            else:
                measured_gap = estimate(observation["visual"]) if estimate else true_gap
                command = pid.update(measured_gap, dt)
                if estimate is not None and episode == 1 and step in (6, 25, 60, 110):
                    from PIL import Image

                    folder = Path("cnn_debug_frames")
                    folder.mkdir(exist_ok=True)
                    filename = f"step_{step:03d}_true_{true_gap:.2f}_pred_{measured_gap:.2f}.png"
                    Image.fromarray(observation["visual"].copy()).save(folder / filename)
                    print(f"Saved {folder / filename}")

                print(
                    f"step={step:03d} | true={true_gap:.2f} m | "
                    f"estimated={measured_gap:.2f} m | command={command:+.2f}"
                )

            action = np.array([command, steering_command(env)], dtype=np.float32)
            observation, _, done, info = env.step(action)
            ego = env.get_ego_vehicle()
            velocity = ego.get_velocity()
            waypoint_dist = env.get_wpt_dist(get_vehicle_pos(ego))
            print(
                f"waypoint_dist={waypoint_dist:.2f} m | "
                f"limit={env._config.terminal.out_lane_thres:.2f} m | "
                f"steer={ego.get_control().steer:+.3f} | "
                f"speed={math.hypot(velocity.x, velocity.y):.2f} m/s"
            )
            true_gap = env.get_distance()
            minimum_gap = min(minimum_gap, true_gap)
            total_error += abs(true_gap - args.target_gap)
            if env.is_collision():
                reason = "collision"
                break
            if done:
                reason = ", ".join(
                    key for key in ("destination_reached", "out_of_lane",
                                    "terminal_dist", "time_exceeded", "is_collision")
                    if bool(info.get(key, False))
                ) or "environment_done"
                break

        print(
            f"Episode {episode}: {step} steps | min true gap {minimum_gap:.2f} m | "
            f"mean gap tracking error {total_error / step:.2f} m | end: {reason}"
        )


if __name__ == "__main__":
    main()
