"""Run the frozen gap controllers under one synchronous CARLA protocol."""

import math
import random
from pathlib import Path

import carla
import numpy as np

from analysis.audit_distance_cnn_sync import FrameCamera, TERMINAL_KEYS
from analysis.collect_distance_cnn_sync import horizontal_speed, speed_command
from car_dreamer import load_task_configs
from generate_dataset import make_weather
from run_gap_pid import GapPID, load_gap_estimator
from shared_follow_route import Scenario, SharedRouteFollowing
from training.closed_loop import read_snapshot

TARGET_GAP_M = 6.0
DISTANCE_LIMIT_M = 10.0
LANE_LIMIT_M = 5.0
PID_GAINS = (0.9, 0.05, 0.15)
WARMUP_STEPS = 5
LEAD_PROFILES = ("steady", "pull_away", "brake_restart")


def duration_steps(seconds, dt):
    """Round up to complete ticks, tolerating the simulator's float timestep."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("The duration must be positive and finite.")
    ratio = seconds / dt
    nearest = round(ratio)
    if nearest >= 1 and math.isclose(ratio, nearest, rel_tol=1e-6):
        return nearest
    return math.ceil(ratio)


class SelectorFollowing(SharedRouteFollowing):
    """Same route steering, with deterministic lead speed profiles."""

    lead_profile = "steady"

    def get_nonego_vehicle_control(self):
        if self.lead_profile == "steady":
            return super().get_nonego_vehicle_control()

        # The common steady lead behavior also applies during neutral warmup.
        control_step = self._time_step - WARMUP_STEPS
        if control_step < 0:
            return super().get_nonego_vehicle_control()
        dt = self._config.world.fixed_delta_seconds
        time_s = control_step * dt
        if self.lead_profile == "pull_away":
            # An immediate strong acceleration from rest exercises upper-gap exits.
            target = 4.0 if time_s % 8.0 < 2.4 else 2.0
        else:
            # Brief repeated braking and restarts exercise short gaps.
            phase = time_s % 6.0
            target = 0.0 if 0.9 <= phase < 1.6 else 2.0
        command = speed_command(horizontal_speed(self.nonego), target)
        return carla.VehicleControl(
            throttle=max(0.0, command / 3),
            brake=max(0.0, -command / 3),
            steer=self.route.steer(self.nonego),
        )


def starting_state(env, snapshot):
    frame, time_s, pose, gap = snapshot
    actors = (env.ego, env.nonego)
    velocities = [actor.get_velocity() for actor in actors]
    return {
        "frame": frame,
        "simulation_time_s": time_s,
        "state": pose.tolist(),
        "initial_gap_m": gap,
        "actor_types": [actor.type_id for actor in actors],
        "velocities": [[velocity.x, velocity.y, velocity.z] for velocity in velocities],
    }


class RolloutEngine:
    """Reuse the same simulator, estimator preprocessing and PID for every mode."""

    def __init__(self, port=2000, *, max_camera_age_steps=5):
        if max_camera_age_steps < 0:
            raise ValueError("Maximum camera age cannot be negative.")
        config = load_task_configs("carla_follow").update({
            "env.display.enable": False,
            "env.world.carla_port": port,
            "env.action.discrete": False,
            "env.terminal.terminal_dist": DISTANCE_LIMIT_M,
            "env.terminal.out_lane_thres": LANE_LIMIT_M,
            "env.observation.enabled": ["camera", "collision"],
        })
        self.env = SelectorFollowing(config=config.env)
        self.camera = FrameCamera(self.env._world, config.env.observation.camera)
        self.env._observer._data_handlers["camera"] = self.camera
        self.world = self.env._world.carla_world
        self.dt_s = self.world.get_settings().fixed_delta_seconds
        if self.dt_s is None or not math.isfinite(self.dt_s) or self.dt_s <= 0:
            raise RuntimeError("CARLA must use a positive fixed timestep.")
        self.max_camera_age_steps = max_camera_age_steps
        self.estimators = {}

    def close(self):
        try:
            self.env._observer.destroy()
            for actor_id in list(self.env._world.actor_ids):
                self.env._world.destroy_actor(actor_id)
        finally:
            self.env._world._set_synchronous_mode(False)

    def estimator(self, path):
        path = Path(path).resolve()
        if path not in self.estimators:
            self.estimators[path] = load_gap_estimator(path)
        return self.estimators[path]

    def step(self, command, current):
        steer = self.env.ego_steer_command()
        _, _, done, info = self.env.step(np.array([command, steer], dtype=np.float32))
        following = read_snapshot(self.env, self.dt_s)
        if (following[0] != current[0] + 1
                or not math.isclose(following[1] - current[1], self.dt_s, abs_tol=1e-4)):
            raise RuntimeError("Nonconsecutive CARLA snapshots during the rollout.")
        flags = [key for key in TERMINAL_KEYS if bool(info.get(key, False))]
        if self.env.is_collision() and "is_collision" not in flags:
            flags.append("is_collision")
        if following[3] > DISTANCE_LIMIT_M and "terminal_dist" not in flags:
            flags.append("terminal_dist")
        return following, flags, bool(done or flags), steer

    def run(self, case, mode, checkpoints, selector=None, horizon_s=2.0,
            decision_s=2.0, max_steps=360, *, windows=None):
        condition = case["condition"]
        seed = int(case["seed"])
        lead_profile = case["lead_profile"]
        if lead_profile not in LEAD_PROFILES:
            raise ValueError(f"Unknown lead profile {lead_profile!r}.")
        if mode == "selector" and selector is None:
            raise ValueError("Selector mode requires a fitted selector.")
        active = condition if mode in ("specialist", "weather_matched") else mode
        predict_context = self.estimator(checkpoints["generalist"])
        if mode != "selector":
            self.estimator(checkpoints[active])

        horizon_steps = duration_steps(horizon_s, self.dt_s)
        decision_steps = duration_steps(decision_s, self.dt_s)
        if windows is not None:
            if windows < 1:
                raise ValueError("The number of collection windows must be positive.")
            max_steps = windows * horizon_steps
        if max_steps < 1:
            raise ValueError("The rollout must contain at least one control step.")

        random.seed(seed)
        np.random.seed(seed)
        route = case["route"]
        if isinstance(route, dict):
            route = Scenario(**{**route, "branch_roads": tuple(route.get("branch_roads", ()))})
        self.env.select_scenario(route)
        self.env.lead_start_index = int(case["lead_start_index"])
        self.env.lead_profile = lead_profile
        self.world.set_weather(make_weather(*condition.split("_", 1)))
        self.env.reset()
        current = read_snapshot(self.env, self.dt_s)
        initial = starting_state(self.env, current)
        snapshots = {current[0]: current[3]}
        result = {
            "case": case,
            "mode": mode,
            "initialization": initial,
            "dt_s": self.dt_s,
            "horizon_s": horizon_s,
            "horizon_steps": horizon_steps,
            "effective_horizon_s": horizon_steps * self.dt_s,
            "decision_interval_s": decision_s,
            "decision_steps": decision_steps,
            "target_gap_m": TARGET_GAP_M,
            "distance_limit_m": DISTANCE_LIMIT_M,
            "lane_limit_m": LANE_LIMIT_M,
            "gains": list(PID_GAINS),
            "warmup_steps": WARMUP_STEPS,
            "max_camera_age_steps": self.max_camera_age_steps,
            "route_centerline_xy": self.env.route.xy.tolist(),
            "rows": [],
            "warmup_terminal": False,
            "terminal_reason": "step_limit",
            "terminal_flags": [],
        }

        # Allow the sensor pipeline to warm up without PID actions, as in fixed PID.
        for _ in range(WARMUP_STEPS):
            current, flags, done, _ = self.step(0.0, current)
            snapshots[current[0]] = current[3]
            if done:
                result["warmup_terminal"] = True
                result["terminal_flags"] = flags
                result["terminal_reason"] = ", ".join(flags) or "environment_done"
                return result
        initial["control_start"] = starting_state(self.env, current)
        initial["first_pid_gap_m"] = current[3]

        pid = GapPID(TARGET_GAP_M, *PID_GAINS)
        latest = None
        predictions = {}
        active = None if mode == "selector" else active
        probability = None
        start_time_s = current[1]
        for index in range(max_steps):
            frame, simulation_time_s, _, true_gap = current
            for capture_frame, image in self.camera.drain():
                if capture_frame > frame:
                    raise RuntimeError("Camera capture came from a future simulator frame.")
                if (capture_frame in snapshots and image.any()
                        and (latest is None or capture_frame > latest[0])):
                    latest = (capture_frame, image)
                    predictions.clear()
            if latest is None or frame - latest[0] > self.max_camera_age_steps:
                result["terminal_reason"] = "camera_unavailable"
                break
            capture_frame, image = latest
            if "generalist" not in predictions:
                predictions["generalist"] = float(predict_context(image))
            context_gap = predictions["generalist"]
            if not math.isfinite(context_gap):
                result["terminal_reason"] = "invalid_prediction"
                break

            probabilities = None
            switched = False
            if mode == "selector" and index % decision_steps == 0:
                probabilities = selector.score(condition, context_gap)
                selected = max(probabilities, key=probabilities.get)
                switched = active is not None and selected != active
                if switched:
                    # Retain controller memory, except for a one-tick derivative kick.
                    pid.previous_error = None
                active = selected
                probability = float(probabilities[active])
            if active not in predictions:
                predictions[active] = float(self.estimator(checkpoints[active])(image))
            predicted_gap = predictions[active]
            if not math.isfinite(predicted_gap):
                result["terminal_reason"] = "invalid_prediction"
                break
            command = pid.update(predicted_gap, self.dt_s)
            next_snapshot, flags, done, steer = self.step(command, current)
            control = self.env.ego.get_control()
            threshold = getattr(selector, "threshold", None)
            result["rows"].append({
                "step": index + 1,
                "time_s": simulation_time_s - start_time_s,
                "decision_frame": frame,
                "next_frame": next_snapshot[0],
                "capture_frame": capture_frame,
                "capture_age_steps": frame - capture_frame,
                "true_gap_m": true_gap,
                "post_step_true_gap_m": next_snapshot[3],
                "capture_true_gap_m": snapshots[capture_frame],
                "generalist_gap_m": context_gap,
                "specialist_gap_m": predicted_gap,
                "predicted_gap_m": predicted_gap,
                "controller": active,
                "probabilities": probabilities,
                "selected_probability": probability,
                "low_confidence": probability is not None and threshold is not None and probability < threshold,
                "controller_switched": switched,
                "fallback_used": False,
                "pid_command": command,
                "throttle": float(control.throttle),
                "brake": float(control.brake),
                "steering_command": steer,
                "flags": flags,
                **self.env.route_measurements(),
            })
            current = next_snapshot
            snapshots[current[0]] = current[3]
            if done:
                result["terminal_flags"] = flags
                result["terminal_reason"] = ", ".join(flags) or "environment_done"
                break
        return result
