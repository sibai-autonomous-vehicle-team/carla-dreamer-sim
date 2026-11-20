"""Validate and report paired CARLA rollouts for a lighting/weather specialist."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = tuple(f"{time}_{weather}" for time in ("day", "sunset", "night")
                   for weather in ("sunny", "cloudy", "rainy"))
SEEDS = range(42, 47)
PAIR_FIELDS = (
    "condition", "target_m", "dt_s", "gains", "distance_limit_m", "lane_limit_m",
    "lead_start_index", "warmup_steps", "max_steps",
)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read_rollout(path, seed, route, mode, condition):
    summary = json.loads(path.with_suffix(".json").read_text())
    with path.with_suffix(".csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    check(summary["seed"] == seed and summary["scenario"] == route["name"],
          f"Seed or route mismatch in {path}")
    check(summary["condition"] == condition and summary["finish_m"] == route["finish_m"],
          f"Condition or finish mismatch in {path}")
    check(bool(rows) and len(rows) == summary["steps"]
          and rows[-1]["terminal_reason"] == summary["reason"],
          f"Trace and terminal summary differ in {path}")
    check([int(row["step"]) for row in rows] == list(range(1, len(rows) + 1))
          and not any(row["terminal_reason"] for row in rows[:-1]),
          f"Nonconsecutive steps or an early terminal flag in {path}")

    gaps = np.array([float(row["true_gap_m"]) for row in rows])
    successor_gaps = np.array([float(row["post_step_true_gap_m"]) for row in rows])
    final_gaps = np.append(gaps, successor_gaps[-1])
    time_s = np.array([float(row["time_s"]) for row in rows])
    commands = np.array([float(row["pid_command"]) for row in rows])
    pred = np.array([float(row["predicted_gap_m"]) if row["predicted_gap_m"] else np.nan
                     for row in rows])
    available = np.array([bool(row["predicted_gap_m"]) for row in rows])
    check(np.isfinite(summary["dt_s"]) and summary["dt_s"] > 0
          and np.allclose(time_s, np.arange(len(rows)) * summary["dt_s"]),
          f"Decision times disagree with timestep in {path}")
    check(np.isfinite(final_gaps).all() and (final_gaps >= 0).all()
          and np.allclose(gaps[1:], successor_gaps[:-1])
          and np.isclose(gaps[0], summary["initialization"]["initial_gap_m"])
          and np.isclose(final_gaps.min(), summary["min_true_gap_m"])
          and np.isclose(final_gaps.max(), summary["max_true_gap_m"])
          and np.isclose(np.abs(successor_gaps - summary["target_m"]).mean(),
                         summary["tracking_mae_m"]),
          f"Recorded gap statistics do not agree with trace in {path}")
    check(np.isfinite(commands).all() and np.all(np.abs(commands) <= 3)
          and np.isfinite(pred[available]).all(),
          f"Nonfinite prediction or invalid command in {path}")
    checkpoint = summary.get("checkpoint")
    if mode == "ground_truth":
        check(not checkpoint, f"Ground-truth trace references a checkpoint in {path}")
    else:
        check(isinstance(checkpoint, str) and bool(checkpoint),
              f"Missing CNN checkpoint in {path}")
        first_pid = summary["initialization"].get("first_pid_step")
        if first_pid is not None:
            check(1 <= first_pid <= len(rows) and available[first_pid - 1],
                  f"Missing CNN prediction at first PID step in {path}")
    return summary, time_s, final_gaps, pred, commands


def compare_start(current, reference):
    current_pose = np.array(current["state"]).reshape(2, 6)
    reference_pose = np.array(reference["state"]).reshape(2, 6)
    position_delta = current_pose[:, :3] - reference_pose[:, :3]
    rotation_delta = (current_pose[:, 3:] - reference_pose[:, 3:] + 180) % 360 - 180
    first_gap_delta = None
    if "first_pid_gap_m" in current and "first_pid_gap_m" in reference:
        first_gap_delta = abs(current["first_pid_gap_m"] - reference["first_pid_gap_m"])
    return {
        "same_vehicle_types_as_gt": current["actor_types"] == reference["actor_types"],
        "initial_gap_delta_vs_gt_m": abs(current["initial_gap_m"] - reference["initial_gap_m"]),
        "max_initial_position_delta_vs_gt_m": float(np.linalg.norm(position_delta, axis=1).max()),
        "max_initial_orientation_delta_vs_gt_deg": float(abs(rotation_delta).max()),
        "max_initial_velocity_delta_vs_gt_mps": float(np.linalg.norm(
            np.array(current["velocities"]) - np.array(reference["velocities"]), axis=1).max()),
        "same_first_pid_step_as_gt": current.get("first_pid_step") == reference.get("first_pid_step"),
        "first_pid_gap_delta_vs_gt_m": first_gap_delta,
    }


def row_result(summary, errors, commands, mode):
    flags = set(summary["reason"].split(", "))
    final = summary["final_route"]
    return {
        "seed": summary["seed"], "scenario": summary["scenario"], "mode": mode,
        "condition": summary["condition"], "checkpoint": summary.get("checkpoint"),
        "finished": (flags == {"destination_reached"}
                     and final["ego_progress_m"] >= summary["finish_m"]
                     and final["ego_route_error_m"] <= 1.0),
        "reason": summary["reason"], "steps": summary["steps"],
        "ego_progress_m": final["ego_progress_m"], "goal_progress_m": summary["finish_m"],
        "min_true_gap_m": summary["min_true_gap_m"], "max_true_gap_m": summary["max_true_gap_m"],
        "tracking_mae_m": summary["tracking_mae_m"],
        "collision": "is_collision" in flags, "distance_violation": "terminal_dist" in flags,
        "route_violation": "out_of_lane" in flags,
        "timeout": bool({"step_limit", "time_exceeded"} & flags),
        "empty_camera": "empty_camera" in flags,
        "min_pid_command": float(commands.min()),
        "max_pid_command": float(commands.max()),
        "mean_abs_pid_command": float(np.abs(commands).mean()),
        "throttle_steps": int((commands > 0).sum()),
        "brake_steps": int((commands < 0).sum()),
        "saturated_pid_steps": int((np.abs(commands) == 3).sum()),
        "prediction_decisions": len(errors),
        "decision_prediction_mae_m": float(np.abs(errors).mean()) if len(errors) else None,
        "decision_prediction_bias_m": float(errors.mean()) if len(errors) else None,
    }


def mode_totals(rows, errors):
    return {
        "episodes": len(rows),
        "steps": sum(row["steps"] for row in rows),
        "finishes": sum(row["finished"] for row in rows),
        "collisions": sum(row["collision"] for row in rows),
        "distance_violations": sum(row["distance_violation"] for row in rows),
        "route_violations": sum(row["route_violation"] for row in rows),
        "timeouts": sum(row["timeout"] for row in rows),
        "empty_camera": sum(row["empty_camera"] for row in rows),
        "min_true_gap_m": min(row["min_true_gap_m"] for row in rows),
        "max_true_gap_m": max(row["max_true_gap_m"] for row in rows),
        "mean_run_tracking_mae_m": float(np.mean([row["tracking_mae_m"] for row in rows])),
        "min_pid_command": min(row["min_pid_command"] for row in rows),
        "max_pid_command": max(row["max_pid_command"] for row in rows),
        "throttle_steps": sum(row["throttle_steps"] for row in rows),
        "brake_steps": sum(row["brake_steps"] for row in rows),
        "saturated_pid_steps": sum(row["saturated_pid_steps"] for row in rows),
        "prediction_decisions": len(errors),
        "decision_prediction_mae_m": float(np.abs(errors).mean()) if len(errors) else None,
        "decision_prediction_bias_m": float(errors.mean()) if len(errors) else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--routes", type=Path, help="Route manifest (default: EXPERIMENT/routes.json)")
    args = parser.parse_args()
    live = args.experiment / "live"
    route_file = args.routes or args.experiment / "routes.json"
    routes = json.loads(route_file.read_text())["test"]
    check(len(routes) == len(SEEDS), "Expected five test scenarios for seeds 42–46")
    check(len({route["name"] for route in routes}) == len(routes), "Duplicate test route names")
    condition_label = args.condition.replace("_", " ").capitalize()
    modes = {
        "ground_truth": "Ground truth",
        "day_transfer": "Daytime-cloudy reference",
        "specialist": f"{condition_label} specialist",
    }
    records = []
    errors_by_mode = {mode: [] for mode in modes}
    checkpoints = {}
    settings = None
    figure, axes = plt.subplots(len(routes), 2, figsize=(12, 13), layout="constrained")
    for seed, route, (gap_axis, command_axis) in zip(SEEDS, routes, axes):
        reference = None
        for mode, label in modes.items():
            path = live / mode / f"seed_{seed}"
            summary, times, gaps, predictions, commands = read_rollout(
                path, seed, route, mode, args.condition,
            )
            current_settings = {key: summary[key] for key in PAIR_FIELDS}
            if settings is None:
                settings = current_settings
            check(current_settings == settings, f"Unmatched condition, gates or controls in {path}")
            checkpoint = summary.get("checkpoint")
            check(checkpoints.setdefault(mode, checkpoint) == checkpoint,
                  f"Checkpoint differs within mode {mode}")
            if reference is None:
                reference = summary
            check(summary["route_centerline_xy"] == reference["route_centerline_xy"],
                  f"Unmatched route centerline in {path}")
            start = compare_start(summary["initialization"], reference["initialization"])
            check(start["same_vehicle_types_as_gt"], f"Unmatched vehicle types in {path}")
            errors = (predictions - gaps[:-1])[np.isfinite(predictions)]
            errors_by_mode[mode].append(errors)
            records.append(row_result(summary, errors, commands, mode) | start)

            color = gap_axis.plot(np.append(times, times[-1] + summary["dt_s"]), gaps,
                                  label=label)[0].get_color()
            if mode != "ground_truth":
                gap_axis.plot(times, predictions, "--", color=color, alpha=0.65)
            command_axis.plot(times, commands, color=color)
        gap_axis.axhline(settings["distance_limit_m"], linestyle=":", color="tab:red", linewidth=1)
        gap_axis.axhline(settings["target_m"], linestyle=":", color="gray", linewidth=1)
        gap_axis.set(title=f"Seed {seed} · {route['name']}", ylabel="Gap (m)")
        command_axis.set(ylabel="PID command", ylim=(-3.2, 3.2))
        command_axis.axhline(0, color="gray", linewidth=0.6)

    axes[0, 0].legend(fontsize=8)
    axes[-1, 0].set_xlabel("Simulation time (s)")
    axes[-1, 1].set_xlabel("Simulation time (s)")
    figure.suptitle(f"{condition_label} CARLA · dashed lines show CNN predictions")
    figure.savefig(live / "comparison.png", dpi=145)
    plt.close(figure)

    totals = {
        mode: mode_totals([row for row in records if row["mode"] == mode],
                          np.concatenate(errors_by_mode[mode]))
        for mode in modes
    }
    with (live / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (live / "summary.json").write_text(json.dumps(
        {"settings": settings, "mode_labels": modes, "checkpoints": checkpoints,
         "routes": str(route_file), "by_mode": totals, "runs": records},
        indent=2, allow_nan=False) + "\n")
    for mode, stats in totals.items():
        print(f"{modes[mode]} finished {stats['finishes']}/{stats['episodes']} routes")


if __name__ == "__main__":
    main()
