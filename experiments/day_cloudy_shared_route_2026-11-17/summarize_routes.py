"""Check and summarize the five paired shared-route CARLA runs."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SEEDS = range(42, 47)
MODES = ("ground_truth", "cnn")
FAILURE_FLAGS = {"is_collision", "out_of_lane", "terminal_dist", "time_exceeded"}


def load_run(root, mode, seed):
    base = root / mode / f"seed_{seed}"
    summary = json.loads(base.with_suffix(".json").read_text())
    with base.with_suffix(".csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if summary["seed"] != seed or not rows or len(rows) != summary["steps"]:
        raise ValueError(f"{base} has inconsistent seed or step count")
    reason = summary["reason"]
    if rows[-1]["terminal_reason"] != reason or any(r["terminal_reason"] for r in rows[:-1]):
        raise ValueError(f"{base} has inconsistent terminal reasons")
    flags = set(reason.split(", "))
    final = summary["final_route"]
    progress = final["ego_progress_m"]
    route_error = final["ego_route_error_m"]
    ego_arrived = progress >= summary["finish_m"] and route_error <= 1.0
    if ("destination_reached" in flags) != ego_arrived:
        raise ValueError(f"{base} has inconsistent finish flag and ego arrival geometry")
    result = {
        "seed": seed,
        "scenario": summary["scenario"],
        "mode": mode,
        "completed": ego_arrived and not flags & FAILURE_FLAGS,
        "terminal_reason": reason,
        "steps": summary["steps"],
        "collision": "is_collision" in flags,
        "distance_violation": "terminal_dist" in flags,
        "route_violation": "out_of_lane" in flags,
        "timeout": bool({"step_limit", "time_exceeded"} & flags),
        "finish_m": summary["finish_m"],
        "final_ego_progress_m": progress,
        "final_ego_route_error_m": route_error,
        "max_ego_route_error_m": summary["max_ego_route_error_m"],
        "min_true_gap_m": summary["min_true_gap_m"],
        "max_true_gap_m": summary["max_true_gap_m"],
        "tracking_mae_m": summary["tracking_mae_m"],
    }
    return summary, result, rows


def compare_starts(seed, truth, cnn):
    fields = (
        "scenario", "finish_m", "route_centerline_xy", "gains",
        "distance_limit_m", "lane_limit_m", "target_m", "dt_s",
    )
    for field in fields:
        if truth[field] != cnn[field]:
            raise ValueError(f"Seed {seed} has mismatched {field}")
    a, b = (run["initialization"] for run in (truth, cnn))
    if a["actor_types"] != b["actor_types"]:
        raise ValueError(f"Seed {seed} has mismatched vehicles")
    pose_delta = np.asarray(a["state"]).reshape(2, 6) - np.asarray(b["state"]).reshape(2, 6)
    angle_delta = (pose_delta[:, 3:] + 180) % 360 - 180
    velocity_delta = np.asarray(a["velocities"]) - np.asarray(b["velocities"])
    gap_a, gap_b = a.get("first_pid_gap_m"), b.get("first_pid_gap_m")
    return {
        "seed": seed,
        "scenario": truth["scenario"],
        "same_route_config_and_vehicle_types": True,
        "max_initial_position_delta_m": float(np.linalg.norm(pose_delta[:, :3], axis=1).max()),
        "max_initial_orientation_delta_deg": float(np.abs(angle_delta).max()),
        "max_initial_velocity_delta_mps": float(np.linalg.norm(velocity_delta, axis=1).max()),
        "initial_gap_delta_m": abs(a["initial_gap_m"] - b["initial_gap_m"]),
        "same_first_pid_step": a.get("first_pid_step") == b.get("first_pid_step"),
        "first_pid_gap_delta_m": abs(gap_a - gap_b) if gap_a is not None and gap_b is not None else None,
    }


def numbers(rows, key):
    return np.array([float(row[key]) if row[key] else np.nan for row in rows])


def plot_pair(root, seed, loaded):
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True, sharey="row",
                             layout="constrained")
    for (summary, result, rows), (gap, commands, progress) in zip(loaded, axes.T):
        t = numbers(rows, "time_s")
        dt = summary["dt_s"]
        t_after = t + dt
        true_gap = numbers(rows, "post_step_true_gap_m")
        gap.plot(np.r_[t[0], t_after], np.r_[float(rows[0]["true_gap_m"]), true_gap],
                 label="True gap")
        predicted = numbers(rows, "predicted_gap_m")
        if np.isfinite(predicted).any():
            gap.plot(t, predicted, label="CNN estimate")
        gap.axhline(summary["target_m"], ls="--", c="0.3", lw=1, label="Target")
        gap.axhline(summary["distance_limit_m"], ls=":", c="tab:red", label="Gap cutoff")
        mode_label = "CNN" if result["mode"] == "cnn" else "Ground truth"
        reason_label = result["terminal_reason"].replace("_", " ")
        if result["completed"]:
            reason_label = "Ego arrived"
        elif result["distance_violation"]:
            reason_label = "Gap cutoff"
        gap.set_title(f"{mode_label} · {reason_label}")
        gap.set_ylabel("Gap (m)")
        gap.legend(fontsize="small")

        commands.plot(t, numbers(rows, "pid_command"), label="Applied")
        alternate = numbers(rows, "alternate_pid_command")
        if np.isfinite(alternate).any():
            commands.plot(t, alternate, ls="--", alpha=0.7, label="Other input, same trajectory")
        commands.axhline(0, c="0.3", lw=0.7)
        commands.set(ylabel="Longitudinal command", ylim=(-3.2, 3.2))
        commands.legend(fontsize="small")

        progress.plot(t_after, numbers(rows, "ego_progress_m"), label="Ego")
        progress.plot(t_after, numbers(rows, "lead_progress_m"), label="Lead")
        progress.axhline(summary["finish_m"], ls="--", c="0.3", lw=1, label="Ego finish")
        progress.set(xlabel="Simulation time (s)", ylabel="Route progress (m)")
        progress.legend(fontsize="small")

    summary = loaded[0][0]
    route = summary["scenario"].replace("_", " ")
    fig.suptitle(f"{summary['target_m']:g} m gap PID · {route} · seed {seed}")
    fig.savefig(root / f"pid_seed_{seed}.png", dpi=160)
    plt.close(fig)


def write_csv(root, name, records):
    with (root / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Directory with ground_truth and cnn runs.")
    root = parser.parse_args().root
    pairs = []
    for seed in SEEDS:
        truth, cnn = (load_run(root, mode, seed) for mode in MODES)
        pairs.append((seed, truth, cnn, compare_starts(seed, truth[0], cnn[0])))
    runs = [run[1] for _, truth, cnn, _ in pairs for run in (truth, cnn)]
    starts = [start for _, _, _, start in pairs]
    counts = {}
    for mode in MODES:
        mode_runs = [run for run in runs if run["mode"] == mode]
        counts[mode] = {
            key: sum(run[key] for run in mode_runs)
            for key in ("completed", "collision", "distance_violation", "route_violation", "timeout")
        }
        counts[mode]["runs"] = len(mode_runs)
    for seed, truth, cnn, _ in pairs:
        plot_pair(root, seed, (truth, cnn))
    write_csv(root, "pid_summary.csv", runs)
    write_csv(root, "pair_start_checks.csv", starts)
    report = {"counts": counts, "runs": runs, "start_checks": starts}
    (root / "pid_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for mode, values in counts.items():
        print(mode, values)


if __name__ == "__main__":
    main()
