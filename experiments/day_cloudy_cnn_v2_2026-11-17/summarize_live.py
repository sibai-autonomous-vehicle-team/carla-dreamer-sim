"""Summarize and plot the held-out manifest's three live PID modes."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


root = Path(__file__).resolve().parent
routes = json.loads((root / "routes.json").read_text())["test"]
modes = ("ground_truth", "cnn_v1", "cnn_v2")
names = ("Ground truth", "CNN v1", "CNN v2")
results = []
for seed, route in enumerate(routes, start=42):
    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 7), layout="constrained")
    starts = []
    for mode, label in zip(modes, names):
        run = root / "live" / mode / f"seed_{seed}"
        summary = json.loads(run.with_suffix(".json").read_text())
        with run.with_suffix(".csv").open(newline="") as handle:
            trace = list(csv.DictReader(handle))
        assert summary["scenario"] == route["name"]
        assert summary["steps"] == len(trace) and summary["reason"] == trace[-1]["terminal_reason"]
        starts.append(summary["initialization"])
        flags = set(summary["reason"].split(", "))
        finish = summary["final_route"]["ego_progress_m"]
        reached = (flags == {"destination_reached"} and finish >= summary["finish_m"]
                   and summary["final_route"]["ego_route_error_m"] <= 1)
        results.append(dict(
            seed=seed, scenario=summary["scenario"], mode=mode, finished=reached,
            reason=summary["reason"], steps=summary["steps"],
            ego_progress_m=finish, goal_progress_m=summary["finish_m"],
            min_true_gap_m=summary["min_true_gap_m"], max_true_gap_m=summary["max_true_gap_m"],
            tracking_mae_m=summary["tracking_mae_m"],
            collision="is_collision" in flags, distance_violation="terminal_dist" in flags,
            route_violation="out_of_lane" in flags,
            timeout=bool({"time_exceeded", "step_limit"} & flags),
        ))
        time = [float(row["time_s"]) for row in trace]
        decision_gap = [float(row["true_gap_m"]) for row in trace]
        color = axes[0].plot(
            time + [time[-1] + summary["dt_s"]],
            decision_gap + [float(trace[-1]["post_step_true_gap_m"])], label=label,
        )[0].get_color()
        if mode != "ground_truth":
            axes[0].plot(time, [float(row["predicted_gap_m"]) if row["predicted_gap_m"] else np.nan
                                for row in trace], "--", color=color, alpha=0.6)
        axes[1].plot(time, [float(row["pid_command"]) for row in trace], color=color)
        axes[2].plot([t + summary["dt_s"] for t in time],
                     [float(row["ego_progress_m"]) for row in trace], color=color)
    for result, start in zip(results[-3:], starts):
        truth = starts[0]
        result["initial_gap_delta_vs_gt_m"] = abs(start["initial_gap_m"] - truth["initial_gap_m"])
        xyz = np.array(start["state"]).reshape(2, 6)[:, :3]
        ref = np.array(truth["state"]).reshape(2, 6)[:, :3]
        result["max_initial_position_delta_vs_gt_m"] = float(np.linalg.norm(xyz - ref, axis=1).max())
        result["same_vehicle_types_as_gt"] = start["actor_types"] == truth["actor_types"]
    axes[0].axhline(10, color="gray", linestyle=":", linewidth=1)
    axes[0].axhline(6, color="gray", linestyle=":", linewidth=1)
    axes[0].set_ylabel("Gap (m)")
    axes[0].legend()
    axes[1].set(ylabel="PID command", ylim=(-3.2, 3.2))
    axes[2].axhline(route["finish_m"], color="gray", linestyle=":", linewidth=1)
    axes[2].set(xlabel="Simulation time (s)", ylabel="Ego progress (m)")
    figure.suptitle(f"{route['name']} · seed {seed} · dashed lines are CNN estimates")
    figure.savefig(root / "live" / f"seed_{seed}.png", dpi=140)
    plt.close(figure)

with (root / "live" / "summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(results[0]))
    writer.writeheader()
    writer.writerows(results)
for mode in modes:
    records = [row for row in results if row["mode"] == mode]
    print(mode, sum(row["finished"] for row in records), "/", len(records), "finished")
