"""Validate and summarize the matched normal and tight following rollouts."""
import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


root = Path(__file__).resolve().parent
data = []
modes = ("ground_truth", "v2", "v3")
labels = ("Ground truth", "CNN v2", "CNN v3")
for condition, target in (("normal", 6), ("tight", 2.5)):
    for seed in range(42, 47):
        figure, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, layout="constrained")
        baseline = None
        for mode, label in zip(modes, labels):
            run = root / condition / mode / f"seed_{seed}"
            info = json.loads(run.with_suffix(".json").read_text())
            with run.with_suffix(".csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            init = info["initialization"]
            if baseline is None:
                baseline = init
            assert len(rows) == info["steps"] and rows[-1]["terminal_reason"] == info["reason"]
            assert info["target_m"] == target and info["distance_limit_m"] == 10
            assert init["actor_types"] == baseline["actor_types"]
            start = np.asarray(init["state"]).reshape(2, 6)[:, :3]
            reference = np.asarray(baseline["state"]).reshape(2, 6)[:, :3]
            flags = set(info["reason"].split(", "))
            progress = info["final_route"]["ego_progress_m"]
            post_gaps = [float(row["post_step_true_gap_m"]) for row in rows]
            pairs = [(float(row["true_gap_m"]), float(row["predicted_gap_m"]))
                     for row in rows if row["predicted_gap_m"]]
            close = [(g, p) for g, p in pairs if g < 3]
            data.append(dict(
                condition=condition, seed=seed, scenario=info["scenario"], mode=mode,
                finished=flags == {"destination_reached"} and progress >= info["finish_m"],
                steps=info["steps"], reason=info["reason"],
                collision="is_collision" in flags, cutoff="terminal_dist" in flags,
                route_violation="out_of_lane" in flags,
                min_true_gap_m=info["min_true_gap_m"], max_true_gap_m=info["max_true_gap_m"],
                tracking_mae_m=info["tracking_mae_m"],
                steps_below_1m=sum(g < 1 for g in post_gaps),
                steps_below_3m=sum(g < 3 for g in post_gaps),
                close_prediction_frames=len(close),
                close_prediction_mae_m=mean(abs(p-g) for g, p in close) if close else None,
                predicted_at_least_6m_when_close=sum(p >= 6 for _, p in close),
                negative_predictions=sum(p < 0 for _, p in pairs),
                max_position_delta_to_truth_m=float(np.linalg.norm(start-reference, axis=1).max()),
                initial_gap_delta_to_truth_m=abs(init["initial_gap_m"] - baseline["initial_gap_m"]),
            ))
            time = [float(row["time_s"]) for row in rows]
            true_gaps = [float(row["true_gap_m"]) for row in rows]
            color = axes[0].plot(time + [time[-1] + info["dt_s"]],
                                 true_gaps + [post_gaps[-1]], label=label)[0].get_color()
            if mode != "ground_truth":
                prediction = [float(row["predicted_gap_m"]) if row["predicted_gap_m"] else np.nan
                              for row in rows]
                axes[0].plot(time, prediction, "--", color=color, alpha=0.6)
            axes[1].plot(time, [float(row["pid_command"]) for row in rows], color=color)
        axes[0].axhline(target, color="gray", linestyle=":", linewidth=1)
        axes[0].axhline(10, color="gray", linestyle=":", linewidth=1)
        axes[0].set_ylabel("Gap (m)")
        axes[0].legend()
        axes[1].set(xlabel="Simulation time (s)", ylabel="PID command", ylim=(-3.2, 3.2))
        figure.suptitle(f"{info['scenario']} · {target:g} m target · dashed lines show estimates")
        figure.savefig(root / condition / f"seed_{seed}.png", dpi=135)
        plt.close(figure)

with (root / "live_summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(data[0]))
    writer.writeheader()
    writer.writerows(data)
report = {}
for condition in ("normal", "tight"):
    report[condition] = {}
    for mode in modes:
        records = [item for item in data if item["mode"] == mode and item["condition"] == condition]
        close_frames = sum(item["close_prediction_frames"] for item in records)
        report[condition][mode] = dict(
            runs=len(records), finishes=sum(item["finished"] for item in records),
            collisions=sum(item["collision"] for item in records),
            cutoffs=sum(item["cutoff"] for item in records),
            route_violations=sum(item["route_violation"] for item in records),
            min_true_gap_m=min(item["min_true_gap_m"] for item in records),
            max_true_gap_m=max(item["max_true_gap_m"] for item in records),
            mean_per_run_tracking_mae_m=mean(item["tracking_mae_m"] for item in records),
            close_prediction_frames=close_frames,
            close_decision_mae_m=sum((item["close_prediction_mae_m"] or 0) *
                                     item["close_prediction_frames"] for item in records) /
                                 close_frames if close_frames else None,
            negative_predictions=sum(item["negative_predictions"] for item in records),
        )
(root / "live_summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
