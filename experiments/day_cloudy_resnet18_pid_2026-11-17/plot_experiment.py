"""Render training and paired PID plots and summarize the live telemetry."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent


def csv_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def numbers(rows, key):
    return np.array([float(row[key]) if row[key] else np.nan for row in rows])


epochs = csv_rows(root / "epochs.csv")
fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True, layout="constrained")
for ax, key, label in zip(axes, ("train_huber_loss", "validation_mae_m"),
                          ("Training Huber loss", "Validation MAE (m)")):
    ax.plot(numbers(epochs, "epoch"), numbers(epochs, key), marker="o")
    ax.set(ylabel=label)
    ax.grid(alpha=0.3)
axes[-1].set(xlabel="Epoch", xticks=numbers(epochs, "epoch"))
fig.savefig(root / "training_curves.png", dpi=180)
plt.close(fig)

report = []
for seed in range(42, 47):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True, layout="constrained")
    pair = {}
    for column, mode in enumerate(("ground_truth", "cnn")):
        prefix = root / mode / f"seed_{seed}"
        rows = csv_rows(prefix.with_suffix(".csv"))
        result = json.loads(prefix.with_suffix(".json").read_text())
        pair[mode] = result
        t = numbers(rows, "time_s")
        true = numbers(rows, "true_gap_m")
        predicted = numbers(rows, "predicted_gap_m")
        command = numbers(rows, "pid_command")
        alternate = numbers(rows, "alternate_pid_command")
        valid = np.isfinite(alternate)
        flags = result["reason"].split(", ")
        result.update(
            completed=bool({"destination_reached", "step_limit"} & set(flags)
                           and not {"is_collision", "out_of_lane", "terminal_dist"} & set(flags)),
            collision="is_collision" in flags,
            distance_violation="terminal_dist" in flags,
            lane_violation="out_of_lane" in flags,
            route_destination="destination_reached" in flags,
            prediction_frames=int(valid.sum()),
            mean_signed_prediction_error_m=float((predicted[valid] - true[valid]).mean()) if valid.any() else None,
            mean_abs_prediction_error_m=float(abs(predicted[valid] - true[valid]).mean()) if valid.any() else None,
            actual_throttle_steps=int((command > 0).sum()),
            actual_brake_steps=int((command < 0).sum()),
            opposite_sign_steps=int((command[valid] * alternate[valid] < 0).sum()),
            cnn_throttle_truth_brakes=int(((command > 0) & (alternate < 0)).sum()
                                         if mode == "cnn" else
                                         ((alternate > 0) & (command < 0)).sum()),
            cnn_brakes_truth_throttle=int(((command < 0) & (alternate > 0)).sum()
                                         if mode == "cnn" else
                                         ((alternate < 0) & (command > 0)).sum()),
        )
        gap_ax, command_ax = axes[:, column]
        gap_ax.set_title(f"{mode.replace('_', ' ').title()} · {result['reason']}")
        gap_ax.plot(t, true, label="True gap")
        gap_ax.plot(t, predicted, label="CNN estimate", alpha=0.85)
        gap_ax.scatter(t[-1] + result["dt_s"], float(rows[-1]["post_step_true_gap_m"]),
                       marker="x", color="tab:blue", label="Final true gap")
        gap_ax.axhline(result["target_m"], ls="--", c="black", lw=1, label="Target")
        gap_ax.axhline(result["distance_limit_m"], ls=":", c="tab:red", lw=1, label="Cutoff")
        gap_ax.set_ylabel("Gap (m)")
        gap_ax.legend(fontsize=8)
        command_ax.plot(t, command, label="Applied command")
        command_ax.plot(t, alternate, ls="--", alpha=0.65, label="Other input, same trajectory")
        command_ax.axhline(0, c="black", lw=0.7)
        command_ax.set(xlabel="Time (s)", ylabel="PID command", ylim=(-3.2, 3.2))
        command_ax.legend(fontsize=8)
    fig.suptitle(f"6 m gap PID · seed {seed}")
    fig.savefig(root / f"pid_seed_{seed}.png", dpi=160)
    plt.close(fig)
    left, right = (pair[mode]["initialization"] for mode in ("ground_truth", "cnn"))
    pair["start_check"] = {
        "same_first_pid_step": left.get("first_pid_step") == right.get("first_pid_step"),
        "first_pid_gap_delta_m": abs(left["first_pid_gap_m"] - right["first_pid_gap_m"])
        if "first_pid_gap_m" in left and "first_pid_gap_m" in right else None,
        "same_vehicle_types": left["actor_types"] == right["actor_types"],
        "max_initial_position_delta_m": float(
            np.linalg.norm(np.array(left["state"]).reshape(2, 6)[:, :3] -
                           np.array(right["state"]).reshape(2, 6)[:, :3], axis=1).max()),
        "initial_gap_delta_m": abs(left["initial_gap_m"] - right["initial_gap_m"]),
        "max_initial_velocity_delta_mps": float(
            np.linalg.norm(np.array(left["velocities"]) -
                           np.array(right["velocities"]), axis=1).max()),
    }
    report.append(pair)

(root / "pid_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
columns = ["seed", "mode", "steps", "reason", "completed", "collision", "distance_violation",
           "lane_violation", "route_destination", "min_true_gap_m", "max_true_gap_m",
           "tracking_mae_m", "mean_signed_prediction_error_m", "mean_abs_prediction_error_m",
           "actual_throttle_steps", "actual_brake_steps", "opposite_sign_steps",
           "cnn_throttle_truth_brakes", "cnn_brakes_truth_throttle"]
with (root / "pid_summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for pair in report:
        for mode in ("ground_truth", "cnn"):
            writer.writerow(dict(pair[mode], mode=mode))
