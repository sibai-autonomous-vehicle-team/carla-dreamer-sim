"""Compare the historical 10 m CARLA benchmark with paired 9 m runs."""

import argparse
import csv
import json
from functools import cache
from pathlib import Path

import numpy as np

from analysis.summarize_generalist_live import (
    DAY_CLOUDY,
    EXPERIMENTS,
    MODES,
    read_specialist,
    specialist_directory,
)
from analysis.summarize_specialist_live import (
    CONDITIONS,
    PAIR_FIELDS,
    SEEDS,
    check,
    compare_start,
    mode_totals,
    plt,
    read_rollout,
    row_result,
)
from specialists.publish import checksum


OLD_LIMIT = 10
NEW_LIMIT = 9
SIMULATION_FLAGS = (
    "destination_reached", "out_of_lane", "terminal_dist", "time_exceeded", "is_collision",
)
PLOT_LABELS = {
    "live_10m": "10 m live",
    "replay_9m": "9 m replay",
    "live_9m": "9 m live",
}
DELTA_FIELDS = (
    "finishes", "collisions", "distance_violations", "route_violations", "timeouts",
    "empty_camera", "steps", "min_true_gap_m", "max_true_gap_m", "mean_run_tracking_mae_m",
)


def read_baseline(path, seed, route, mode, condition):
    if mode == "specialist":
        return read_specialist(path, seed, route, condition)
    return read_rollout(path, seed, route, mode, condition)


def first_crossing(gaps, limit):
    # The simulator checks the gap after the action, including during warmup.
    crossings = np.flatnonzero(gaps[1:] > limit)
    return int(crossings[0]) + 1 if len(crossings) else None


def validate_distance_limit(rollout, limit, path):
    summary, _, gaps, _, _ = rollout
    crossing = first_crossing(gaps, limit)
    flags = summary["reason"].split(", ")
    check(summary["distance_limit_m"] == limit and
          (crossing == summary["steps"] if "terminal_dist" in flags else crossing is None),
          f"Unexpected cutoff or inconsistent cutoff decision in {path}")


def replay_cutoff(path, rollout):
    summary, times, gaps, predictions, commands = rollout
    crossing = first_crossing(gaps, NEW_LIMIT)
    replay = summary | {"distance_limit_m": NEW_LIMIT}
    if crossing is None:
        return replay, times, gaps, predictions, commands

    # A step-limit exit is replaced by a simulator flag on that same step.
    # Other simulator flags on that step remain in their original order.
    terminal_flags = {"terminal_dist"}
    final_route = summary["final_route"]
    if crossing == summary["steps"]:
        terminal_flags.update(summary["reason"].split(", "))
    else:
        with path.with_suffix(".csv").open(newline="") as stream:
            row = next(row for row in csv.DictReader(stream) if int(row["step"]) == crossing)
        final_route = {field: float(row[field]) for field in final_route}

    truncated_gaps = gaps[:crossing + 1]
    replay.update(
        steps=crossing,
        reason=", ".join(flag for flag in SIMULATION_FLAGS if flag in terminal_flags),
        min_true_gap_m=float(truncated_gaps.min()),
        max_true_gap_m=float(truncated_gaps.max()),
        tracking_mae_m=float(np.abs(truncated_gaps[1:] - summary["target_m"]).mean()),
        final_route=final_route,
    )
    return replay, times[:crossing], truncated_gaps, predictions[:crossing], commands[:crossing]


@cache
def checkpoint_digest(path):
    return checksum(path)


def validate_pair(baseline, current, path):
    old, new = baseline[0], current[0]
    fields = [field for field in PAIR_FIELDS if field != "distance_limit_m"]
    differ = [field for field in fields if old[field] != new[field]]
    check(not differ, f"Settings differ in {path}: {', '.join(differ)}")
    check(old["route_centerline_xy"] == new["route_centerline_xy"],
          f"Route centerline differs in {path}")
    old_checkpoint = Path(old["checkpoint"]).resolve()
    new_checkpoint = Path(new["checkpoint"]).resolve()
    check(old_checkpoint == new_checkpoint or
          checkpoint_digest(old_checkpoint) == checkpoint_digest(new_checkpoint),
          f"Checkpoint differs in {path}")
    start = compare_start(new["initialization"], old["initialization"])
    check(start["same_vehicle_types_as_gt"], f"Vehicle types differ in {path}")
    return {field.replace("_as_gt", "_as_10m").replace("_vs_gt", "_vs_10m"): value
            for field, value in start.items()}


def compare_with_replay(replay, current):
    previous, _, expected_gaps, expected_predictions, expected_commands = replay
    actual, _, gaps, predictions, commands = current
    steps = min(previous["steps"], actual["steps"])
    return {
        "matched_termination": (actual["steps"], actual["reason"])
        == (previous["steps"], previous["reason"]),
        "compared_steps": steps,
        "max_true_gap_delta_m": float(np.max(np.abs(gaps[:steps + 1] - expected_gaps[:steps + 1]))),
        "max_pid_command_delta": float(np.max(np.abs(commands[:steps] - expected_commands[:steps]))),
        "same_predictions": bool(np.array_equal(
            predictions[:steps], expected_predictions[:steps], equal_nan=True,
        )),
    }


def run_record(rollout, mode, path):
    summary, _, gaps, predictions, commands = rollout
    errors = (predictions - gaps[:-1])[np.isfinite(predictions)]
    return row_result(summary, errors, commands, mode) | {"source": str(path)}, errors


def summarize_group(group):
    totals = {}
    for stage, entries in group.items():
        records, errors = zip(*entries)
        totals[stage] = mode_totals(records, np.concatenate(errors))
    for stage, delta in (("replay_9m", "cutoff_only_delta"), ("live_9m", "live_delta")):
        if stage in totals:
            totals[delta] = {
                field: totals[stage][field] - totals["live_10m"][field]
                for field in DELTA_FIELDS
            }
    return totals


def write_csv(output, condition_totals, overall):
    rows = []
    for condition, modes in (condition_totals | {"all": overall}).items():
        for mode, totals in modes.items():
            row = {"condition": condition, "mode": mode}
            for stage, stats in totals.items():
                row.update({f"{stage}_{field}": value for field, value in stats.items()})
            rows.append(row)
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(output, condition_totals, stages):
    figure, axes = plt.subplots(len(MODES), 1, figsize=(12, 6), sharex=True,
                                layout="constrained")
    positions = np.arange(len(condition_totals))
    width = 0.75 / len(stages)
    for mode, axis in zip(MODES, axes):
        for index, stage in enumerate(stages):
            offset = (index - (len(stages) - 1) / 2) * width
            finishes = [modes[mode][stage]["finishes"] for modes in condition_totals.values()]
            bars = axis.bar(positions + offset, finishes, width, label=PLOT_LABELS[stage])
            axis.bar_label(bars, fontsize=8)
        axis.set(ylabel="Routes finished", title=mode.capitalize(), ylim=(0, 6), yticks=range(6))
    axes[0].legend(ncols=len(stages), loc="upper right")
    axes[-1].set_xticks(positions, [condition.replace("_", "\n").title()
                                   for condition in condition_totals])
    figure.suptitle("Live CARLA route completion · 6 m PID target")
    figure.savefig(output / "comparison.png", dpi=145)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--routes", type=Path, default=Path("specialists/routes.json"))
    parser.add_argument("--baseline-generalist", type=Path,
                        default=EXPERIMENTS / "generalist_live_2026-11-18")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Report the 10 m live results and replay without the 9 m runs")
    args = parser.parse_args()
    routes = json.loads(args.routes.read_text())["test"]
    check(len(routes) == len(SEEDS) and len({route["name"] for route in routes}) == len(SEEDS),
          "Expected five distinct test routes in seed order")
    stages = ("live_10m", "replay_9m") if args.baseline_only else tuple(PLOT_LABELS)

    runs = []
    all_entries = {mode: {stage: [] for stage in stages} for mode in MODES}
    condition_totals = {}
    for condition in CONDITIONS:
        reference_dirs = {
            "generalist": args.baseline_generalist / condition,
            "specialist": specialist_directory(condition),
        }
        condition_totals[condition] = {}
        for mode, directory in reference_dirs.items():
            group = {stage: [] for stage in stages}
            for seed, route in zip(SEEDS, routes):
                old_path = directory / f"seed_{seed}"
                baseline = read_baseline(old_path, seed, route, mode, condition)
                validate_distance_limit(baseline, OLD_LIMIT, old_path)
                check(baseline[0]["target_m"] == 6.0, f"Unexpected PID target in {old_path}")
                replay = replay_cutoff(old_path, baseline)
                group["live_10m"].append(run_record(baseline, mode, old_path))
                group["replay_9m"].append(run_record(replay, mode, old_path))
                pair = {"settings_10m": {field: baseline[0][field] for field in PAIR_FIELDS}}
                if not args.baseline_only:
                    new_path = args.experiment / "live" / mode / condition / f"seed_{seed}"
                    current = read_rollout(new_path, seed, route, mode, condition)
                    validate_distance_limit(current, NEW_LIMIT, new_path)
                    pair.update(
                        settings_9m={field: current[0][field] for field in PAIR_FIELDS},
                        start=validate_pair(baseline, current, new_path),
                        live_vs_replay=compare_with_replay(replay, current),
                    )
                    group["live_9m"].append(run_record(current, mode, new_path))
                pair.update({stage: group[stage][-1][0] for stage in stages})
                pair["first_9m_crossing_step_in_10m_live"] = first_crossing(baseline[2], NEW_LIMIT)
                runs.append(pair)
            condition_totals[condition][mode] = summarize_group(group)
            for stage in stages:
                all_entries[mode][stage].extend(group[stage])

    overall = {mode: summarize_group(group) for mode, group in all_entries.items()}
    summary = {
        "old_limit_m": OLD_LIMIT,
        "new_limit_m": NEW_LIMIT,
        "termination_rule": "Post-step true gap strictly greater than the cutoff",
        "target_m": 6.0,
        "routes": str(args.routes),
        "baseline_generalist": str(args.baseline_generalist),
        "day_cloudy_legacy_settings_source": str(DAY_CLOUDY / "commands.sh"),
        "replay_method": "End the archived rollout at its first post-step gap above 9 m.",
        "by_mode": overall,
        "conditions": condition_totals,
        "runs": runs,
    }
    args.experiment.mkdir(parents=True, exist_ok=True)
    write_csv(args.experiment, condition_totals, overall)
    plot_comparison(args.experiment, condition_totals, stages)
    (args.experiment / "summary.json").write_text(json.dumps(
        summary, indent=2, allow_nan=False,
    ) + "\n")
    for mode, totals in overall.items():
        results = ", ".join(f"{PLOT_LABELS[stage]} {totals[stage]['finishes']}/{totals[stage]['episodes']}"
                            for stage in stages)
        print(f"{mode.capitalize()} finishes: {results}")


if __name__ == "__main__":
    main()
