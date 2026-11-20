"""Compare one generalist with condition-matched specialists in live CARLA."""

import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

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


EXPERIMENTS = Path("experiments")
DAY_CLOUDY = EXPERIMENTS / "day_cloudy_cnn_close_v3_2026-11-17"
SUNSET_CLOUDY = EXPERIMENTS / "sunset_cloudy_specialist_2026-11-17"
ALL_SPECIALISTS = EXPERIMENTS / "all_specialists_2026-11-17"
MODES = ("generalist", "specialist")


def specialist_directory(condition):
    if condition == "day_cloudy":
        return DAY_CLOUDY / "normal" / "v3"
    if condition == "sunset_cloudy":
        return SUNSET_CLOUDY / "live" / "specialist"
    return ALL_SPECIALISTS / condition / "live" / "specialist"


def read_specialist(path, seed, route, condition):
    if condition != "day_cloudy":
        return read_rollout(path, seed, route, "specialist", condition)

    # The original cloudy-day runner omitted these fields. Use its recorded
    # normal_v3 command to normalize an ephemeral copy for the shared validator.
    metadata = json.loads(path.with_suffix(".json").read_text())
    defaults = {
        "condition": "day_cloudy",
        "checkpoint": str(DAY_CLOUDY / "finetune_v3" / "best.pth"),
        "warmup_steps": 5,
        "max_steps": 360,
    }
    if defaults.keys() <= metadata.keys():
        return read_rollout(path, seed, route, "specialist", condition)
    with TemporaryDirectory(prefix="carla-day-cloudy-") as directory:
        normalized = Path(directory) / path.name
        normalized.with_suffix(".json").write_text(json.dumps(defaults | metadata))
        normalized.with_suffix(".csv").symlink_to(path.with_suffix(".csv").resolve())
        return read_rollout(normalized, seed, route, "specialist", condition)


def write_comparison(output, condition_totals):
    rows = []
    for condition, totals in condition_totals.items():
        row = {"condition": condition}
        for mode, stats in totals.items():
            row.update({f"{mode}_{key}": value for key, value in stats.items()})
        rows.append(row)
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(output, condition_totals):
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, layout="constrained")
    positions = np.arange(len(condition_totals))
    width = 0.36
    for offset, mode in zip((-width / 2, width / 2), MODES):
        stats = [totals[mode] for totals in condition_totals.values()]
        for ax, field in zip(axes, ("finishes", "mean_run_tracking_mae_m")):
            bars = ax.bar(positions + offset, [row[field] for row in stats], width,
                          label=mode.capitalize())
            ax.bar_label(bars, fmt="%d" if field == "finishes" else "%.2f", fontsize=8)
    axes[0].set(ylabel="Routes finished", ylim=(0, 6.2), yticks=range(6))
    axes[0].legend()
    axes[1].set(ylabel="Mean run tracking MAE (m)")
    axes[1].margins(y=0.16)
    axes[1].set_xticks(positions, [condition.replace("_", "\n").title()
                                 for condition in condition_totals])
    figure.suptitle("Live CARLA · 6 m target · five routes per condition")
    figure.savefig(output / "comparison.png", dpi=145)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--routes", type=Path, default=Path("specialists/routes.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("generalist/cnn.pth"))
    args = parser.parse_args()
    routes = json.loads(args.routes.read_text())["test"]
    check(len(routes) == len(SEEDS) and len({route["name"] for route in routes}) == len(SEEDS),
          "Expected five distinct test routes in seed order")

    records, condition_totals, settings = [], {}, {}
    errors_by_mode = {mode: [] for mode in MODES}
    for condition in CONDITIONS:
        reference_dir = specialist_directory(condition)
        profile_path = Path("specialists") / condition / "pid.json"
        profile = json.loads(profile_path.read_text())
        expected = {
            "condition": condition, "target_m": profile["target_gap"],
            "gains": [profile[name] for name in ("kp", "ki", "kd")],
            "distance_limit_m": profile["terminal_dist"], "lane_limit_m": profile["out_lane_thres"],
            "warmup_steps": profile["warmup_steps"], "max_steps": profile["steps"],
        }
        profile_routes = profile_path.parent / profile["route_manifest"]
        check(profile["condition"] == condition and profile["episodes"] == len(routes)
              and profile["seed"] == SEEDS.start and profile["shared_route"]
              and json.loads(profile_routes.read_text())["test"] == routes,
              f"Unexpected route or condition in {profile_path}")
        condition_rows = {mode: [] for mode in MODES}
        condition_errors = {mode: [] for mode in MODES}
        for seed, route in zip(SEEDS, routes):
            live = args.experiment / condition / f"seed_{seed}"
            baseline = reference_dir / f"seed_{seed}"
            current = read_rollout(live, seed, route, "generalist", condition)
            reference = read_specialist(baseline, seed, route, condition)
            actual, native = current[0], reference[0]
            current_settings = {key: actual[key] for key in PAIR_FIELDS}
            check(all(current_settings[key] == value for key, value in expected.items())
                  and current_settings == {key: native[key] for key in PAIR_FIELDS}
                  and current_settings == settings.setdefault(condition, current_settings),
                  f"Unmatched weather, gains, timestep, or termination settings in {live}")
            check(actual["route_centerline_xy"] == native["route_centerline_xy"],
                  f"Unmatched route centerline in {live}")
            check(Path(actual["checkpoint"]).resolve() == args.checkpoint.resolve(),
                  f"Unexpected generalist checkpoint in {live}")
            start = {key.replace("_gt", "_specialist"): value for key, value in
                     compare_start(actual["initialization"], native["initialization"]).items()}
            check(start["same_vehicle_types_as_specialist"], f"Unmatched vehicle types in {live}")
            for mode, rollout, path in zip(MODES, (current, reference), (live, baseline)):
                metadata, _, gaps, predictions, commands = rollout
                errors = (predictions - gaps[:-1])[np.isfinite(predictions)]
                record = row_result(metadata, errors, commands, mode) | {"source": str(path)}
                if mode == "generalist":
                    record.update(start)
                condition_rows[mode].append(record)
                condition_errors[mode].append(errors)
                errors_by_mode[mode].append(errors)
                records.append(record)
        condition_totals[condition] = {
            mode: mode_totals(condition_rows[mode], np.concatenate(condition_errors[mode]))
            for mode in MODES
        }
    overall = {
        mode: mode_totals([row for row in records if row["mode"] == mode],
                          np.concatenate(errors_by_mode[mode]))
        for mode in MODES
    }
    summary = {
        "checkpoint": str(args.checkpoint), "routes": str(args.routes),
        "settings": settings, "by_mode": overall, "conditions": condition_totals,
        "day_cloudy_legacy_settings_source": str(DAY_CLOUDY / "commands.sh"),
        "runs": records,
    }
    write_comparison(args.experiment, condition_totals)
    plot_comparison(args.experiment, condition_totals)
    (args.experiment / "summary.json").write_text(json.dumps(
        summary, indent=2, allow_nan=False,
    ) + "\n")
    for mode, stats in overall.items():
        print(f"{mode.capitalize()} finished {stats['finishes']}/{stats['episodes']} routes")


if __name__ == "__main__":
    main()

