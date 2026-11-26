"""Audit the nine braking-data CNN retrains offline and in paired CARLA runs."""

import argparse
from collections import Counter
import csv
from functools import cache
import json
import math
from pathlib import Path

import numpy as np

from analysis.summarize_specialist_live import (
    CONDITIONS,
    PAIR_FIELDS,
    SEEDS,
    check,
    mode_totals,
    plt,
    read_rollout,
    row_result,
)
from selector.reporting import START_TOLERANCES, state_deltas
from specialists.publish import checksum


MODES = {"before": "Existing specialist", "after": "Retrained with braking data"}
SOURCE_COUNTS = {"original": (640, 80, 80), "braking": (400, 50, 50)}
SPLIT_FIELDS = ("train_ids", "val_ids", "test_ids")
GAP_BINS = (
    ("[0, 1) m", 0, 1), ("[1, 2) m", 1, 2), ("[2, 3) m", 2, 3),
    ("[3, 6) m", 3, 6), ("[6, 8) m", 6, 8), ("[8, 10] m", 8, 10),
    (">10 m", 10, math.inf),
)


def read_json(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@cache
def checkpoint_hash(path):
    return checksum(path)


def torch_load(path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


def validate_partition(data, experiment, condition, settings):
    merged = read_json(data / "split.json")
    sources = {source: read_json(data / f"split_{source}.json") for source in SOURCE_COUNTS}
    for source, split in sources.items():
        counts = SOURCE_COUNTS[source]
        check(tuple(len(split[key]) for key in SPLIT_FIELDS) == counts,
              f"Unexpected {source} split sizes for {condition}")
        ids = [index for key in SPLIT_FIELDS for index in split[key]]
        expected = range(0, 800) if source == "original" else range(800, 1300)
        check(sorted(ids) == list(expected), f"Duplicate or missing {source} IDs for {condition}")
    old = read_json(Path("experiments/generalist_all_conditions_2026-11-18/training/splits")
                    / f"{condition}.json")
    check(all(old[key] == sources["original"][key] for key in SPLIT_FIELDS),
          f"Existing holdout was changed for {condition}")
    for key in SPLIT_FIELDS:
        target = sources["original"][key] + sources["braking"][key]
        check(sorted(merged[key]) == sorted(target), f"Merged split differs for {condition}: {key}")

    train = experiment / condition / "train"
    with (train / "epochs.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    epochs = [int(row["epoch"]) for row in rows]
    check(epochs == list(range(1, settings["epochs"] + 1)),
          f"Missing or nonconsecutive epochs for {condition}")
    check(all(math.isfinite(float(row[field])) for row in rows
              for field in ("train_huber_loss", "validation_mae_m")),
          f"Nonfinite epoch metrics for {condition}")
    best = min(rows, key=lambda row: float(row["validation_mae_m"]))
    checkpoint = torch_load(train / "best.pth")
    check(int(best["epoch"]) == checkpoint["epoch"] and
          np.isclose(float(best["validation_mae_m"]), checkpoint["validation_mae"]),
          f"Checkpoint disagrees with the best validation epoch for {condition}")
    for recorded in (checkpoint, read_json(train / "split.json")):
        check(all(recorded[key] == merged[key] for key in (*SPLIT_FIELDS, "seed")),
              f"Trained checkpoint or recorded training split disagrees for {condition}")
    return sources, {
        "training_episodes": len(merged["train_ids"]),
        "validation_episodes": len(merged["val_ids"]),
        "test_episodes": len(merged["test_ids"]),
        "epochs": len(rows), "best_epoch": checkpoint["epoch"],
        "best_validation_mae_m": checkpoint["validation_mae"],
        "checkpoint": str(train / "best.pth"),
        "checkpoint_sha256": checkpoint_hash((train / "best.pth").resolve()),
    }


def error_metrics(errors):
    if not len(errors):
        return {
            "frames": 0, "mae_m": None, "rmse_m": None, "signed_error_m": None,
            "p95_absolute_error_m": None, "max_absolute_error_m": None,
        }
    absolute = np.abs(errors)
    return {
        "frames": len(errors),
        "mae_m": float(absolute.mean()),
        "rmse_m": float(np.sqrt(np.square(errors).mean())),
        "signed_error_m": float(errors.mean()),
        "p95_absolute_error_m": float(np.quantile(absolute, 0.95)),
        "max_absolute_error_m": float(absolute.max()),
    }


def offline_result(path, data, source, expected_ids, expected_targets):
    result = read_json(path / "test_metrics.json")
    values = torch_load(path / "test_predictions.pth")
    checkpoint_path = Path(result["checkpoint_path"]).resolve()
    check(Path(result["dataset_path"]).resolve() == data.resolve() and
          Path(result["split_path"]).resolve() == (data / f"split_{source}.json").resolve() and
          result["split_source"] == "external", f"Unexpected data or split provenance in {path}")
    check(all(result[key] == values[key] for key in
              ("checkpoint_path", "dataset_path", "split_source", "split_path")),
          f"Metrics and predictions have different sources in {path}")
    check(result["test_episodes"] == len(expected_ids) and values["test_ids"] == expected_ids,
          f"Test ID list disagrees with the reserved split in {path}")
    predicted = values["prediction"].numpy().astype(np.float64)
    target = values["target"].numpy().astype(np.float64)
    check(predicted.ndim == target.ndim == 1 and predicted.shape == target.shape and
          len(target) == result["test_frames"] and np.isfinite(predicted).all() and
          np.isfinite(target).all() and (target >= 0).all(), f"Invalid held-out predictions in {path}")
    check(np.array_equal(target, expected_targets), f"Test labels differ from aligned source gaps in {path}")
    errors = predicted - target
    metrics = error_metrics(errors)
    for key, value in metrics.items():
        if key != "frames":
            check(np.isclose(value, result[key], rtol=2e-5, atol=1e-6),
                  f"Recomputed {key} differs from saved metric in {path}")
    return {
        "episodes": len(expected_ids), **metrics, "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash(checkpoint_path),
        "checkpoint_epoch": result["checkpoint_epoch"], "source_metrics": str(path / "test_metrics.json"),
    }, target, errors


def bin_rows(condition, source, mode, target, errors):
    for label, lower, upper in GAP_BINS:
        if lower == 10:
            mask = target > lower
        elif upper == 10:
            mask = (target >= lower) & (target <= upper)
        else:
            mask = (target >= lower) & (target < upper)
        yield {"condition": condition, "source": source, "mode": mode, "true_gap_bin": label,
               **error_metrics(errors[mask])}


def load_offline(experiment, conditions, settings):
    offline, summary, bins, training = {}, [], [], {}
    for condition in conditions:
        data = experiment / "data" / condition
        splits, training[condition] = validate_partition(data, experiment, condition, settings)
        baseline = experiment / condition / "before"
        deployed = read_json(baseline / "pid.json")["cnn_sha256"]
        check(checkpoint_hash((baseline / "cnn.pth").resolve()) == deployed,
              f"Frozen baseline differs from its recorded fingerprint for {condition}")
        expected_hashes = {
            "before": deployed, "after": training[condition]["checkpoint_sha256"],
        }
        distances = torch_load(data / "distances.pth")
        for source in SOURCE_COUNTS:
            ids = splits[source]["test_ids"]
            expected_targets = np.concatenate([distances[index][:-1].numpy() for index in ids])
            for mode in MODES:
                path = experiment / condition / "offline" / mode / source
                record, target, errors = offline_result(path, data, source, ids, expected_targets)
                check(record["checkpoint_sha256"] == expected_hashes[mode],
                      f"Unexpected {mode} checkpoint evaluated for {condition} {source}")
                offline[condition, source, mode] = record, target, errors
                summary.append({"condition": condition, "source": source, "mode": mode, **record})
                bins.extend(bin_rows(condition, source, mode, target, errors))
    for source in SOURCE_COUNTS:
        for mode in MODES:
            entries = [offline[condition, source, mode] for condition in conditions]
            target = np.concatenate([entry[1] for entry in entries])
            errors = np.concatenate([entry[2] for entry in entries])
            summary.append({
                "condition": "all", "source": source, "mode": mode,
                "episodes": sum(entry[0]["episodes"] for entry in entries), **error_metrics(errors),
                "checkpoint": None, "checkpoint_sha256": None, "checkpoint_epoch": None,
                "source_metrics": None,
            })
            bins.extend(bin_rows("all", source, mode, target, errors))
    return offline, summary, bins, training


def live_start(before, after):
    start = state_deltas([before, after])
    first_before, first_after = before.get("first_pid_gap_m"), after.get("first_pid_gap_m")
    first_gap_delta = None if first_before is None or first_after is None else abs(first_after - first_before)
    step = before.get("first_pid_step")
    same_control_step = step is not None and step == after.get("first_pid_step")
    matched = (start["same_actor_types"] and same_control_step and first_gap_delta is not None
               and first_gap_delta <= START_TOLERANCES["gap_m"]
               and all(start[field] <= tolerance for field, tolerance in START_TOLERANCES.items()))
    return {"matched_start": matched, **{f"initial_{k}": v for k, v in start.items()},
            "same_first_pid_step": same_control_step, "first_pid_gap_delta_m": first_gap_delta}


def load_live(experiment, conditions, routes, offline):
    rows, paired = [], []
    settings = None
    for condition in conditions:
        for seed, route in zip(SEEDS, routes):
            arms = {}
            for mode in MODES:
                path = experiment / condition / "live" / mode / f"seed_{seed}"
                result, _, gaps, predictions, commands = read_rollout(path, seed, route, mode, condition)
                current_settings = {key: result[key] for key in PAIR_FIELDS if key != "condition"}
                if settings is None:
                    settings = current_settings
                check(current_settings == settings, f"Control settings disagree in {path}")
                check(result["target_m"] == 6 and result["distance_limit_m"] == 10,
                      f"Incorrect target or cutoff in {path}")
                crossings = np.flatnonzero(gaps[1:] > result["distance_limit_m"])
                flags = set(result["reason"].split(", "))
                check(list(crossings) == ([result["steps"] - 1] if "terminal_dist" in flags else []),
                      f"Unexpected distance cutoff decision in {path}")
                expected = offline[condition, "original", mode][0]["checkpoint_sha256"]
                check(checkpoint_hash(Path(result["checkpoint"]).resolve()) == expected,
                      f"Live checkpoint differs from offline evaluation in {path}")
                errors = (predictions - gaps[:-1])[np.isfinite(predictions)]
                record = row_result(result, errors, commands, mode)
                record.update(specification_violation=bool(flags & {
                    "is_collision", "terminal_dist", "out_of_lane",
                }), trace=str(path.with_suffix(".csv")))
                arms[mode] = result, record, errors
            before, after = arms["before"][0], arms["after"][0]
            check(before["route_centerline_xy"] == after["route_centerline_xy"],
                  f"Route geometry differs for {condition} seed {seed}")
            start = live_start(before["initialization"], after["initialization"])
            paired.append({"condition": condition, "seed": seed, "scenario": route["name"], **start})
            for _, record, errors in arms.values():
                rows.append((record | start, errors))
    return rows, paired, settings


def live_totals(live, conditions):
    output = []
    for condition in (*conditions, "all"):
        for scope in ("matched_pairs", "all_attempts"):
            for mode in MODES:
                entries = [(record, errors) for record, errors in live
                           if record["mode"] == mode and (condition == "all" or record["condition"] == condition)
                           and (scope == "all_attempts" or record["matched_start"])]
                heading = {"condition": condition, "scope": scope, "mode": mode}
                if not entries:
                    output.append(heading | {"episodes": 0})
                    continue
                records, errors = zip(*entries)
                output.append(heading | mode_totals(records, np.concatenate(errors)) | {
                    "specification_violations": sum(record["specification_violation"] for record in records),
                    "terminal_reasons": dict(Counter(record["reason"] for record in records)),
                })
    return output


def plot_comparison(output, conditions, offline, live_summary):
    rows = 3 if live_summary else 2
    figure, axes = plt.subplots(rows, 1, sharex=True, figsize=(11.5, 2.7 * rows),
                                layout="constrained")
    positions = np.arange(len(conditions))
    width = 0.36
    colors = {"before": "#28517A", "after": "#007F78"}
    for ax, source in zip(axes, SOURCE_COUNTS):
        for offset, mode in zip((-width / 2, width / 2), MODES):
            ax.bar(positions + offset, [offline[condition, source, mode][0]["mae_m"]
                                        for condition in conditions], width,
                   color=colors[mode], label=MODES[mode])
        ax.set(ylabel="Gap MAE (m)",
               title=f"{source.capitalize()} holdout  ·  {SOURCE_COUNTS[source][-1]} episodes per condition")
    axes[0].legend(ncols=2, frameon=False)
    if live_summary:
        ax = axes[-1]
        matched = {(row["condition"], row["mode"]): row for row in live_summary
                   if row["scope"] == "matched_pairs"}
        for offset, mode in zip((-width / 2, width / 2), MODES):
            values = [matched[condition, mode].get("finishes", np.nan) for condition in conditions]
            bars = ax.bar(positions + offset, values, width=width, color=colors[mode])
            ax.bar_label(bars, [f"{row.get('finishes', 0)}/{row['episodes']}" if row["episodes"] else ""
                                for condition in conditions for row in [matched[condition, mode]]],
                         fontsize=8, padding=2)
        ax.set(ylabel="Routes completed", ylim=(0, 6), yticks=range(6),
               title="Paired CARLA  ·  6 m target and 10 m true-gap cutoff")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    axes[-1].set_xticks(positions, [condition.replace("_", "\n").title()
                                  for condition in conditions])
    figure.suptitle("Distance CNN retraining with braking trajectories", fontsize=15)
    figure.savefig(output / "comparison.png", dpi=210)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--routes", type=Path, default=Path("specialists/routes.json"))
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    parser.add_argument("--offline-only", action="store_true",
                        help="Write the offline comparison when CARLA rollouts are unavailable.")
    args = parser.parse_args()
    conditions = tuple(dict.fromkeys(args.conditions))
    training_settings = read_json(args.experiment / "settings.json")
    offline, offline_rows, bins, training = load_offline(args.experiment, conditions, training_settings)
    live_rows, paired, settings, live_summary = [], [], None, []
    if not args.offline_only:
        routes = read_json(args.routes)["test"]
        check(len(routes) == len(SEEDS) and len({route["name"] for route in routes}) == len(SEEDS),
              "Expected five distinct test routes in seed order")
        live_rows, paired, settings = load_live(args.experiment, conditions, routes, offline)
        live_summary = live_totals(live_rows, conditions)
        write_csv(args.experiment / "live_runs.csv", [record for record, _ in live_rows])
        # A missing match is a blank metric rather than an observed zero.
        fields = list(next(row for row in live_summary if row["episodes"]))
        write_csv(args.experiment / "live_summary.csv", [
            {field: row.get(field) for field in fields} for row in live_summary
        ])
    write_csv(args.experiment / "offline_summary.csv", offline_rows)
    write_csv(args.experiment / "offline_gap_bins.csv", bins)
    summary = {
        "mode_labels": MODES,
        "offline": offline_rows,
        "offline_aggregation": "Pooled metrics weight each aligned frame equally within its source.",
        "training": training,
        "split_note": "Existing 800-trajectory split is episode-disjoint. Braking trajectories with "
                      "identical recorded routes are grouped within each condition.",
        "live_status": "offline_only" if args.offline_only else "completed",
        "live_settings": settings,
        "live_routes": None if args.offline_only else str(args.routes),
        "start_matching_tolerances": START_TOLERANCES,
        "live_comparison": live_summary,
        "live_start_checks": paired,
    }
    (args.experiment / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    plot_comparison(args.experiment, conditions, offline, live_summary)
    for source in SOURCE_COUNTS:
        before, after = [row for row in offline_rows if row["condition"] == "all" and row["source"] == source]
        print(f"{source.capitalize()} holdout MAE {before['mae_m']:.3f} → {after['mae_m']:.3f} m "
              f"over {after['episodes']} episodes and {after['frames']:,} frames")
    if live_summary:
        for mode in MODES:
            row = next(row for row in live_summary if row["condition"] == "all"
                       and row["mode"] == mode and row["scope"] == "matched_pairs")
            print(f"{MODES[mode]} completed {row.get('finishes', 0)}/{row['episodes']} matched routes")


if __name__ == "__main__":
    main()
