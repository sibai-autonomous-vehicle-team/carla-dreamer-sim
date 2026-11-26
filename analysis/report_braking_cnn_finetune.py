"""Verify and summarize synchronized fine-tuning of the 1,300-trajectory specialist bases."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.report_braking_cnn_retrain import (
    SOURCE_COUNTS, bin_rows, check, checkpoint_hash, error_metrics, offline_result,
    read_json, torch_load, write_csv,
)
from train_distance_cnn import CONDITIONS


BASE_EXPERIMENT = Path("experiments/specialists_braking_1300_2026-11-23")
EXPERIMENT = Path("experiments/specialists_braking_1300_sync_2026-11-24")
DEPLOYED, BASE, CANDIDATE = "deployed_800", "base_1300", "finetuned_1300"
BAND_NAMES = ("close_0_to_3_m", "standard_3_to_8_m", "standard_8_to_12_m")
EXPECTED_EPOCHS = 20


def finite_numbers(values, label):
    check(all(np.isfinite(x) and x >= 0 for x in values), f"Invalid {label}")


def validate_selection(folder, base_path):
    summary = read_json(folder / "summary.json")
    check(Path(summary["initial_checkpoint"]).resolve() == base_path.resolve(),
          f"Unexpected initial checkpoint at {folder}")
    check(summary["seed"] == 42 and summary["legacy_replay_frames"] == 0,
          f"Unexpected training recipe at {folder}")

    baseline = summary["validation_baseline_band_mae_m"]
    finite_numbers(baseline.values(), f"baseline band metrics at {folder}")
    limits = {name: min(baseline[name] * 0.9, 0.6) if name == BAND_NAMES[0]
              else baseline[name] + 0.1 for name in BAND_NAMES}
    recorded_limits = summary["validation_gate_limits_mae_m"]
    check(limits.keys() == recorded_limits.keys() and
          all(np.isclose(value, recorded_limits[key]) for key, value in limits.items()),
          f"Unexpected validation thresholds at {folder}")

    with (folder / "epochs.csv").open(newline="") as handle:
        epochs = list(csv.DictReader(handle))
    check([int(row["epoch"]) for row in epochs] == list(range(1, EXPECTED_EPOCHS + 1)),
          f"Expected {EXPECTED_EPOCHS} consecutive epochs at {folder}")
    eligible, scores = [], {}
    for row in epochs:
        epoch = int(row["epoch"])
        bands = {name: float(row[f"validation_{name}_mae"]) for name in BAND_NAMES}
        score = float(row["validation_score_m"])
        severe = int(row["validation_severe_false_far"])
        finite_numbers([*bands.values(), score, float(row["validation_mae_m"]),
                        float(row["train_huber"]), severe], f"epoch {epoch} values at {folder}")
        check(np.isclose(score, np.mean(list(bands.values())), rtol=1e-7),
              f"Validation score differs from the equal-band mean at {folder} epoch {epoch}")
        failed = {name for name, value in bands.items() if value > limits[name]}
        if severe:
            failed.add("severe_false_far")
        check(failed == set(filter(None, row["validation_failed_gates"].split(", "))),
              f"Validation gate replay differs at {folder} epoch {epoch}")
        if not failed:
            eligible.append(epoch)
        scores[epoch] = score
    best = min(eligible or list(scores), key=scores.__getitem__)
    candidate = folder / ("best.pth" if eligible else "best_diagnostic.pth")
    status = "eligible_checkpoint" if eligible else "no_eligible_checkpoint"
    selected = candidate if eligible else base_path
    checkpoint = torch_load(candidate)
    check(summary["selection_status"] == status and summary["validation_eligible_epochs"] == eligible
          and summary["best_epoch"] == best == checkpoint["epoch"]
          and summary["selected_epoch"] == (best if eligible else None),
          f"Saved checkpoint does not match the validation-only selection at {folder}")
    check(Path(summary["evaluated_candidate_checkpoint"]).resolve() == candidate.resolve() and
          Path(summary["selected_checkpoint"]).resolve() == selected.resolve() and
          np.isclose(checkpoint["validation_score_m"], scores[best], rtol=1e-7) and
          np.isclose(summary["finetuned_validation_score_m"], scores[best], rtol=1e-7),
          f"Saved checkpoint provenance or score differs at {folder}")
    best_row = epochs[best - 1]
    return summary, {
        "selection_status": status,
        "candidate_epoch": best,
        "selected_epoch": summary["selected_epoch"],
        "eligible_epochs": eligible,
        "candidate_failed_gates": list(filter(None, best_row["validation_failed_gates"].split(", "))),
        "baseline_validation_score_m": summary["original_validation_score_m"],
        "candidate_validation_score_m": scores[best],
        "candidate_checkpoint": str(candidate.resolve()),
        "candidate_sha256": checkpoint_hash(candidate.resolve()),
        "selected_checkpoint": str(selected.resolve()),
        "validation_gate_limits_mae_m": limits,
        "training_summary": str(folder / "summary.json"),
    }


def sync_results(folder, summary, base_path, selection):
    saved = torch_load(folder / "test_predictions.pth")
    target = saved["true_gap_m"].numpy().astype(np.float64)
    check(target.ndim == 1 and len(target) == summary["synchronized_split_frames"]["test"]
          and len(target) > 0 and np.isfinite(target).all() and (target >= 0).all(),
          f"Invalid synchronized labels at {folder}")
    check(len(saved["route_group"]) == len(saved["source"]) == len(target),
          f"Synchronized prediction rows disagree with route or source labels at {folder}")
    for key, mode, path in (("original", BASE, base_path),
                            ("finetuned", CANDIDATE, Path(selection["candidate_checkpoint"]))):
        prediction = saved[key].numpy().astype(np.float64)
        check(prediction.shape == target.shape and np.isfinite(prediction).all(),
              f"Invalid {key} synchronized predictions at {folder}")
        errors = prediction - target
        stats = error_metrics(errors)
        recorded = summary[key]["test"]["overall"]
        for field in ("frames", "mae_m", "signed_error_m"):
            check(np.isclose(stats[field], recorded[field], rtol=2e-5, atol=1e-6),
                  f"Recomputed {key} synchronized {field} disagrees at {folder}")
        yield ("sync", mode), target, errors, {
            "episodes": None, "checkpoint": str(path.resolve()),
            "checkpoint_sha256": checkpoint_hash(path.resolve()),
            "checkpoint_epoch": selection["candidate_epoch"] if mode == CANDIDATE else None,
            "source_metrics": str(folder / "summary.json"),
        }


def condition_results(experiment, base_experiment, condition):
    base_path = base_experiment / condition / "train" / "best.pth"
    sync = experiment / condition / "sync"
    summary, selection = validate_selection(sync, base_path)
    results = list(sync_results(sync, summary, base_path, selection))
    deployed = base_experiment / condition / "before"
    deployed_hash = read_json(deployed / "pid.json")["cnn_sha256"]
    check(checkpoint_hash((deployed / "cnn.pth").resolve()) == deployed_hash,
          f"Frozen deployed checkpoint differs from its fingerprint for {condition}")
    data = base_experiment / "data" / condition
    distances = torch_load(data / "distances.pth")
    check(len(distances) == 1300, f"Expected 1,300 trajectories in {data}")
    for source, counts in SOURCE_COUNTS.items():
        ids = read_json(data / f"split_{source}.json")["test_ids"]
        check(len(ids) == counts[-1] and len(set(ids)) == len(ids),
              f"Unexpected {source} test IDs for {condition}")
        target = np.concatenate([distances[episode][:-1].numpy() for episode in ids])
        for mode, folder, expected_hash in (
            (DEPLOYED, base_experiment / condition / "offline" / "before" / source, deployed_hash),
            (BASE, base_experiment / condition / "offline" / "after" / source, checkpoint_hash(base_path.resolve())),
            (CANDIDATE, experiment / condition / "offline" / source, selection["candidate_sha256"]),
        ):
            record, heldout, errors = offline_result(folder, data, source, ids, target)
            check(record["checkpoint_sha256"] == expected_hash,
                  f"Unexpected {mode} weights in {condition} {source}")
            metadata = {key: record[key] for key in
                        ("episodes", "checkpoint", "checkpoint_sha256", "checkpoint_epoch", "source_metrics")}
            results.append(((source, mode), heldout, errors, metadata))
    return selection, results


def summarize(experiment, base_experiment, conditions):
    metrics, bins, selections, aggregate = [], [], {}, {}
    for condition in conditions:
        selection, result = condition_results(experiment, base_experiment, condition)
        selections[condition] = selection
        for (source, mode), targets, errors, metadata in result:
            gate_status = selection["selection_status"] if mode == CANDIDATE else None
            metrics.append({"condition": condition, "source": source, "mode": mode,
                            "selection_status": gate_status, **metadata, **error_metrics(errors)})
            bins.extend(bin_rows(condition, source, mode, targets, errors))
            aggregate.setdefault((source, mode), []).append((targets, errors, metadata["episodes"]))
    for (source, mode), entries in aggregate.items():
        targets, errors, episodes = zip(*entries)
        targets, errors = np.concatenate(targets), np.concatenate(errors)
        metrics.append({
            "condition": "all", "source": source, "mode": mode, "selection_status": None,
            "episodes": sum(episodes) if episodes[0] is not None else None, "checkpoint": None,
            "checkpoint_sha256": None, "checkpoint_epoch": None, "source_metrics": None,
            **error_metrics(errors),
        })
        bins.extend(bin_rows("all", source, mode, targets, errors))
    eligible_count = sum(row["selection_status"] == "eligible_checkpoint" for row in selections.values())
    report = {
        "conditions": list(conditions),
        "eligible_conditions": eligible_count,
        "selection": selections,
        "metrics": metrics,
        "comparison_notes": [
            "Aggregate MAE weights each held-out frame equally.",
            "The fine-tuned arm includes diagnostic candidates when no epoch cleared validation.",
            "Synchronized comparison uses held-out captures from the fine-tuning manifests.",
            "The 800-lineage baseline is the frozen deployed model, which may already be fine-tuned.",
        ],
    }
    write_csv(experiment / "comparison.csv", metrics)
    write_csv(experiment / "gap_bins.csv", bins)
    (experiment / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def plot(experiment, report):
    conditions = report["conditions"]
    indexed = {(row["condition"], row["source"], row["mode"]): row for row in report["metrics"]}
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11.5, 8), layout="constrained")
    styles = {
        DEPLOYED: ("Deployed 800-lineage CNN", "#28517A"),
        BASE: ("1,300 before synchronized fine-tuning", "#7A8E9B"),
        CANDIDATE: ("1,300 synchronized candidate", "#007F78"),
    }
    positions = np.arange(len(conditions))
    for axis, source in zip(axes, ("original", "braking", "sync")):
        modes = (BASE, CANDIDATE) if source == "sync" else (DEPLOYED, BASE, CANDIDATE)
        for offset, mode in zip(np.arange(len(modes)) - (len(modes) - 1) / 2, modes):
            label, color = styles[mode]
            bars = axis.bar(positions + 0.23 * offset,
                            [indexed[condition, source, mode]["mae_m"] for condition in conditions],
                            width=0.23, color=color, label=label)
            if mode == CANDIDATE:
                for condition, bar in zip(conditions, bars):
                    if report["selection"][condition]["selection_status"] == "no_eligible_checkpoint":
                        bar.set_facecolor("white")
                        bar.set_edgecolor(color)
                        bar.set_hatch("///")
        axis.set(ylabel="Gap MAE (m)", title=f"{source.capitalize()} holdout")
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(ncols=3, frameon=False, fontsize=9)
    axes[-1].set_xticks(positions, [c.replace("_", "\n").title() for c in conditions])
    fig.suptitle("Synchronized fine-tuning of the 1,300-trajectory specialists\n"
                 "Hatched bars show diagnostic candidates that missed the validation gates",
                 fontsize=13)
    fig.savefig(experiment / "comparison.png", dpi=210)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=EXPERIMENT)
    parser.add_argument("--base-experiment", type=Path, default=BASE_EXPERIMENT)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    args = parser.parse_args()
    conditions = tuple(dict.fromkeys(args.conditions))
    report = summarize(args.experiment, args.base_experiment, conditions)
    plot(args.experiment, report)
    print(f"Validation gates passed in {report['eligible_conditions']}/{len(conditions)} conditions")
    for source in ("sync", "original", "braking"):
        pooled = {row["mode"]: row for row in report["metrics"]
                  if row["condition"] == "all" and row["source"] == source}
        before, after = pooled[BASE], pooled[CANDIDATE]
        print(f"{source.capitalize()} holdout MAE {before['mae_m']:.3f} → {after['mae_m']:.3f} m "
              f"over {after['frames']:,} matched frames")


if __name__ == "__main__":
    main()
