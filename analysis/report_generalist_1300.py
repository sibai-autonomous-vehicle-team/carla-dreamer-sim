"""Report paired test performance for the original and expanded generalists."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import torch

from analysis.summarize_generalist import error_metrics, read_evaluation, read_json
from evaluate_distance_cnn import gap_masks
from train_distance_cnn import CONDITIONS, load_episode_split


REPO = Path(__file__).resolve().parents[1]
OLD_TRAINING = REPO / "experiments/generalist_all_conditions_2026-11-18/training"
MERGED_DATA = REPO / "experiments/specialists_braking_1300_2026-11-23/data"
PARTITIONS = ("train_ids", "val_ids", "test_ids")
HOLDOUTS = ("original", "braking")
MODELS = ("old_800", "new_1300")
SEED = 42


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def checkpoint_selection(directory, checkpoint, expected_epochs=10):
    with (directory / "epochs.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    require([int(row["epoch"]) for row in rows] == list(range(1, expected_epochs + 1)),
            f"{directory} has an incomplete or out-of-order training log")
    best = min(rows, key=lambda row: float(row["validation_mae_m"]))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    epoch = int(best["epoch"])
    validation = float(best["validation_mae_m"])
    require(saved["epoch"] == epoch and saved["validation_mae"] == validation,
            f"{checkpoint} is not the checkpoint selected by minimum validation MAE")
    split = read_json(directory / "split.json")
    require(split["seed"] == SEED and split["mode"] == "pooled"
            and split["conditions"] == saved["conditions"]
            and saved["seed"] == SEED and saved.get("training_seed", SEED) == SEED
            and set(split["conditions"]) == set(CONDITIONS),
            f"Unexpected training split, seed, or conditions in {checkpoint}")
    return {
        "checkpoint": str(checkpoint.resolve()),
        "epoch": epoch,
        "validation_mae_m": validation,
        "epochs": expected_epochs,
        "seed": SEED,
        "split_path": str((directory / "split.json").resolve()),
    }, split["conditions"]


def baseline_selection(directory, historical_training):
    checkpoint = directory / "cnn.pth"
    metadata = read_json(directory / "cnn.json")
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    historical_checkpoint = historical_training / "best.pth"
    require(checksum == metadata["cnn_sha256"]
            and checksum == hashlib.sha256(historical_checkpoint.read_bytes()).hexdigest(),
            f"{checkpoint} differs from the documented historical checkpoint")
    selected, splits = checkpoint_selection(historical_training, checkpoint)
    require(metadata["seed"] == selected["seed"]
            and metadata["selected_epoch"] == selected["epoch"]
            and metadata["validation_mae_m"] == selected["validation_mae_m"]
            and set(metadata["conditions"]) == set(CONDITIONS),
            f"Historical checkpoint metadata differs in {directory}")
    return {**selected, "checkpoint_sha256": checksum,
            "historical_checkpoint": str(historical_checkpoint.resolve())}, splits


def condition_splits(condition, data, old_training, new_training, splits):
    source = data / condition
    original = load_episode_split(source / "split_original.json", 800)
    braking = read_json(source / "split_braking.json")
    merged = load_episode_split(source / "split.json", 1300)
    expected = {}
    for model, directory, episode_count, reference in (
        ("old_800", old_training, 800, original),
        ("new_1300", new_training, 1300, merged),
    ):
        recorded = load_episode_split(directory / "splits" / f"{condition}.json", episode_count)
        require(recorded == reference
                and all(recorded[key] == splits[model][condition][key] for key in recorded),
                f"{model} training split disagrees with the source IDs in {condition}")
        expected[model] = {key: len(recorded[key]) for key in PARTITIONS}
    require(original["seed"] == braking["seed"] == merged["seed"] == SEED,
            f"Unexpected split seed in {condition}")
    require(tuple(len(original[key]) for key in PARTITIONS) == (640, 80, 80)
            and tuple(len(braking[key]) for key in PARTITIONS) == (400, 50, 50)
            and sorted(episode for key in PARTITIONS for episode in braking[key])
            == list(range(800, 1300))
            and all(merged[key] == original[key] + braking[key] for key in PARTITIONS),
            f"Original and braking partitions do not reproduce the merged split in {condition}")
    return {"original": original, "braking": braking}, expected


def record_braking_routes(source, split, routes, scenario_seeds):
    episodes = read_json(source / "episodes.json")
    require(len(episodes) == 1300, f"Unexpected episode manifest count in {source}")
    for partition in PARTITIONS:
        for episode_id in split[partition]:
            episode = episodes[episode_id]
            require(episode["episode"] == episode_id
                    and episode["source_episode"] == episode_id - 800
                    and episode["split"] == partition.removesuffix("_ids"),
                    f"Braking episode manifest and saved split disagree in {source}")
            routes[partition][episode["exact_route_sha256"]] += 1
            scenario_seeds[partition].add(episode["scenario_seed"])


def read_paired_evaluations(directory, condition, dataset, summaries, holdout, ids):
    expected_split = str((dataset / f"split_{holdout}.json").resolve())
    expected_dataset = str(dataset.resolve())
    targets = None
    errors = {}
    for model in MODELS:
        path = directory / model / condition / holdout
        metrics, error, current_targets = read_evaluation(path, ids)
        summary = summaries[model]
        require(metrics["checkpoint_path"] == summary["checkpoint"]
                and metrics["checkpoint_epoch"] == summary["epoch"]
                and metrics["dataset_path"] == expected_dataset
                and metrics["split_source"] == "external"
                and metrics["split_path"] == expected_split,
                f"Unexpected checkpoint, dataset, or split in {path}")
        if targets is not None:
            require(torch.equal(targets, current_targets),
                    f"Ordered target frames differ between checkpoints in {path}")
        targets = current_targets
        errors[model] = error
    return errors, targets


def compare_errors(errors, count):
    metrics = {model: error_metrics(errors[model]) if count else
               dict.fromkeys(("mae_m", "rmse_m", "signed_error_m", "p95_absolute_error_m",
                              "max_absolute_error_m")) for model in MODELS}
    return {
        **{f"{model}_{key}": value for model, result in metrics.items()
           for key, value in result.items()},
        "new_minus_old_mae_m": (
            metrics["new_1300"]["mae_m"] - metrics["old_800"]["mae_m"] if count else None
        ),
    }


def compare_holdout(holdout, condition, episode_count, errors, targets):
    row = {
        "holdout": holdout,
        "condition": condition,
        "test_episodes": episode_count,
        "test_frames": targets.numel(),
        **compare_errors(errors, targets.numel()),
    }
    bins = []
    for label, mask in gap_masks(targets).items():
        count = int(mask.sum())
        bins.append({
            "holdout": holdout,
            "condition": condition,
            "true_gap_bin": label,
            "test_frames": count,
            **compare_errors({model: errors[model][mask] for model in MODELS}, count),
        })
    return row, bins


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.2), sharex=True, sharey=True,
                             layout="constrained")
    titles = {"original": "Original following holdout", "braking": "Braking holdout"}
    colors = ("#407396", "#dc813e")
    labels = ("Generalist trained on 800 per condition", "Generalist trained on 1,300 per condition")
    conditions = (*CONDITIONS, "Pooled")
    width = 0.36
    for holdout, ax in zip(HOLDOUTS, axes):
        group = {row["condition"]: row for row in rows if row["holdout"] == holdout}
        for model, label, color, direction in zip(MODELS, labels, colors, (-1, 1)):
            ax.bar([i + direction * width / 2 for i in range(len(conditions))],
                   [group[condition][f"{model}_mae_m"] for condition in conditions],
                   width, color=color, label=label)
        ax.set_title(titles[holdout], loc="left")
        ax.set_ylabel("Test MAE (m)")
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.25)
        ax.axvline(len(CONDITIONS) - 0.5, color="#777777", linewidth=0.8)
    axes[0].legend(loc="upper left", ncol=2, frameon=False, fontsize=9)
    axes[-1].set_xticks(range(len(conditions)), [c.replace("_", "\n") for c in conditions])
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=MERGED_DATA)
    parser.add_argument("--old-training", type=Path, default=OLD_TRAINING)
    args = parser.parse_args()
    experiment = args.experiment
    new_training = experiment / "training"
    summaries, splits = {}, {}
    summaries["old_800"], splits["old_800"] = baseline_selection(
        experiment / "baseline", args.old_training,
    )
    summaries["new_1300"], splits["new_1300"] = checkpoint_selection(
        new_training, new_training / "best.pth",
    )

    rows, bins = [], []
    counts = {model: dict.fromkeys(PARTITIONS, 0) for model in MODELS}
    pooled_targets = {holdout: [] for holdout in HOLDOUTS}
    pooled_errors = {holdout: {model: [] for model in MODELS} for holdout in HOLDOUTS}
    pooled_episodes = dict.fromkeys(HOLDOUTS, 0)
    braking_routes = {partition: Counter() for partition in PARTITIONS}
    braking_seeds = {partition: set() for partition in PARTITIONS}
    for condition in CONDITIONS:
        dataset = args.data / condition
        holdouts, split_counts = condition_splits(
            condition, args.data, args.old_training, new_training, splits,
        )
        record_braking_routes(dataset, holdouts["braking"], braking_routes, braking_seeds)
        for model in MODELS:
            for key in PARTITIONS:
                counts[model][key] += split_counts[model][key]
        for holdout, split in holdouts.items():
            ids = split["test_ids"]
            errors, targets = read_paired_evaluations(
                experiment / "evaluation", condition, dataset, summaries, holdout, ids,
            )
            row, group = compare_holdout(holdout, condition, len(ids), errors, targets)
            rows.append(row)
            bins.extend(group)
            pooled_episodes[holdout] += len(ids)
            pooled_targets[holdout].append(targets)
            for model in MODELS:
                pooled_errors[holdout][model].append(errors[model])

    pooled = {}
    for holdout in HOLDOUTS:
        row, group = compare_holdout(
            holdout, "Pooled", pooled_episodes[holdout],
            {model: torch.cat(pooled_errors[holdout][model]) for model in MODELS},
            torch.cat(pooled_targets[holdout]),
        )
        pooled[holdout] = row
        bins.extend(group)

    train_routes, test_routes = braking_routes["train_ids"], braking_routes["test_ids"]
    shared_routes = train_routes.keys() & test_routes.keys()
    summary = {
        "comparison_basis": "Paired offline predictions on exact ordered test IDs and true targets. "
        "Pooled metrics weight each frame equally. Signed error is predicted gap minus true gap.",
        "braking_route_overlap": {
            "exact_directed_routes_in_train_and_test": len(shared_routes),
            "test_episodes_on_train_route_geometries": sum(test_routes[route] for route in shared_routes),
            "scenario_seeds_in_train_and_test": len(braking_seeds["train_ids"] & braking_seeds["test_ids"]),
            "description": "Braking route grouping was applied per condition, so identical route "
            "geometries can cross partitions in the pooled experiment.",
        },
        "dataset": str(args.data.resolve()),
        "training": {model: {**summaries[model], "episode_counts": counts[model]} for model in MODELS},
        "pooled": pooled,
        "conditions": rows,
        "gap_bins": bins,
    }
    write_csv(experiment / "comparison.csv", [*rows, *pooled.values()])
    write_csv(experiment / "gap_bins.csv", bins)
    (experiment / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    plot_comparison([*rows, *pooled.values()], experiment / "comparison.png")
    for holdout, row in pooled.items():
        print(f"{holdout}: {row['test_episodes']:,} episodes, {row['test_frames']:,} frames | "
              f"MAE from {row['old_800_mae_m']:.3f} m to {row['new_1300_mae_m']:.3f} m")


if __name__ == "__main__":
    main()
