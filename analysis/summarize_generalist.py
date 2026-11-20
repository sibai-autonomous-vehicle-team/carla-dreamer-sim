"""Compare pooled CNN predictions with the archived specialist evaluations."""
import argparse
import csv
import json
import math
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "experiments"
SPECIALISTS = EXPERIMENTS / "all_specialists_2026-11-17"
DAY_ARCHIVE = EXPERIMENTS / "day_cloudy_resnet18_pid_2026-11-17"
DAY_DEPLOYED = EXPERIMENTS / "day_cloudy_cnn_close_v3_2026-11-17"
SUNSET = EXPERIMENTS / "sunset_cloudy_specialist_2026-11-17"
CONDITIONS = (
    "day_sunny", "day_cloudy", "day_rainy",
    "sunset_sunny", "sunset_cloudy", "sunset_rainy",
    "night_sunny", "night_cloudy", "night_rainy",
)


def read_json(path):
    return json.loads(path.read_text())


def historical_archive(condition):
    if condition == "day_cloudy":
        return DAY_ARCHIVE / "split.json", DAY_ARCHIVE
    experiment = SUNSET if condition == "sunset_cloudy" else SPECIALISTS / condition
    return experiment / "archive" / "split.json", experiment / "archive_evaluation"


def read_evaluation(directory, test_ids):
    metrics = read_json(directory / "test_metrics.json")
    data = torch.load(directory / "test_predictions.pth", map_location="cpu", weights_only=True)
    prediction, target = data["prediction"], data["target"]
    if list(data["test_ids"]) != test_ids or metrics["test_episodes"] != len(test_ids):
        raise ValueError(f"Evaluation episode IDs or count differ in {directory}")
    if prediction.shape != target.shape or target.ndim != 1 or target.numel() != metrics["test_frames"]:
        raise ValueError(f"Evaluation frame shapes or count differ in {directory}")
    if not target.numel() or not (torch.isfinite(target).all() and torch.isfinite(prediction).all()):
        raise ValueError(f"Missing or nonfinite distances in {directory}")
    return metrics, prediction.to(torch.float64) - target.to(torch.float64), target


def error_metrics(error):
    absolute = error.abs()
    return {
        "mae_m": absolute.mean().item(),
        "rmse_m": error.square().mean().sqrt().item(),
        "signed_error_m": error.mean().item(),
        "p95_absolute_error_m": torch.quantile(absolute, 0.95).item(),
        "max_absolute_error_m": absolute.max().item(),
    }


def verified_deployed_mae(condition, test_ids, targets, comparison):
    count = targets.numel()
    if int(comparison["archived_test_frames"]) != count:
        raise ValueError(f"Published specialist frame count differs for {condition}")
    expected = float(comparison["native_archived_prediction_mae_m"])
    if condition == "day_cloudy":
        report = read_json(DAY_DEPLOYED / "finetune_v3" / "summary.json")
        # This legacy evaluation saved aggregate statistics and the reference checkpoint.
        legacy_reference = DAY_ARCHIVE / "best.pth"
        if Path(report["legacy_reference"]).resolve() != legacy_reference.resolve():
            raise ValueError("Day-cloudy deployed specialist used a different archive checkpoint")
        split = torch.load(legacy_reference, map_location="cpu", weights_only=True)
        legacy = report["finetuned"]["legacy_test"]["overall"]
        if list(split["test_ids"]) != test_ids or legacy["frames"] != count:
            raise ValueError("Day-cloudy deployed specialist archived split or count differs")
        mae = float(legacy["mae_m"])
    else:
        experiment = SUNSET if condition == "sunset_cloudy" else SPECIALISTS / condition
        _, errors, deployed_targets = read_evaluation(experiment / "specialist_evaluation", test_ids)
        if not torch.equal(targets, deployed_targets):
            raise ValueError(f"Deployed specialist target sequence differs for {condition}")
        mae = errors.abs().mean().item()
    if not math.isclose(mae, expected, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError(f"Published deployed specialist MAE differs for {condition}")
    return mae


def plot_comparison(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    positions = list(range(len(rows)))
    width = 0.24
    series = (
        ("generalist_mae_m", "Generalist", "#245d9c"),
        ("archive_specialist_mae_m", "Archive-only specialist", "#91b9d8"),
        ("deployed_specialist_mae_m", "Deployed specialist", "#d78a4b"),
    )
    for offset, (key, label, color) in enumerate(series, start=-1):
        ax.bar([x + offset * width for x in positions], [row[key] for row in rows],
               width, label=label, color=color)
    ax.set_xticks(positions, [row["condition"].replace("_", "\n") for row in rows])
    ax.set_ylabel("Test mean absolute error (m)")
    ax.legend()
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.23)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    experiment = parser.parse_args().experiment
    with (SPECIALISTS / "comparison.csv").open(newline="") as stream:
        comparison = {row["condition"]: row for row in csv.DictReader(stream)}
    rows, pooled_generalist, pooled_archive = [], [], []
    checkpoints = set()
    for condition in CONDITIONS:
        saved_split = experiment / "training" / "splits" / f"{condition}.json"
        archive_split, archive_evaluation = historical_archive(condition)
        split, previous = read_json(saved_split), read_json(archive_split)
        for partition in ("train_ids", "val_ids", "test_ids"):
            if sorted(split[partition]) != sorted(previous[partition]):
                raise ValueError(f"{condition} {partition} differs from the specialist split")
        test_ids = list(split["test_ids"])
        metrics, errors, targets = read_evaluation(experiment / "evaluation" / condition, test_ids)
        _, archive_errors, archive_targets = read_evaluation(archive_evaluation, test_ids)
        if not torch.equal(targets, archive_targets):
            raise ValueError(f"The archived test target sequence differs for {condition}")
        checkpoints.add(metrics["checkpoint_path"])
        deployed_mae = verified_deployed_mae(condition, test_ids, targets, comparison[condition])
        generalist = error_metrics(errors)
        archive = error_metrics(archive_errors)
        rows.append({
            "condition": condition,
            "test_episodes": len(test_ids),
            "test_frames": targets.numel(),
            **{f"generalist_{key}": value for key, value in generalist.items()},
            **{f"archive_specialist_{key}": value for key, value in archive.items()},
            "deployed_specialist_mae_m": deployed_mae,
        })
        pooled_generalist.append(errors)
        pooled_archive.append(archive_errors)
    if len(checkpoints) != 1:
        raise ValueError("The condition evaluations must all use the same generalist checkpoint")

    frames = sum(row["test_frames"] for row in rows)
    summary = {
        "generalist_checkpoint": checkpoints.pop(),
        "test_episodes": sum(row["test_episodes"] for row in rows),
        "test_frames": frames,
        "pooled_generalist": error_metrics(torch.cat(pooled_generalist)),
        "pooled_archive_specialist": error_metrics(torch.cat(pooled_archive)),
        "pooled_deployed_specialist_mae_m": (
            sum(row["test_frames"] * row["deployed_specialist_mae_m"] for row in rows) / frames
        ),
        "conditions": rows,
        "comparison_basis": "Historical held-out episode splits and exact ordered target arrays. "
        "Day-cloudy deployed specialist verified using its legacy reference checkpoint and frame count.",
    }
    with (experiment / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (experiment / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    plot_comparison(rows, experiment / "comparison.png")
    print(f"Verified {summary['test_episodes']} test episodes and {frames:,} test frames")
    print(f"Pooled generalist MAE: {summary['pooled_generalist']['mae_m']:.3f} m")


if __name__ == "__main__":
    main()
