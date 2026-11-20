"""Evaluate a saved distance CNN on reserved episodes or an external archive."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet18

from train_distance_cnn import episode_batches


def test_split_ids(split, episode_count):
    required = ("train_ids", "val_ids", "test_ids")
    if not isinstance(split, dict) or not all(key in split for key in required):
        raise ValueError("The episode split must contain train_ids, val_ids, and test_ids.")
    seen = set()
    for key in required:
        ids = split[key]
        if not isinstance(ids, (list, tuple)) or any(type(episode_id) is not int for episode_id in ids):
            raise ValueError(f"{key} must contain integer episode IDs.")
        if key == "test_ids" and not ids:
            raise ValueError("test_ids must contain at least one episode.")
        if any(not 0 <= episode_id < episode_count for episode_id in ids):
            raise ValueError(f"{key} contains IDs outside the target archive's {episode_count} episodes.")
        unique = set(ids)
        if len(unique) != len(ids) or seen & unique:
            raise ValueError("Episode IDs must be unique within each split and disjoint across splits.")
        seen.update(unique)
    return list(split["test_ids"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("dataset/day_cloudy"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path)
    split_args = parser.add_mutually_exclusive_group()
    split_args.add_argument("--all-episodes", action="store_true",
                            help="Evaluate every trajectory in an external dataset.")
    split_args.add_argument("--split", type=Path,
                            help="Use train_ids, val_ids, and test_ids from the target dataset's split.json.")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    distances = torch.load(args.data / "distances.pth", map_location="cpu", weights_only=True)
    provenance = {
        "dataset_path": str(args.data.resolve()),
        "checkpoint_path": str(args.checkpoint.resolve()),
        "split_source": "all_episodes" if args.all_episodes else "external" if args.split else "checkpoint",
        "split_path": str(args.split.resolve()) if args.split else None,
    }
    if args.all_episodes:
        test_ids = list(range(len(distances)))
    else:
        if args.split is None and not {"train_ids", "val_ids", "test_ids"} <= checkpoint.keys():
            parser.error("The checkpoint has no saved episode split. Supply --split or use --all-episodes.")
        try:
            split = json.loads(args.split.read_text()) if args.split else checkpoint
            test_ids = test_split_ids(split, len(distances))
        except (OSError, ValueError) as error:
            parser.error(str(error))

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    print(f"Device: {device}")
    print(f"Test episodes: {len(test_ids)}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'not recorded')}")
    predictions, targets, examples = [], [], {}
    with torch.inference_mode():
        for frames, gaps in episode_batches(
            args.data, distances, test_ids, args.batch_size, shuffle=False
        ):
            images = frames.permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 255.0
            predicted = model((images - mean) / std).squeeze(1).cpu()
            predictions.append(predicted)
            targets.append(gaps.cpu())
            if args.output:
                masks = gap_masks(gaps)
                for label, mask in masks.items():
                    candidates = mask.nonzero().flatten()
                    if not len(candidates):
                        continue
                    worst = candidates[(predicted[candidates] - gaps[candidates]).abs().argmax()]
                    error = float(abs(predicted[worst] - gaps[worst]))
                    if label not in examples or error > examples[label][0]:
                        examples[label] = (error, frames[worst].clone(),
                                           float(gaps[worst]), float(predicted[worst]))

    if not targets:
        raise ValueError("No aligned test pairs were found.")
    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
        raise ValueError("Nonfinite predictions or distance labels were found.")
    errors = predictions - targets
    absolute_errors = errors.abs()
    metrics = {
        **provenance,
        "test_episodes": len(test_ids),
        "test_frames": len(targets),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "mae_m": float(absolute_errors.mean()),
        "rmse_m": float(errors.square().mean().sqrt()),
        "signed_error_m": float(errors.mean()),
        "p95_absolute_error_m": float(torch.quantile(absolute_errors, 0.95)),
        "max_absolute_error_m": float(absolute_errors.max()),
        "bins": {},
    }
    for label, mask in gap_masks(targets).items():
        metrics["bins"][label] = {
            "frames": int(mask.sum()),
            "mae_m": float(absolute_errors[mask].mean()) if mask.any() else None,
            "signed_error_m": float(errors[mask].mean()) if mask.any() else None,
        }
    print(json.dumps(metrics, indent=2))
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        torch.save({"prediction": predictions, "target": targets, "test_ids": test_ids, **provenance},
                   args.output / "test_predictions.pth")
        plot_results(args.output, predictions, targets, examples)


def gap_masks(targets):
    return {
        "[0, 3) m": (targets >= 0) & (targets < 3),
        "[3, 6) m": (targets >= 3) & (targets < 6),
        "[6, 8) m": (targets >= 6) & (targets < 8),
        "[8, 10] m": (targets >= 8) & (targets <= 10),
        ">10 m": targets > 10,
    }


def plot_results(output, predictions, targets, examples):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    ax.scatter(targets, predictions, s=6, alpha=0.18, linewidths=0)
    limits = [min(targets.min(), predictions.min()), max(targets.max(), predictions.max())]
    ax.plot(limits, limits, "--", color="black", linewidth=1, label="Perfect prediction")
    ax.set(xlabel="True gap (m)", ylabel="Predicted gap (m)", xlim=limits, ylim=limits,
           aspect="equal")
    ax.legend()
    fig.savefig(output / "predicted_vs_true.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(examples), figsize=(3 * len(examples), 3.4),
                             squeeze=False, layout="constrained")
    for ax, (label, (_, frame, true, predicted)) in zip(axes.flat, examples.items()):
        ax.imshow(frame)
        ax.set_title(f"{label}\nTrue {true:.2f} m / CNN {predicted:.2f} m")
        ax.axis("off")
    fig.suptitle("Largest absolute error in each true-gap bin")
    fig.savefig(output / "test_camera_frames.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
