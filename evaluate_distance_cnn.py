"""Evaluate a saved distance CNN on its reserved test episodes."""

import argparse
from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet18

from train_distance_cnn import episode_batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("dataset/day_cloudy"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    test_ids = checkpoint["test_ids"]
    if not test_ids or set(test_ids) & (
        set(checkpoint["train_ids"]) | set(checkpoint["val_ids"])
    ):
        raise ValueError("Test episode IDs must be nonempty and separate from training and validation.")
    distances = torch.load(args.data / "distances.pth", map_location="cpu", weights_only=True)

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    print(f"Device: {device}")
    print(f"Test episodes: {len(test_ids)}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'not recorded')}")
    predictions, targets = [], []
    with torch.inference_mode():
        for frames, gaps in episode_batches(
            args.data, distances, test_ids, args.batch_size, shuffle=False
        ):
            images = frames.permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 255.0
            predictions.append(model((images - mean) / std).squeeze(1).cpu())
            targets.append(gaps.cpu())

    if not targets:
        raise ValueError("No aligned test pairs were found.")
    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
        raise ValueError("Nonfinite predictions or distance labels were found.")
    errors = predictions - targets
    absolute_errors = errors.abs()
    long_gap = targets > 10
    count = long_gap.sum().item()
    if count:
        mae = absolute_errors[long_gap].mean().item()
        print(f"True gap >10 m: {count} test frames | MAE {mae:.3f} m")
    else:
        print("True gap >10 m: 0 test frames")
    print(f"Test image-distance pairs: {len(targets)}")
    print(f"MAE: {absolute_errors.mean().item():.3f} m")
    print(f"RMSE: {errors.square().mean().sqrt().item():.3f} m")
    print(f"95th percentile absolute error: {torch.quantile(absolute_errors, 0.95).item():.3f} m")
    print(f"Maximum absolute error: {absolute_errors.max().item():.3f} m")
    print(f"Mean signed error (positive = gap overestimate): {errors.mean().item():+.3f} m")

    for lower, upper in [(0, 1), (1, 3), (3, float("inf"))]:
        mask = (targets >= lower) & (targets < upper)
        count = mask.sum().item()
        if count:
            label = f"{lower}-{upper} m" if upper != float("inf") else "3 m and above"
            print(
                f"True gap {label}: {count} pairs | "
                f"MAE {absolute_errors[mask].mean().item():.3f} m | "
                f"mean signed error {errors[mask].mean().item():+.3f} m"
            )


if __name__ == "__main__":
    main()
