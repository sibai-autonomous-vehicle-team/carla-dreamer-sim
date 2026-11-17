"""Fine-tune ResNet-18 to estimate the lead-car gap from CARLA camera frames."""

import argparse
import random
import time
from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def episode_batches(root, distances, episode_ids, batch_size, shuffle):
    for episode_id in episode_ids:
        path = root / "obses" / f"episode_{episode_id:03d}.pth"
        frames = torch.load(path, map_location="cpu", weights_only=True)
        gaps = distances[episode_id]
        if len(frames) != len(gaps):
            raise ValueError(f"Frame and distance counts differ in {path}")

        # Frames are saved before the action and gaps after it.
        # Match frame 1 to gap 0, frame 2 to gap 1, and so on.
        frames = frames[1:]
        gaps = gaps[:-1]
        order = torch.randperm(len(gaps)) if shuffle else torch.arange(len(gaps))
        for indices in order.split(batch_size):
            if indices.numel():
                yield frames[indices], gaps[indices]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("dataset/day_cloudy"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    if args.batch_size < 1 or args.epochs < 1:
        parser.error("--batch-size and --epochs must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Run with .venv-cnn/bin/python on the GPU desktop.")

    torch.manual_seed(42)
    device = torch.device("cuda")
    distances = torch.load(args.data / "distances.pth", map_location="cpu", weights_only=True)
    if len(distances) < 10:
        parser.error("At least 10 episodes are required for the train/validation/test split.")

    # Split entire episodes so adjacent frames cannot leak between sets.
    episode_ids = list(range(len(distances)))
    random.Random(42).shuffle(episode_ids)
    train_end = 8 * len(episode_ids) // 10
    val_end = 9 * len(episode_ids) // 10
    train_ids = episode_ids[:train_end]
    val_ids = episode_ids[train_end:val_end]
    test_ids = episode_ids[val_end:]
    print(f"Episodes: {len(train_ids)} train, {len(val_ids)} validation, {len(test_ids)} test")

    print("Loading pretrained ResNet-18 weights...")
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.HuberLoss()
    preprocess = weights.transforms()
    mean = torch.tensor(preprocess.mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(preprocess.std, device=device).view(1, 3, 1, 1)

    def predict(frames):
        images = frames.permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 255.0
        return model((images - mean) / std).squeeze(1)

    start = time.perf_counter()

    checkpoint = Path(f"distance_resnet18_{args.data.name}_{time.time_ns()}.pth")
    best_mae = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        # A new generator is needed for each pass through the training data.
        epoch_ids = train_ids.copy()
        random.Random(42 + epoch).shuffle(epoch_ids)
        train_batches = episode_batches(
            args.data, distances, epoch_ids, args.batch_size, shuffle=True
        )
        for step, (frames, gaps) in enumerate(train_batches, start=1):
            optimizer.zero_grad()
            predictions = predict(frames)
            loss = loss_fn(predictions, gaps.to(device))
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                print(
                    f"Epoch {epoch}/{args.epochs} | "
                    f"batch {step} | loss {loss.item():.3f}"
                )

        model.eval()
        absolute_error = 0.0
        count = 0
        with torch.inference_mode():
            val_batches = episode_batches(
                args.data, distances, val_ids, args.batch_size, shuffle=False
            )
            for frames, gaps in val_batches:
                absolute_error += (predict(frames) - gaps.to(device)).abs().sum().item()
                count += gaps.numel()

        if count == 0:
            raise ValueError("The validation episodes contain no aligned frame-gap pairs.")
        mae = absolute_error / count
        print(f"Epoch {epoch}/{args.epochs} | validation MAE: {mae:.3f} m")

        # Keep the model with the lowest validation error across all epochs.
        if mae < best_mae:
            best_mae = mae
            temporary = checkpoint.with_suffix(".pth.tmp")
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "validation_mae": mae,
                "train_ids": train_ids,
                "val_ids": val_ids,
                "test_ids": test_ids,
                "batch_size": args.batch_size,
            }, temporary)
            temporary.replace(checkpoint)
            print(f"Saved best checkpoint to {checkpoint}")

    print(f"Best validation MAE: {best_mae:.3f} m")
    print(f"Training + validation time: {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
