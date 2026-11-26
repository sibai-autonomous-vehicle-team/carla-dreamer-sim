"""Fine-tune ResNet-18 to estimate the lead-car gap from CARLA camera frames."""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


CONDITIONS = tuple(
    f"{light}_{weather}"
    for light in ("day", "sunset", "night")
    for weather in ("sunny", "cloudy", "rainy")
)


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


def split_episodes(count, seed):
    episode_ids = list(range(count))
    random.Random(seed).shuffle(episode_ids)
    train_end = 8 * count // 10
    val_end = 9 * count // 10
    return dict(seed=seed, train_ids=episode_ids[:train_end],
                val_ids=episode_ids[train_end:val_end], test_ids=episode_ids[val_end:])


def load_episode_split(path, episode_count):
    split = json.loads(path.read_text())
    keys = ("train_ids", "val_ids", "test_ids")
    if not isinstance(split, dict) or type(split.get("seed")) is not int:
        raise ValueError("The episode split must contain an integer seed.")
    seen = set()
    for key in keys:
        ids = split.get(key)
        if not isinstance(ids, list) or not ids or any(type(i) is not int for i in ids):
            raise ValueError(f"{key} must be a nonempty list of integer episode IDs.")
        if any(not 0 <= i < episode_count for i in ids):
            raise ValueError(f"{key} contains IDs outside the dataset's {episode_count} episodes.")
        unique = set(ids)
        if len(unique) != len(ids) or seen & unique:
            raise ValueError("Episode IDs must be unique within each split and disjoint across splits.")
        seen.update(unique)
    if len(seen) != episode_count:
        raise ValueError(f"The split covers {len(seen)} of the dataset's {episode_count} episodes.")
    return {key: split[key] for key in ("seed", *keys)}


def pooled_batches(datasets, episode_refs, batch_size, shuffle):
    for condition, episode_id in episode_refs:
        root, distances = datasets[condition]
        yield from episode_batches(root, distances, (episode_id,), batch_size, shuffle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--data", type=Path, default=Path("dataset/day_cloudy"))
    inputs.add_argument("--mixed-data", type=Path,
                        help="Archive directory containing the nine condition subdirectories.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int,
                        help="Training seed. Defaults to the supplied split's seed or 42.")
    splits = parser.add_mutually_exclusive_group()
    splits.add_argument("--split", type=Path,
                        help="Use a saved seed and train_ids, val_ids, and test_ids for --data.")
    splits.add_argument("--split-dir", type=Path,
                        help="Use one saved <condition>.json split per condition for --mixed-data.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--amp", action="store_true",
                        help="Train with CUDA automatic mixed precision and channels-last layout.")
    args = parser.parse_args()
    if args.batch_size < 1 or args.epochs < 1:
        parser.error("--batch-size and --epochs must be positive")
    if args.mixed_data and not args.output:
        parser.error("--mixed-data requires --output to save the per-condition splits")
    if args.mixed_data and args.split:
        parser.error("--split is supported for single-condition training with --data")
    if args.split_dir and not args.mixed_data:
        parser.error("--split-dir requires --mixed-data")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Run with .venv-cnn/bin/python on the GPU desktop.")

    device = torch.device("cuda")
    amp_dtype = None
    if args.amp:
        amp_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False)
                     else torch.float16)
    precision = "float32" if amp_dtype is None else str(amp_dtype).removeprefix("torch.")
    print(f"Training precision: {precision}; validation precision: float32")
    roots = ([args.mixed_data / condition for condition in CONDITIONS]
             if args.mixed_data else [args.data])
    seed = args.seed if args.seed is not None else 42
    datasets = {}
    condition_splits = {}
    train_refs, val_refs, test_refs = [], [], []
    for root in roots:
        distance_file = root / "distances.pth"
        if not distance_file.is_file():
            parser.error(f"Missing distance archive: {distance_file}")
        distances = torch.load(distance_file, map_location="cpu", weights_only=True)
        if len(distances) < 10:
            parser.error(f"At least 10 episodes are required in {root} for the split")
        # Keep the specialist's complete-episode split independently for each condition.
        split_path = args.split_dir / f"{root.name}.json" if args.split_dir else args.split
        if split_path:
            try:
                condition_split = load_episode_split(split_path, len(distances))
            except (OSError, ValueError) as error:
                parser.error(f"{split_path}: {error}")
        else:
            condition_split = split_episodes(len(distances), seed)
        datasets[root.name] = (root, distances)
        condition_splits[root.name] = condition_split
        train_refs.extend((root.name, i) for i in condition_split["train_ids"])
        val_refs.extend((root.name, i) for i in condition_split["val_ids"])
        test_refs.extend((root.name, i) for i in condition_split["test_ids"])
    if args.mixed_data:
        split_seed = condition_splits[roots[0].name]["seed"]
        if any(s["seed"] != split_seed for s in condition_splits.values()):
            parser.error("Saved per-condition splits must use the same seed")
        split = dict(seed=split_seed, mode="pooled", conditions={
            root.name: dict(data=str(root), **condition_splits[root.name]) for root in roots
        })
    else:
        split = condition_splits[args.data.name]
    if args.seed is None:
        seed = split["seed"]
    torch.manual_seed(seed)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "split.json").write_text(json.dumps(split, indent=2) + "\n")
        (args.output / "epochs.csv").write_text("epoch,train_huber_loss,validation_mae_m\n")
        if args.mixed_data:
            split_dir = args.output / "splits"
            split_dir.mkdir(exist_ok=True)
            for condition, condition_split in condition_splits.items():
                (split_dir / f"{condition}.json").write_text(json.dumps(condition_split, indent=2) + "\n")
    print(f"Episodes: {len(train_refs)} train, {len(val_refs)} validation, {len(test_refs)} test")

    print("Loading pretrained ResNet-18 weights...")
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    if args.amp:
        model.to(device, memory_format=torch.channels_last)
    else:
        model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    loss_fn = nn.HuberLoss()
    preprocess = weights.transforms()
    mean = torch.tensor(preprocess.mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(preprocess.std, device=device).view(1, 3, 1, 1)

    def predict(frames):
        if args.amp:
            # Transfer one byte per value, then normalize in FP32 on-device.
            images = frames.to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
        else:
            images = frames.permute(0, 3, 1, 2).to(device, dtype=torch.float32) / 255.0
        return model((images - mean) / std).squeeze(1)

    start = time.perf_counter()

    checkpoint = Path(f"distance_resnet18_{roots[0].name}_{time.time_ns()}.pth")
    if args.output:
        checkpoint = args.output / "best.pth"
    best_mae = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        # A new generator is needed for each pass through the training data.
        epoch_refs = train_refs.copy()
        random.Random(seed + epoch).shuffle(epoch_refs)
        train_batches = pooled_batches(datasets, epoch_refs, args.batch_size, shuffle=True)
        train_loss = torch.zeros((), device=device, dtype=torch.float64) if args.amp else 0.0
        train_count = 0
        for step, (frames, gaps) in enumerate(train_batches, start=1):
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=amp_dtype, enabled=args.amp):
                predictions = predict(frames)
            loss = loss_fn(predictions.float(), gaps.to(device, non_blocking=args.amp))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # Keep the running sum on-device in the fast path.
            batch_loss = loss.detach().double() if args.amp else loss.item()
            train_loss += batch_loss * gaps.numel()
            train_count += gaps.numel()
            if step % 50 == 0:
                print(
                    f"Epoch {epoch}/{args.epochs} | "
                    f"batch {step} | loss {loss.item():.3f}"
                )

        model.eval()
        # Validation deliberately runs without autocast.
        absolute_error = torch.zeros((), device=device, dtype=torch.float64) if args.amp else 0.0
        count = 0
        with torch.inference_mode():
            val_batches = pooled_batches(datasets, val_refs, args.batch_size, shuffle=False)
            for frames, gaps in val_batches:
                batch_error = (predict(frames) - gaps.to(device, non_blocking=args.amp)).abs().sum()
                absolute_error += batch_error.double() if args.amp else batch_error.item()
                count += gaps.numel()

        if count == 0:
            raise ValueError("The validation episodes contain no aligned frame-gap pairs.")
        mae = float(absolute_error) / count
        if args.output:
            with (args.output / "epochs.csv").open("a", newline="") as stream:
                csv.writer(stream).writerow((epoch, float(train_loss) / train_count, mae))
        print(f"Epoch {epoch}/{args.epochs} | validation MAE: {mae:.3f} m")

        # Keep the model with the lowest validation error across all epochs.
        if mae < best_mae:
            best_mae = mae
            temporary = checkpoint.with_suffix(".pth.tmp")
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "validation_mae": mae,
                **split,
                **({"training_seed": seed} if seed != split["seed"] else {}),
                "batch_size": args.batch_size,
                "training_precision": precision,
                "validation_precision": "float32",
                "memory_format": "channels_last" if args.amp else "contiguous",
            }, temporary)
            temporary.replace(checkpoint)
            print(f"Saved best checkpoint to {checkpoint}")

    print(f"Best validation MAE: {best_mae:.3f} m")
    print(f"Training + validation time: {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
