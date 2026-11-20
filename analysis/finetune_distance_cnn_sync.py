"""Fine-tune a distance ResNet-18 on camera-frame-matched CARLA labels."""

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet18

from train_distance_cnn import episode_batches


SPLITS = ("train", "validation", "test")
CLOSE_MAX_MAE_M = 0.60
SEVERE_CLOSE_PREDICTION_M = 6.0


@dataclass
class Samples:
    images: torch.Tensor
    gaps: torch.Tensor
    groups: list[str]
    episodes: int
    sources: list[str]


def load_manifest(path):
    manifest = json.loads(path.read_text())
    if manifest.get("complete") is False:
        raise ValueError(f"Capture collection is still incomplete: {path}")
    declared_count = manifest.get("planned_episodes")
    if declared_count is not None and sum(len(manifest.get(split, [])) for split in SPLITS) != declared_count:
        raise ValueError(f"Capture split counts differ from the collector plan: {path}")
    group_splits, route_splits, files = {}, {}, set()
    dataset, resolved = {}, {}
    for split in SPLITS:
        if not manifest.get(split):
            raise ValueError(f"The {split} split must contain an episode")
        images, gaps, groups, sources = [], [], [], []
        resolved[split] = []
        for entry in manifest[split]:
            record = {"path": entry} if isinstance(entry, str) else entry
            file = (path.parent / record["path"]).resolve()
            if file in files:
                raise ValueError(f"Episode appears more than once in the split: {file}")
            files.add(file)
            episode = torch.load(file, map_location="cpu", weights_only=True, mmap=True)
            route = (episode.get("scenario") or {}).get("name")
            group = record.get("route_group", route)
            source = record.get("source", "standard")
            if not route or not group:
                raise ValueError(f"Missing scenario name or route group: {file}")
            if group_splits.setdefault(group, split) != split or route_splits.setdefault(route, split) != split:
                raise ValueError(f"Route {route} (group {group}) crosses dataset splits")

            frames, targets = episode["images"], episode["distances_m"].float()
            if targets.ndim != 1 or frames.dtype != torch.uint8 or frames.shape != (targets.numel(), 224, 224, 3):
                raise ValueError(f"Expected one 224 x 224 uint8 RGB image per gap: {file}")
            if not len(frames) or not torch.isfinite(targets).all() or (targets < 0).any():
                raise ValueError(f"Missing or invalid gap labels: {file}")
            frame_ids = episode.get("frame_ids")
            if frame_ids is not None and (
                frame_ids.shape != targets.shape or not (frame_ids[1:] > frame_ids[:-1]).all()
            ):
                raise ValueError(f"Missing, repeated or unordered camera frame IDs: {file}")
            images.append(frames)
            gaps.append(targets)
            groups.extend([group] * len(targets))
            sources.extend([source] * len(targets))
            resolved[split].append({"path": str(file), "route": route, "route_group": group,
                                    "source": source, "frames": len(targets)})
        dataset[split] = Samples(torch.cat(images), torch.cat(gaps), groups, len(images), sources)
    return dataset, resolved


def add_legacy_replay(dataset, root, checkpoint, generator, replay_frames):
    distances = torch.load(root / "distances.pth", map_location="cpu", weights_only=True)
    if "train_ids" not in checkpoint or "test_ids" not in checkpoint:
        raise ValueError("Supply the original checkpoint with legacy split IDs using --legacy-reference")
    train_ids, test_ids = checkpoint["train_ids"], checkpoint["test_ids"]
    if not train_ids or not test_ids or set(train_ids) & set(test_ids):
        raise ValueError("Legacy replay requires separate original training and test episode IDs")
    if any(not 0 <= episode < len(distances) for episode in (*train_ids, *test_ids)):
        raise ValueError("Legacy checkpoint split IDs are outside the distance archive")
    if replay_frames == 0:
        return dataset, distances, 0
    shuffled = torch.randperm(len(train_ids), generator=generator).tolist()
    images, gaps, groups = [dataset.images], [dataset.gaps], list(dataset.groups)
    replay_count = 0
    for index in shuffled:
        episode_id = train_ids[index]
        count = min(8, replay_frames - replay_count, len(distances[episode_id]) - 1)
        if count <= 0:
            continue
        path = root / "obses" / f"episode_{episode_id:03d}.pth"
        frames = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if len(frames) != len(distances[episode_id]) or tuple(frames.shape[1:]) != (224, 224, 3):
            raise ValueError(f"Unexpected frame layout in legacy episode {episode_id}")
        indices = torch.randperm(len(frames) - 1, generator=generator)[:count]
        images.append(frames[indices + 1])
        gaps.append(distances[episode_id][indices].float())
        groups.extend(["legacy replay"] * count)
        replay_count += count
        if replay_count == replay_frames:
            break
    sources = dataset.sources + ["legacy replay"] * replay_count
    replay = Samples(torch.cat(images), torch.cat(gaps), groups, dataset.episodes, sources)
    return replay, distances, replay_count


def gap_bins(gap):
    return {
        "0–3 m": gap < 3,
        "3–6 m": (gap >= 3) & (gap < 6),
        "6–8 m": (gap >= 6) & (gap < 8),
        "8–10 m": (gap >= 8) & (gap <= 10),
        ">10–12 m": (gap > 10) & (gap <= 12),
        ">12 m": gap > 12,
    }


def supervision_bands(gap):
    return {
        "close_0_to_3_m": gap < 3,
        "middle_3_to_8_m": (gap >= 3) & (gap < 8),
        "long_8_to_12_m": (gap >= 8) & (gap <= 12),
        "above_12_m": gap > 12,
    }


def sampling_weights(gaps, close_aware=False):
    weights = torch.ones(len(gaps))
    for mask in gap_bins(gaps).values():
        count = int(mask.sum())
        if count:
            # Temper inverse-frequency weighting so rare frames are not repeated excessively.
            weights[mask] = math.sqrt(len(gaps) / (6 * count))
    weights.clamp_(0.5, 3.0)
    if close_aware:
        # Balance scene diversity within each band without letting new close
        # footage dominate the 3–12 m following regime.
        for mask, share in zip(supervision_bands(gaps).values(), (0.25, 0.35, 0.35, 0.05)):
            if mask.any():
                weights[mask] *= share / weights[mask].sum()
    return weights


def error_stats(error):
    if not len(error):
        return {"frames": 0, "mae_m": None, "signed_error_m": None, "p95_abs_error_m": None}
    absolute = error.abs()
    return {
        "frames": len(error),
        "mae_m": float(absolute.mean()),
        "signed_error_m": float(error.mean()),
        "p95_abs_error_m": float(torch.quantile(absolute, 0.95)),
    }


def metrics(pred, gap, groups=None, sources=None):
    if not torch.isfinite(pred).all():
        raise ValueError("Nonfinite distance prediction")
    error = pred - gap
    far = gap > 10
    prediction_far = pred > 10
    close = gap < 3
    report = {
        "overall": error_stats(error),
        "0_to_6_m": error_stats(error[gap < 6]),
        "3_to_8_m": error_stats(error[(gap >= 3) & (gap < 8)]),
        "8_to_12_m": error_stats(error[(gap >= 8) & (gap <= 12)]),
        "above_10_m": error_stats(error[far]),
        "bins": {name: error_stats(error[mask]) for name, mask in gap_bins(gap).items()},
        "true_above_10_m": int(far.sum()),
        "correctly_predicted_above_10_m": int((far & prediction_far).sum()),
        "false_above_10_m": int((~far & prediction_far).sum()),
        "close_frames_predicted_at_least_3_m": int((close & (pred >= 3)).sum()),
        "close_frames_predicted_at_least_6_m": int((close & (pred >= SEVERE_CLOSE_PREDICTION_M)).sum()),
        "close_frames_overestimated_by_more_than_1_m": int((close & (error > 1)).sum()),
        "negative_predictions": int((pred < 0).sum()),
    }
    if groups is not None:
        group_indices = {}
        for i, group in enumerate(groups):
            group_indices.setdefault(group, []).append(i)
        report["by_route_group"] = {
            group: error_stats(error[indices]) for group, indices in group_indices.items()
        }
    if sources is not None:
        report["by_source"] = {}
        for source in sorted(set(sources)):
            mask = torch.tensor([tag == source for tag in sources])
            report["by_source"][source] = metrics(pred[mask], gap[mask])
    true_near = int((~far).sum())
    report["true_at_most_10_m"] = true_near
    report["recall_above_10_m"] = report["correctly_predicted_above_10_m"] / int(far.sum()) if far.any() else None
    report["false_positive_rate_above_10_m"] = report["false_above_10_m"] / true_near if true_near else None
    return report


def selection_score(pred, gap):
    if not torch.isfinite(pred).all():
        raise ValueError("Nonfinite validation prediction")
    error = (pred - gap).abs()
    # Use one score across bands to prevent plentiful midrange frames dominating
    # checkpoint selection and emphasize the onset of the 10 m cutoff.
    bands = ((gap < 8, 1), ((gap >= 8) & (gap <= 10), 2), (gap > 10, 2))
    scores = [(float(error[mask].mean()), weight) for mask, weight in bands if mask.any()]
    return sum(value * weight for value, weight in scores) / sum(weight for _, weight in scores)


def close_validation_bands(samples):
    masks = supervision_bands(samples.gaps)
    standard = torch.tensor([source == "standard" for source in samples.sources])
    bands = {
        "close_0_to_3_m": masks["close_0_to_3_m"],
        "standard_3_to_8_m": standard & masks["middle_3_to_8_m"],
        "standard_8_to_12_m": standard & masks["long_8_to_12_m"],
    }
    missing = [key for key, mask in bands.items() if not mask.any()]
    if missing:
        raise ValueError("Close-aware selection is missing validation frames for " + ", ".join(missing))
    return bands


def validation_score(pred, gap, close_bands):
    if close_bands is None:
        return selection_score(pred, gap), {}
    if not torch.isfinite(pred).all():
        raise ValueError("Nonfinite validation prediction")
    error = (pred - gap).abs()
    band_mae = {name: float(error[mask].mean()) for name, mask in close_bands.items()}
    return sum(band_mae.values()) / len(band_mae), band_mae


class Inference:
    def __init__(self, device):
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def __call__(self, network, frames, augment=False):
        images = frames.permute(0, 3, 1, 2).to(self.device, dtype=torch.float32) / 255.0
        if augment:
            flip = torch.rand(len(images), 1, 1, 1, device=self.device) < 0.5
            images = torch.where(flip, images.flip(-1), images)
            gain = torch.empty(len(images), 1, 1, 1, device=self.device).uniform_(0.9, 1.1)
            images = (images * gain).clamp(0, 1)
        return network((images - self.mean) / self.std).flatten()

    @torch.inference_mode()
    def evaluate(self, network, images, batch_size):
        network.eval()
        return torch.cat([self(network, batch).cpu() for batch in images.split(batch_size)])


@torch.inference_mode()
def evaluate_legacy(inference, network, root, distances, episode_ids, batch_size):
    network.eval()
    predictions, targets = [], []
    for images, gaps in episode_batches(root, distances, episode_ids, batch_size, shuffle=False):
        predictions.append(inference(network, images).cpu())
        targets.append(gaps)
    return metrics(torch.cat(predictions), torch.cat(targets))


def save_best(network, checkpoint_path, epoch, score):
    temporary = checkpoint_path.with_suffix(".pth.tmp")
    torch.save({"model": network.state_dict(), "epoch": epoch, "validation_score_m": score}, temporary)
    temporary.replace(checkpoint_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--legacy-data", type=Path,
                        help="Optional legacy archive for small training-only replay and final legacy-test comparison.")
    parser.add_argument("--legacy-reference", type=Path,
                        help="Original checkpoint defining legacy train and test IDs.")
    parser.add_argument("--replay-frames", type=int, default=384)
    parser.add_argument("--close-aware", action="store_true",
                        help="Balance close gaps and select checkpoints with validation regression limits.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.replay_frames < 0:
        parser.error("Epochs and batch size must be positive, and replay count must be nonnegative")

    if args.legacy_reference and not args.legacy_data:
        parser.error("--legacy-reference requires --legacy-data")

    torch.manual_seed(args.seed)
    order = torch.Generator().manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference = Inference(device)
    dataset, manifest = load_manifest(args.manifest)
    train, val, test = (dataset[split] for split in SPLITS)
    if args.close_aware and not (train.gaps < 3).any():
        raise ValueError("Close-aware training requires synchronized training frames below 3 m")
    close_bands = close_validation_bands(val) if args.close_aware else None
    snapshot = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    network = resnet18(weights=None)
    network.fc = nn.Linear(network.fc.in_features, 1)
    network.load_state_dict(snapshot["model"])
    network.to(device)
    replay_count = 0
    legacy_distances = None
    legacy_reference = snapshot
    if args.legacy_reference:
        legacy_reference = torch.load(args.legacy_reference, map_location="cpu", weights_only=True)
    if args.legacy_data:
        train, legacy_distances, replay_count = add_legacy_replay(
            train, args.legacy_data, legacy_reference, order, args.replay_frames
        )

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "resolved_split.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name, data in dataset.items():
        print(f"{name}: {data.episodes} episodes, {len(set(data.groups))} route groups, {len(data.gaps)} frames")
    print(f"Device: {device}. Legacy replay: {replay_count} frames.")

    baseline_val = inference.evaluate(network, val.images, args.batch_size)
    baseline_val_score, baseline_bands = validation_score(baseline_val, val.gaps, close_bands)
    print(f"Starting checkpoint validation score: {baseline_val_score:.3f} m")
    gate_limits = {}
    if args.close_aware:
        gate_limits = {name: (min(value * 0.90, CLOSE_MAX_MAE_M)
                              if name == "close_0_to_3_m" else value + 0.10)
                       for name, value in baseline_bands.items()}
        for name, limit in gate_limits.items():
            print(f"Validation gate {name}: baseline {baseline_bands[name]:.3f}, max MAE {limit:.3f} m")


    # Leave running BatchNorm statistics fixed on the small new collections.
    # Gradients still flow through the trainable final residual block and head.
    network.requires_grad_(False)
    network.layer4.requires_grad_(True)
    network.fc.requires_grad_(True)
    optimizer = torch.optim.AdamW([
        {"params": network.layer4.parameters(), "lr": 3e-5},
        {"params": network.fc.parameters(), "lr": 5e-4},
    ])
    loss_fn = nn.HuberLoss()
    weights = sampling_weights(train.gaps, close_aware=args.close_aware)
    training_history = []
    eligible_epochs = []
    best_score = diagnostic_score = float("inf")
    best_path = args.output / "best.pth"
    diagnostic_path = args.output / "best_diagnostic.pth"
    for epoch in range(1, args.epochs + 1):
        indices = torch.multinomial(weights, len(weights), replacement=True, generator=order)
        total = 0.0
        for batch in indices.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            pred = inference(network, train.images[batch], augment=True)
            loss = loss_fn(pred, train.gaps[batch].to(device))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)

        predictions = inference.evaluate(network, val.images, args.batch_size)
        score, band_mae = validation_score(predictions, val.gaps, close_bands)
        failures = [name for name, value in band_mae.items() if value > gate_limits[name]]
        severe_false_far = 0
        if args.close_aware:
            severe_false_far = int((close_bands["close_0_to_3_m"] &
                                    (predictions >= SEVERE_CLOSE_PREDICTION_M)).sum())
            if severe_false_far:
                failures.append("severe_false_far")
        val_mae = float((predictions - val.gaps).abs().mean())
        row = {"epoch": epoch, "train_huber": total / len(indices),
               "validation_score_m": score, "validation_mae_m": val_mae}
        gate_note = ""
        if args.close_aware:
            row.update({f"validation_{name}_mae": value for name, value in band_mae.items()})
            row["validation_severe_false_far"] = severe_false_far
            row["validation_failed_gates"] = ", ".join(failures)
            gate_note = f", failed gates {', '.join(failures)}" if failures else ", eligible"
            if not eligible_epochs and failures and score < diagnostic_score:
                diagnostic_score = score
                save_best(network, diagnostic_path, epoch, score)
        training_history.append(row)
        print(f"Epoch {epoch:02d}/{args.epochs}: loss {row['train_huber']:.3f}, "
              f"validation MAE {val_mae:.3f} m, score {score:.3f} m{gate_note}", flush=True)
        if not failures:
            eligible_epochs.append(epoch)
            if score < best_score:
                best_score = score
                save_best(network, best_path, epoch, score)
    with (args.output / "epochs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=training_history[0].keys())
        writer.writeheader()
        writer.writerows(training_history)

    # Evaluate the original and selected checkpoints only after selection.
    if eligible_epochs:
        candidate_path = best_path
    else:
        candidate_path = diagnostic_path
        print(f"No epoch met the validation gates. Retaining {args.checkpoint} as the selected model.")
    best = torch.load(candidate_path, map_location="cpu", weights_only=True)
    reports = {}
    test_predictions = {}
    for name, checkpoint in (("original", snapshot), ("finetuned", best)):
        network.load_state_dict(checkpoint["model"])
        val_pred = inference.evaluate(network, val.images, args.batch_size)
        test_pred = inference.evaluate(network, test.images, args.batch_size)
        test_predictions[name] = test_pred
        reports[name] = {"validation": metrics(val_pred, val.gaps, val.groups, val.sources),
                         "test": metrics(test_pred, test.gaps, test.groups, test.sources)}
        if args.legacy_data:
            reports[name]["legacy_test"] = evaluate_legacy(
                inference, network, args.legacy_data, legacy_distances, legacy_reference["test_ids"], args.batch_size
            )
    torch.save({"true_gap_m": test.gaps, "route_group": test.groups, "source": test.sources, **test_predictions},
               args.output / "test_predictions.pth")
    summary = {
        "initial_checkpoint": str(args.checkpoint.resolve()),
        "seed": args.seed,
        "best_epoch": best["epoch"],
        "selected_epoch": best["epoch"] if eligible_epochs else None,
        "selection_status": "eligible_checkpoint" if eligible_epochs else "no_eligible_checkpoint",
        "selected_checkpoint": str((best_path if eligible_epochs else args.checkpoint).resolve()),
        "evaluated_candidate_checkpoint": str(candidate_path.resolve()),
        "synchronized_split_frames": {name: len(data.gaps) for name, data in dataset.items()},
        "legacy_replay_frames": replay_count,
        "validation_selection": "Weighted mean of MAE below 8, 8–10 and above 10 m (1, 2, 2).",
        "original_validation_score_m": baseline_val_score,
        "finetuned_validation_score_m": best["validation_score_m"],
        **reports,
    }
    if args.close_aware:
        summary.update({
            "validation_selection": "Mean MAE across synchronized 0–3 m and standard 3–8 m and 8–12 m bands.",
            "validation_baseline_band_mae_m": baseline_bands,
            "validation_gate_limits_mae_m": gate_limits,
            "validation_gate_policy": "Require 0–3 m MAE at most 0.60 m and at least 10% better "
                                      "than the starting model, zero predictions at or above 6 m "
                                      "for true gaps below 3 m, and at most 0.10 m extra MAE in "
                                      "each protected standard band.",
            "validation_severe_false_far": {
                "true_gap_below_m": 3.0,
                "predicted_gap_at_least_m": SEVERE_CLOSE_PREDICTION_M,
                "allowed_count": 0,
            },
            "validation_eligible_epochs": eligible_epochs,
            "sampling_band_weights": {"0–3 m": 0.25, "3–8 m": 0.35, "8–12 m": 0.35, ">12 m": 0.05},
        })
    if args.legacy_data:
        summary["legacy_reference"] = str((args.legacy_reference or args.checkpoint).resolve())
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for name in reports:
        report = reports[name]["test"]
        print(f"{name}: test MAE {report['overall']['mae_m']:.3f} m | "
              f"above-10 recall {report['correctly_predicted_above_10_m']}/{report['true_above_10_m']}")
    label = "Selected" if eligible_epochs else "Evaluated diagnostic"
    print(f"{label} epoch: {best['epoch']}. Saved checkpoint: {candidate_path}")


if __name__ == "__main__":
    main()
