"""Build the 1,300-episode specialist datasets without copying RGB tensors."""

import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import torch

from analysis.verify_distance_cnn_archive import episode_files, require
from train_distance_cnn import CONDITIONS


EXISTING = Path("dataset/combined_800")
BRAKING = Path("dataset/braking_lane_tracked_500_per_condition")
EXISTING_SPLITS = Path("experiments/generalist_all_conditions_2026-11-18/training/splits")
OUTPUT = Path("experiments/specialists_braking_1300_2026-11-23/data")
PARTS = ("train_ids", "val_ids", "test_ids")
EXISTING_COUNT = 800
BRAKING_COUNT = 500
SEED = 42


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def load(path, *, mmap=False):
    return torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)


def read_source(folder, count):
    images = episode_files(folder, count)
    distances = load(folder / "distances.pth")
    lengths = load(folder / "seq_lengths.pth")
    require(isinstance(distances, list) and len(distances) == count,
            f"Invalid distances in {folder}")
    require(isinstance(lengths, torch.Tensor) and lengths.dtype == torch.int64
            and lengths.shape == (count,) and bool((lengths > 0).all()),
            f"Invalid sequence lengths in {folder}")
    for image, gaps, length in zip(images, distances, lengths.tolist()):
        require(isinstance(gaps, torch.Tensor) and gaps.dtype == torch.float32
                and gaps.shape == (length,) and bool(torch.isfinite(gaps).all())
                and bool((gaps >= 0).all()), f"Invalid distances for {image}")
        frames = load(image, mmap=True)
        require(isinstance(frames, torch.Tensor) and frames.dtype == torch.uint8
                and frames.shape == (length, 224, 224, 3), f"Invalid RGB sequence in {image}")
    return images, distances, lengths


def braking_routes(folder, lengths):
    records, groups = [], defaultdict(list)
    for episode, steps in enumerate(lengths.tolist()):
        path = folder / "metadata" / f"episode_{episode:03d}.pth"
        record = load(path, mmap=True)
        plan = record["plan"]
        scenario = plan["scenario"]
        require(record["condition_episode"] == episode and plan["condition"] == folder.name,
                f"Metadata episode identity differs in {path}")
        frame_ids = record["frames"]
        require(frame_ids.shape == (steps + 1,) and bool((frame_ids.diff() == 1).all()),
                f"Nonconsecutive capture frames in {path}")
        route = scenario["full_route"]
        require(bool(route), f"Missing persisted route in {path}")
        route_hash = hashlib.sha256(json.dumps(route, separators=(",", ":")).encode()).hexdigest()
        groups[route_hash].append(episode + EXISTING_COUNT)
        records.append({"scenario_seed": scenario["seed"], "plan_episode": plan["episode"],
                        "exact_route_sha256": route_hash, "ending": record["ending"]})
    return records, groups


def split_braking(groups):
    generator = random.Random(SEED)
    route_groups = list(groups.values())
    generator.shuffle(route_groups)
    # Pack repeated routes together first, leaving singletons to fill exact counts.
    route_groups.sort(key=len, reverse=True)
    split = {"seed": SEED, **{key: [] for key in PARTS}}
    remaining = dict(zip(PARTS, (400, 50, 50)))
    for group in route_groups:
        fits = [key for key in PARTS if remaining[key] >= len(group)]
        require(bool(fits), "Repeated braking route cannot fit the requested split sizes")
        part = generator.choices(fits, weights=[remaining[key] for key in fits], k=1)[0]
        split[part].extend(group)
        remaining[part] -= len(group)
    require(not any(remaining.values()), "Braking splits do not fill the requested sizes")
    return split


def validate_split(split, ids, counts):
    require(split["seed"] == SEED, "Unexpected split seed")
    require(tuple(len(split[key]) for key in PARTS) == counts, "Unexpected split sizes")
    require(sorted(episode for key in PARTS for episode in split[key]) == list(ids),
            "Split has overlapping or missing episodes")


def verify_view(destination, images, distances, lengths, splits, records):
    observations = destination / "obses"
    require(len(list(observations.glob("*.pth"))) == len(images),
            f"Unexpected linked image count in {destination}")
    for episode, source in enumerate(images):
        link = observations / f"episode_{episode:03d}.pth"
        require(link.is_symlink() and link.resolve(strict=True) == source.resolve(strict=True),
                f"Linked image differs from source at {link}")
    require(torch.equal(load(destination / "seq_lengths.pth"), lengths),
            f"Stored sequence lengths differ in {destination}")
    copy = load(destination / "distances.pth")
    require(len(copy) == len(distances) and all(torch.equal(a, b) for a, b in zip(copy, distances)),
            f"Stored distances differ in {destination}")
    for name, split in splits.items():
        require(json.loads((destination / name).read_text()) == split, f"Split differs in {destination / name}")
    require(json.loads((destination / "episodes.json").read_text()) == records,
            f"Episode provenance differs in {destination}")


def build_condition(output, condition):
    old_images, old_gaps, old_lengths = read_source(EXISTING / condition, EXISTING_COUNT)
    new_images, new_gaps, new_lengths = read_source(BRAKING / condition, BRAKING_COUNT)
    new_records, groups = braking_routes(BRAKING / condition, new_lengths)
    original = json.loads((EXISTING_SPLITS / f"{condition}.json").read_text())
    validate_split(original, range(EXISTING_COUNT), (640, 80, 80))
    braking = split_braking(groups)
    total = EXISTING_COUNT + BRAKING_COUNT
    validate_split(braking, range(EXISTING_COUNT, total), (400, 50, 50))
    merged = {"seed": SEED, **{key: original[key] + braking[key] for key in PARTS}}
    validate_split(merged, range(total), (1040, 130, 130))
    splits = {"split.json": merged, "split_original.json": original, "split_braking.json": braking}

    images = old_images + new_images
    distances = old_gaps + new_gaps
    lengths = torch.cat((old_lengths, new_lengths))
    partition = {episode: key.removesuffix("_ids") for key in PARTS for episode in merged[key]}
    records = []
    for episode, (image, steps) in enumerate(zip(images, lengths.tolist())):
        is_braking = episode >= EXISTING_COUNT
        source_id = episode - EXISTING_COUNT if is_braking else episode
        details = new_records[source_id] if is_braking else {}
        records.append({"episode": episode, "source": BRAKING.name if is_braking else EXISTING.name,
                        "source_episode": source_id, "source_image": str(image),
                        "split": partition[episode], "aligned_frames": steps - 1, **details})
    destination = output / condition
    if destination.exists():
        verify_view(destination, images, distances, lengths, splits, records)
    else:
        with tempfile.TemporaryDirectory(prefix=f".{condition}.", dir=output) as temporary:
            stage = Path(temporary) / condition
            observations = stage / "obses"
            observations.mkdir(parents=True)
            for episode, source in enumerate(images):
                link = observations / f"episode_{episode:03d}.pth"
                link.symlink_to(os.path.relpath(source, destination / "obses"))
            torch.save(distances, stage / "distances.pth")
            torch.save(lengths, stage / "seq_lengths.pth")
            for name, split in splits.items():
                write_json(stage / name, split)
            write_json(stage / "episodes.json", records)
            # Relative links are constructed for the final location and checked there.
            stage.rename(destination)
        verify_view(destination, images, distances, lengths, splits, records)

    counts = {key: len(merged[key]) for key in PARTS}
    ends = {part: dict(Counter(row["ending"] for row in records[EXISTING_COUNT:] if row["split"] == part))
            for part in ("train", "val", "test")}
    print(f"{condition}: 1,300 episodes, 1,040/130/130 split, {len(groups)} distinct braking routes",
          flush=True)
    return {"episodes": total, "split_counts": counts, "aligned_frames": int(lengths.sum()) - total,
            "episodes_without_aligned_frames": [row["episode"] for row in records if not row["aligned_frames"]],
            "distinct_exact_braking_routes": len(groups),
            "repeated_exact_braking_routes": sum(len(group) > 1 for group in groups.values()),
            "braking_endings_by_split": ends}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    manifest = json.loads((BRAKING / "manifest.json").read_text())
    require(manifest["dataset_layout"] == "original_condition_files"
            and "T pre-action RGB" in manifest["observations"]
            and "T post-action distance" in manifest["observations"],
            "Braking source does not use the existing trainer's alignment convention")
    complete = json.loads((BRAKING / "complete.json").read_text())
    require(complete["episodes_per_condition"] == {condition: BRAKING_COUNT for condition in CONDITIONS},
            "Braking collection is incomplete")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = {"schema_version": 1, "seed": SEED, "existing_source": str(EXISTING),
                  "existing_split_dir": str(EXISTING_SPLITS), "braking_source": str(BRAKING),
                  "episode_order": "Original 0–799, braking 800–1299",
                  "alignment": "RGB[1:] with distances[:-1], both sources",
                  "braking_route_grouping": "Identical persisted directed routes are held together per condition",
                  "conditions": {condition: build_condition(args.output, condition) for condition in CONDITIONS}}
        pending = args.output / ".merge_manifest.json.tmp"
        write_json(pending, result)
        pending.replace(args.output / "merge_manifest.json")


if __name__ == "__main__":
    main()
