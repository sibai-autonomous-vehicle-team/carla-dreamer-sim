"""Build an episode-indexed numerical cache from an archive or extracted conditions."""
import argparse
import gzip
import hashlib
import io
import json
import os
import tarfile
import tempfile
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path, PurePosixPath
from zipfile import is_zipfile

import torch

from .config import LOCAL_DIR, Settings, load_data
from .dataset import assign_splits, episode_slices, first_failure, safety_margin, start_groups, transition_window


WIDTHS = {"states": 12, "actions": 2, "distances": None, "costs": None}
FILENAMES = {name + ".pth" for name in (*WIDTHS, "seq_lengths")}


def read_tensor_file(file):
    # Dataset files use the ZIP-based torch.save format. Never unpickle a module.
    if not is_zipfile(file):
        raise ValueError("Expected a ZIP-based PyTorch tensor file")
    file.seek(0)
    return torch.load(file, map_location="cpu", weights_only=True)


def read_conditions(source):
    groups = {}

    def add(condition, name, file):
        fields = groups.setdefault(condition, {})
        if name in fields:
            raise ValueError(f"Duplicate {condition}/{name}")
        fields[name] = read_tensor_file(file)

    if source.is_dir():
        for lengths_path in sorted(source.rglob("seq_lengths.pth")):
            folder = lengths_path.parent
            for name in sorted(FILENAMES):
                with (folder / name).open("rb") as file:
                    add(folder.name, Path(name).stem, file)
    else:
        with source.open("rb") as raw:
            compressed = raw.read(2) == b"\x1f\x8b"
            raw.seek(0)
            stream = gzip.GzipFile(fileobj=raw) if compressed else raw
            with stream, tarfile.open(fileobj=stream, mode="r|", bufsize=1024**2) as tar:
                for member in tar:
                    path = PurePosixPath(member.name)
                    if member.isfile() and path.name in FILENAMES:
                        group = path.parent.name
                        if group not in groups:
                            print(f"Reading {group}", flush=True)
                        # Deserialize just this numerical member. Camera payloads pass
                        # through the decompressor without being extracted or decoded.
                        with tar.extractfile(member) as file:
                            add(group, path.stem, io.BytesIO(file.read()))
    if not groups:
        raise ValueError(f"No numerical episode files under {source}")
    return groups


def pack_episodes(groups):
    columns = {field: [] for field in WIDTHS}
    lengths, conditions, episode_ids = [], [], []
    for condition, group in sorted(groups.items()):
        if set(group) != {*WIDTHS, "seq_lengths"}:
            raise ValueError(f"Incomplete condition {condition}: {sorted(group)}")
        sizes = group["seq_lengths"]
        if (not isinstance(sizes, torch.Tensor) or sizes.ndim != 1 or
                sizes.dtype != torch.int64 or not len(sizes) or (sizes <= 0).any()):
            raise ValueError(f"Invalid episode lengths for {condition}")
        for name, width in WIDTHS.items():
            episodes = group[name]
            if not isinstance(episodes, (tuple, list)) or len(episodes) != len(sizes):
                raise ValueError(f"Episode count mismatch for {condition}/{name}")
            for index, (tensor, length) in enumerate(zip(episodes, sizes.tolist())):
                shape = (length,) if width is None else (length, width)
                if not isinstance(tensor, torch.Tensor) or tensor.shape != shape or tensor.dtype != torch.float32:
                    raise ValueError(f"Invalid {condition}/{name} episode {index}, expected float32 {shape}")
            combined = torch.cat(episodes)
            if not torch.isfinite(combined).all():
                raise ValueError(f"Nonfinite {condition}/{name}")
            if name == "distances" and (combined < 0).any():
                raise ValueError(f"Negative unsigned distance in {condition}")
            if name == "costs" and not ((combined == 0) | (combined == 1)).all():
                raise ValueError(f"Nonbinary collision reading in {condition}")
            columns[name].append(combined)
        lengths.append(sizes)
        conditions.extend([condition] * len(sizes))
        episode_ids.extend(range(len(sizes)))
    return {**{k: torch.cat(v) for k, v in columns.items()}, "lengths": torch.cat(lengths),
            "conditions": conditions, "episode_ids": torch.tensor(episode_ids), "format_version": 1}


def combined_cache(old, new, cfg):
    """Give both training populations the same fixed evaluation scenarios."""
    arrays = defaultdict(list)
    sources, conditions, local_ids = [], [], []
    coverage = {}
    for source, packed in (("old", old), ("new", new)):
        coverage[source] = dict(sorted(Counter(packed["conditions"]).items()))
        missing = set(cfg.shared_conditions) - set(coverage[source])
        if missing:
            raise ValueError(f"{source} is missing controlled-comparison conditions: {sorted(missing)}")
        ids = [i for i, condition in enumerate(packed["conditions"]) if condition in cfg.shared_conditions]
        mask = torch.zeros(len(packed["lengths"]), dtype=torch.bool)
        mask[ids] = True
        rows = torch.repeat_interleave(mask, packed["lengths"])
        for field in WIDTHS:
            arrays[field].append(packed[field][rows])
        arrays["lengths"].append(packed["lengths"][ids])
        local_ids.append(packed["episode_ids"][ids])
        sources.extend([source] * len(ids))
        conditions.extend(packed["conditions"][i] for i in ids)
    data = {key: torch.cat(parts) for key, parts in arrays.items()}
    data.update(format_version=1, sources=sources, conditions=conditions, episode_ids=torch.cat(local_ids))
    fingerprint = hashlib.sha256(json.dumps([sources, conditions], separators=(",", ":")).encode())
    for key in (*WIDTHS, "lengths", "episode_ids"):
        tensor = data[key].contiguous()
        fingerprint.update(f"{key}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        fingerprint.update(memoryview(tensor.numpy()).cast("B"))
    data["dataset_id"] = fingerprint.hexdigest()
    groups = start_groups(data, cfg.start_tolerance_m)
    splits = assign_splits(groups, cfg.train_fraction, cfg.val_fraction, cfg.split_seed)
    split_settings = dict(seed=cfg.split_seed, start_tolerance_m=cfg.start_tolerance_m,
                         train_fraction=cfg.train_fraction, val_fraction=cfg.val_fraction)
    assignment = json.dumps([data["dataset_id"], groups, splits, split_settings], sort_keys=True).encode()
    data.update(splits=splits, split_id=hashlib.sha256(assignment).hexdigest(),
                group_ids=groups, split_settings=split_settings, input_coverage=coverage)
    return data


def transition_counts(data, rows, cfg):
    gaps, costs, actions = (data[key][rows] for key in ("distances", "costs", "actions"))
    length = len(gaps)
    margin = safety_margin(gaps, cfg.safety_buffer_m, cfg.max_gap_m)
    hit = first_failure(margin)
    start, end = transition_window(length, hit, cfg)
    _, available_end = transition_window(length, None, cfg)
    eligible = max(0, end - start)
    before_failure_cut = max(0, available_end - start)
    current = gaps[start + cfg.gap_offset : start + cfg.gap_offset + eligible]
    chosen = actions[start + cfg.action_offset : start + cfg.action_offset + eligible]
    first_near = hit is not None and gaps[hit].item() <= cfg.safety_buffer_m
    first_far = hit is not None and gaps[hit].item() >= cfg.max_gap_m
    enters = hit is not None and eligible > 0 and end + cfg.gap_offset == hit
    omitted_initial = min(length, start + cfg.action_offset)
    unknown_end = hit is None
    return Counter(
        episodes=1, rows=length, eligible_transitions=eligible,
        raw_brake_commands=int((actions[:, 0] < 0).sum()), eligible_brake_commands=int((chosen[:, 0] < 0).sum()),
        transitions_start_near_lower=int((current <= cfg.safety_buffer_m + cfg.boundary_band_m).sum()),
        transitions_start_near_upper=int((current >= cfg.max_gap_m - cfg.boundary_band_m).sum()),
        first_near_failures=int(first_near), first_far_failures=int(first_far),
        eligible_first_near_successors=int(first_near and enters), eligible_first_far_successors=int(first_far and enters),
        first_failures_without_trainable_entry=int(hit is not None and not enters),
        post_hit_aligned_rows_excluded=before_failure_cut - eligible,
        rows_missing_initial_alignment_or_history=omitted_initial,
        rows_missing_final_alignment_or_successor=length - omitted_initial - before_failure_cut,
        unusable_rows=length - eligible, known_truncations=0,
        unknown_ends_without_buffer_failure=int(unknown_end), unknown_recording_end_reasons=1,
        any_collision_sensor_episodes=int((costs > 0).any()),
        any_lower_violation_episodes=int((gaps <= cfg.safety_buffer_m).any()),
        any_upper_violation_episodes=int((gaps >= cfg.max_gap_m).any()),
        later_return_inside_after_first_hit=int(hit is not None and (margin[hit + 1:] > 0).any()),
        later_return_inside_no_collision=int(hit is not None and (margin[hit + 1:] > 0).any() and not costs.any()),
    )


def write_comparison_reports(data, output, cfg):
    split_for = {i: split for split, ids in data["splits"].items() for i in ids}
    records, totals = [], defaultdict(Counter)
    for i, rows in enumerate(episode_slices(data)):
        source, condition, split = data["sources"][i], data["conditions"][i], split_for[i]
        records.append(dict(source=source, condition=condition, episode=int(data["episode_ids"][i]),
                            group=data["group_ids"][i], split=split))
        counts = transition_counts(data, rows, cfg)
        for key in ("all", source, f"{source}/{condition}", f"{source}/{split}",
                    f"{source}/{condition}/{split}", f"all/{split}"):
            totals[key].update(counts)
    identifiers = dict(dataset_id=data["dataset_id"], split_id=data["split_id"])
    manifest = dict(**identifiers, settings=data["split_settings"], method=(
        "Transitive closure of first-row planar ego and leader position matches, both within "
        "the stated tolerance, across conditions and archives. Identifiers and seeds are unavailable. "
        "Similar geometry can group distinct simulator scenarios."), assignments=records)
    stats = dict(**identifiers, input_coverage=data["input_coverage"],
                 conditions=list(cfg.shared_conditions),
                 failure_bounds_m=[cfg.safety_buffer_m, cfg.max_gap_m], near_boundary_band_m=cfg.boundary_band_m,
                 action_offset=cfg.action_offset, gap_offset=cfg.gap_offset, dt=cfg.dt,
                 definitions="First failure means measured h <= 0. Near-boundary counts are eligible "
                 "current states inside the band. Collision readings are independent. End reasons are "
                 "unknown even when a distance violation was observed. All row exclusions are disjoint: "
                 "missing initial alignment/history, missing final alignment/successor, or aligned rows "
                 "at or after the first hit. Geometric returns inside the range are diagnostic only.",
                 counts=totals)
    for name, report in (("manifest", manifest), ("stats", stats)):
        output.with_suffix(f".{name}.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--new-source", type=Path, help="Add the braking-enabled data and freeze a grouped episode split")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true", help="replace an existing cache")
    parser.add_argument("--conditions", nargs="+", choices=Settings().shared_conditions,
                        help="Conditions to include; omit for all nine. Use one for a condition-specific expert.")
    args = parser.parse_args()
    if args.conditions and not args.new_source:
        parser.error("--conditions requires --new-source")
    if args.output is None:
        args.output = LOCAL_DIR / ("hj_nine.pt" if args.new_source else "numerical.pt")
    if any(p is not None and p.resolve() == args.output.resolve() for p in (args.source, args.new_source)):
        parser.error("Output must differ from each source")
    if args.output.exists() and not args.force:
        parser.exit(0, f"Cache exists: {args.output}. Use --force to rebuild.\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    if args.new_source:
        old = load_data(args.source) if args.source.suffix == ".pt" else pack_episodes(read_conditions(args.source))
        new = pack_episodes(read_conditions(args.new_source))
        cfg = replace(Settings(), shared_conditions=tuple(args.conditions)) if args.conditions else Settings()
        data = combined_cache(old, new, cfg)
        data["source_paths"] = {"old": str(args.source.resolve()), "new": str(args.new_source.resolve())}
        write_comparison_reports(data, args.output, cfg)
    else:
        data = pack_episodes(read_conditions(args.source))
        data["source"] = str(args.source.resolve())
    with tempfile.NamedTemporaryFile(dir=args.output.parent, suffix=".tmp", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(data, tmp_path)
        os.replace(tmp_path, args.output)
    finally:
        tmp_path.unlink(missing_ok=True)
    print(f"Saved {len(data['lengths']):,} episodes, {len(data['states']):,} rows to {args.output}")


if __name__ == "__main__":
    main()
