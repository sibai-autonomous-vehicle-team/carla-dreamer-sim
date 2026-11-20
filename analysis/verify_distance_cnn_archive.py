"""Verify one merged CARLA archive against the documented Linux sources.

Run from the repository root. Checksum paths are relative to that directory.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch


DATASET = Path("dataset")
MERGE_FILE = DATASET / "combined_800" / "merge_manifest.json"
CONSOLIDATION_FILE = Path(".local/consolidated-following-sources.json")
RECORD_SHAPES = {"actions": (2,), "costs": (), "distances": (), "states": (12,)}
NUMERIC_FIELDS = (*RECORD_SHAPES, "seq_lengths")
CHUNK_BYTES = 4 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def episode_files(directory, count):
    files = sorted((directory / "obses").glob("*.pth"))
    expected_names = [f"episode_{episode:03d}.pth" for episode in range(count)]
    require([file.name for file in files] == expected_names,
            f"Unexpected episode filenames or count in {directory}")
    return files


def read_arrays(directory, count):
    data = {
        field: torch.load(directory / f"{field}.pth", map_location="cpu", weights_only=True)
        for field in NUMERIC_FIELDS
    }
    lengths = data["seq_lengths"]
    require(isinstance(lengths, torch.Tensor) and lengths.dtype == torch.int64
            and lengths.shape == (count,) and (lengths > 0).all(),
            f"Invalid sequence lengths in {directory}")
    steps = lengths.tolist()
    for field, trailing_shape in RECORD_SHAPES.items():
        records = data[field]
        require(len(records) == count, f"Incorrect number of {field} entries in {directory}")
        for episode, (record, length) in enumerate(zip(records, steps)):
            require(isinstance(record, torch.Tensor) and record.dtype == torch.float32
                    and record.shape == (length, *trailing_shape) and torch.isfinite(record).all(),
                    f"Invalid {field} in {directory}, episode {episode:03d}")
    require(all((gap >= 0).all() for gap in data["distances"]), f"Negative distance in {directory}")
    return data


def compare_arrays(merged, sources):
    offset = 0
    for directory, source in sources:
        count = len(source["seq_lengths"])
        for field in NUMERIC_FIELDS:
            section = merged[field][offset:offset + count]
            matches = (torch.equal(section, source[field]) if field == "seq_lengths"
                       else all(torch.equal(a, b) for a, b in zip(section, source[field])))
            require(matches, f"{field} differs from source {directory}")
        offset += count


def checksum(path, comparison=None):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        if comparison is None:
            for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
                digest.update(block)
        else:
            require(path.stat().st_size == comparison.stat().st_size,
                    f"File size differs between {path} and {comparison}")
            with comparison.open("rb") as expected:
                for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
                    require(block == expected.read(len(block)), f"Bytes differ in {path} and {comparison}")
                    digest.update(block)
                require(expected.read(1) == b"", f"Unexpected trailing bytes in {comparison}")
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, help="Condition name in the merge manifest")
    parser.add_argument("--output", required=True, type=Path, help="Directory for verification outputs")
    args = parser.parse_args()

    merge = json.loads(MERGE_FILE.read_text())
    if args.condition not in merge["conditions"]:
        parser.error(f"Unknown condition: {args.condition}")
    expected = merge["conditions"][args.condition]
    merged_directory = MERGE_FILE.parent / args.condition
    counts = [item["count_per_condition"] for item in merge["sources"]]
    source_directories = [DATASET / item["folder"] / args.condition for item in merge["sources"]]
    require(sum(counts) == expected["episodes"], "Source counts do not sum to the merged count")
    merged = read_arrays(merged_directory, expected["episodes"])
    sources = [(path, read_arrays(path, count)) for path, count in zip(source_directories, counts)]
    compare_arrays(merged, sources)

    merged_files = episode_files(merged_directory, expected["episodes"])
    source_files = [file for path, count in zip(source_directories, counts)
                    for file in episode_files(path, count)]
    hashes = {path: checksum(path) for path in (MERGE_FILE, CONSOLIDATION_FILE)}
    for field in NUMERIC_FIELDS:
        path = merged_directory / f"{field}.pth"
        hashes[path] = checksum(path)
    lengths = merged["seq_lengths"].tolist()
    require(sum(lengths) == expected["steps"], "Sequence lengths do not match manifest step count")

    for index, (path, reference, length) in enumerate(zip(merged_files, source_files, lengths), start=1):
        # Read tensor metadata without materializing the image arrays.
        frames = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        require(isinstance(frames, torch.Tensor) and frames.dtype == torch.uint8
                and frames.shape == (length, 224, 224, 3), f"Invalid RGB sequence in {path}")
        hashes[path] = checksum(path, reference)
        if index % 100 == 0 or index == len(merged_files):
            print(f"{args.condition} checked {index}/{len(merged_files)} image files", flush=True)

    aligned = torch.cat([gap[:-1] for gap in merged["distances"]])
    bands = {
        "0_to_3": aligned < 3,
        "3_to_6": (aligned >= 3) & (aligned < 6),
        "6_to_8": (aligned >= 6) & (aligned < 8),
        "8_to_10": (aligned >= 8) & (aligned <= 10),
        "above_10": aligned > 10,
    }
    verification = {
        "condition": args.condition, "archive": str(merged_directory),
        "episodes": expected["episodes"], "raw_steps": expected["steps"],
        "aligned_frames": len(aligned),
        "documented_sources": [
            {"path": str(path), "episodes": len(data["seq_lengths"]),
             "raw_steps": int(data["seq_lengths"].sum()),
             "aligned_frames": int(data["seq_lengths"].sum()) - len(data["seq_lengths"])}
            for path, data in sources
        ],
        "image_files_verified": len(merged_files), "image_byte_matches": len(merged_files),
        "image_shape": "episode length × 224 × 224 × 3", "image_dtype": "uint8",
        "numeric_tensors_finite_and_source_equal": list(NUMERIC_FIELDS),
        "aligned_gap_bin_counts": {band: int(mask.sum()) for band, mask in bands.items()},
        "checksum_entry_count": len(hashes),
        "checksum_verification_working_directory": "repository root",
        "mac_source_verified": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "dataset.sha256").write_text(
        "".join(f"{digest}  {path}\n" for path, digest in hashes.items()))
    (args.output / "dataset_verification.json").write_text(
        json.dumps(verification, indent=2) + "\n")
    print(f"{args.condition} verified {len(merged_files)} episodes and {len(aligned)} aligned frames",
          flush=True)


if __name__ == "__main__":
    main()
