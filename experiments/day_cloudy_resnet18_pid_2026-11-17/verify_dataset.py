"""Verify all archived episodes and record the input checksums."""
import hashlib
import json
from pathlib import Path

import torch

output = Path(__file__).resolve().parent
root = Path("dataset/combined_800/day_cloudy")
load = lambda path: torch.load(path, map_location="cpu", weights_only=True)
lengths = load(root / "seq_lengths.pth").tolist()
fields = {field: load(root / f"{field}.pth")
          for field in ("actions", "states", "costs", "distances")}
paths = sorted((root / "obses").glob("episode_*.pth"))
assert len(lengths) == len(paths) == 800
assert all(len(values) == len(lengths) for values in fields.values())
assert [p.name for p in paths] == [f"episode_{i:03d}.pth" for i in range(800)]
with (output / "dataset.sha256").open("w") as manifest:
    for path in sorted(root.glob("*.pth")) + paths:
        index = int(path.stem.split("_")[1]) if path.parent.name == "obses" else None
        if index is not None:
            n = lengths[index]
            frames = load(path)
            assert frames.shape == (n, 224, 224, 3) and frames.dtype == torch.uint8
            assert all(len(values[index]) == n and torch.isfinite(values[index]).all()
                       for values in fields.values())
            assert (fields["distances"][index] >= 0).all()
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(
                hashlib, "file_digest") else None
            if digest is None:
                hasher = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(block)
                digest = hasher.hexdigest()
        print(f"{digest}  {path}", file=manifest)
split = json.loads((output / "split.json").read_text())
parts = [split[label] for label in ("train_ids", "val_ids", "test_ids")]
assert sorted(sum(parts, [])) == list(range(800))
result = {
    "episodes": len(lengths), "raw_steps": sum(lengths),
    "aligned_frames": sum(n - 1 for n in lengths),
    "terminal_collision_labels": sum(bool(cost[-1]) for cost in fields["costs"]),
    "split_episodes": [len(ids) for ids in parts],
    "split_aligned_frames": [sum(lengths[i] - 1 for i in ids) for ids in parts],
}
(output / "dataset_verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(result)
