"""Combine the synchronized standard and close runs without moving captures."""
import json
import os
from pathlib import Path

output = Path(__file__).resolve().parent
sources = {
    "standard": output.parent / "day_cloudy_cnn_v2_2026-11-17" / "captures" / "split.json",
    "close": output / "close_captures" / "split.json",
}
groups = ("train", "validation", "test")
merged = {split: [] for split in groups}
route_owners = {}
for source, path in sources.items():
    manifest = json.loads(path.read_text())
    if not manifest["complete"] or len(manifest["episodes"]) != manifest["planned_episodes"]:
        raise ValueError(f"Incomplete synchronized capture manifest: {path}")
    for split in groups:
        for episode in (row for row in manifest["episodes"] if row["split"] == split):
            route = episode["scenario"]
            if route_owners.setdefault(route, split) != split:
                raise ValueError(f"Route {route} occurs in multiple splits")
            archive = path.parent / episode["path"]
            if not archive.is_file():
                raise FileNotFoundError(archive)
            merged[split].append({
                "path": os.path.relpath(archive, output),
                "route_group": route,
                "source": source,
            })
for split in groups:
    episodes = len(merged[split])
    print(f"{split}: {episodes} synchronized episode files")
(output / "merged_split.json").write_text(json.dumps(merged, indent=2) + "\n")
