"""Join synchronized standard and close captures for a single specialist."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standard", type=Path, required=True)
    parser.add_argument("--close", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    args = parser.parse_args()
    merged = {split: [] for split in ("train", "validation", "test")}
    route_owners = {}
    for source, path in (("standard", args.standard), ("close", args.close)):
        manifest = json.loads(path.read_text())
        plan = json.loads((path.parent / "collection_plan.json").read_text())
        weather = plan["weather"]
        if weather != args.condition:
            raise ValueError(f"Expected {args.condition} captures, got {weather} in {path}")
        episodes = manifest["episodes"]
        if not manifest["complete"] or len(episodes) != manifest["planned_episodes"]:
            raise ValueError(f"Incomplete synchronized captures: {path}")
        for episode in episodes:
            route, split = episode["scenario"], episode["split"]
            if route_owners.setdefault(route, split) != split:
                raise ValueError(f"Route {route} occurs in multiple splits")
            archive = path.parent / episode["path"]
            if not archive.is_file():
                raise FileNotFoundError(archive)
            merged[split].append({
                "path": os.path.relpath(archive, args.output.parent),
                "route_group": route,
                "source": source,
            })
    for split, episodes in merged.items():
        print(f"{split}: {len(episodes)} synchronized episode files")
    args.output.write_text(json.dumps(merged, indent=2) + "\n")


if __name__ == "__main__":
    main()
