"""Freeze matched selector cases on audited and separated Town03 routes."""

import argparse
import hashlib
import json
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import carla
import numpy as np

from shared_follow_route import Scenario, SharedRoute
from train_distance_cnn import CONDITIONS


PARTITIONS = ("train", "validation", "test")
TOWN03_XODR = Path(".local/carla-0.9.15/CarlaUE4/Content/Carla/Maps/OpenDrive/Town03.xodr")
PREVIOUS_MANIFESTS = (
    Path("specialists/routes.json"),
    Path("experiments/day_cloudy_cnn_v2_2026-11-17/routes.json"),
    Path("experiments/sunset_cloudy_specialist_2026-11-17/routes.json"),
)
ARCHIVE = Path("dataset/combined_800")
ARCHIVE_SPLITS = Path("experiments/generalist_all_conditions_2026-11-18/training/splits")
MIN_ROUTE_SEPARATION_M = 20.0
ARCHIVE_NEARBY_M = 3.0
START_INDICES = (7, 10, 12)
LEAD_PROFILES = ("steady", "pull_away", "brake_restart")
SEED_BASES = {"train": 10_000, "validation": 20_000, "test": 30_000}

# These candidates were traced offline over the actual Town03 OpenDRIVE map.
# Each row gives route type, OpenDRIVE road/lane/s and explicit junction branches.
ROUTE_SPECS = {
    "train": (
        ("straight", 67, -2, 205.0, ()),
        ("bend", 67, 2, 185.0, ()),
        ("turn", 6, 5, 5.0, (1374,)),
        ("straight", 67, -1, 305.0, (100,)),
        ("bend", 76, 1, 135.0, ()),
        ("turn", 24, -1, 35.0, (130, 1020)),
    ),
    "validation": (
        ("straight", 6, -1, 25.0, ()),
        ("bend", 52, -1, 25.0, ()),
        ("turn", 62, -1, 15.0, (1861,)),
    ),
    "test": (
        ("straight", 30, 4, 45.0, ()),
        ("bend", 2, 2, 45.0, ()),
        ("turn", 17, 1, 5.0, (1713,)),
    ),
}


def make_scenarios():
    return {
        split: [
            Scenario(
                name=f"{split}_{index + 1:02d}_{kind}",
                road=road,
                lane=lane,
                start_s=start_s,
                length=60,
                finish_m=35.0,
                branch_roads=branches,
            )
            for index, (kind, road, lane, start_s, branches) in enumerate(specs)
        ]
        for split, specs in ROUTE_SPECS.items()
    }


def min_distance(first, second):
    """Minimum horizontal distance between two sampled centerlines."""
    return float(np.linalg.norm(first[:, None] - second[None, :], axis=2).min())


def audit_routes(road_map, scenarios):
    routes = {
        scenario.name: SharedRoute(road_map, scenario)
        for group in scenarios.values()
        for scenario in group
    }
    previous_routes = {}
    for manifest in PREVIOUS_MANIFESTS:
        for group in json.loads(manifest.read_text()).values():
            for spec in group:
                scenario = Scenario(**{**spec, "branch_roads": tuple(spec["branch_roads"])})
                previous_routes[scenario] = SharedRoute(road_map, scenario)
    previous_xy = np.concatenate([route.xy for route in previous_routes.values()])
    separation = min(
        min_distance(first.xy, second.xy)
        for first, second in combinations(routes.values(), 2)
    )
    if separation < MIN_ROUTE_SEPARATION_M:
        raise ValueError(f"Fresh route corridors too close ({separation:.2f} m)")
    records = {}
    for name, route in routes.items():
        previous = min_distance(route.xy, previous_xy)
        if previous < MIN_ROUTE_SEPARATION_M:
            raise ValueError(f"Route {name} too close to a previous route ({previous:.2f} m)")
        records[name] = {
            "centerline_xy": route.xy.tolist(),
            "length_m": float(route.cumulative[-1]),
            "distance_to_previous_manifest_m": previous,
        }
    audit = {
        "previous_manifests": [str(path) for path in PREVIOUS_MANIFESTS],
        "previous_route_count": len(previous_routes),
        "minimum_route_separation_m": separation,
        "minimum_separation_from_previous_manifests_m": min(
            row["distance_to_previous_manifest_m"] for row in records.values()
        ),
        "routes": records,
    }
    return routes, audit


def audit_archive_overlap(routes):
    """Check train pose overlap without interpreting an unseen route as unseen scenery."""
    import torch

    if any(
        not (ARCHIVE / condition / "states.pth").is_file()
        or not (ARCHIVE_SPLITS / (condition + ".json")).is_file()
        for condition in CONDITIONS
    ):
        return {"status": "unavailable", "archive": str(ARCHIVE)}
    pose_parts = []
    for condition in CONDITIONS:
        states = torch.load(
            ARCHIVE / condition / "states.pth", map_location="cpu", weights_only=True
        )
        split = json.loads((ARCHIVE_SPLITS / (condition + ".json")).read_text())
        pose_parts.extend(states[index][:, :2].numpy() for index in split["train_ids"])
    poses = np.concatenate(pose_parts)
    records = {}
    for name, route in routes.items():
        low = route.xy.min(axis=0) - ARCHIVE_NEARBY_M
        high = route.xy.max(axis=0) + ARCHIVE_NEARBY_M
        nearby = poses[np.all((poses >= low) & (poses <= high), axis=1)]
        within_radius = sum(
            bool(np.any(np.sum((nearby - point) ** 2, axis=1) <= ARCHIVE_NEARBY_M ** 2))
            for point in route.xy
        )
        records[name] = {
            "route_waypoint_count": len(route.xy),
            "waypoints_with_nearby_archived_ego_pose": within_radius,
        }
    return {
        "status": "checked",
        "archive": str(ARCHIVE),
        "archive_training_splits": str(ARCHIVE_SPLITS),
        "archived_training_ego_pose_count": len(poses),
        "nearby_radius_m": ARCHIVE_NEARBY_M,
        "routes": records,
    }


def make_cases(scenarios, variant):
    cases = []
    for split in PARTITIONS:
        if split == "test" and variant:
            continue
        for route_index, scenario in enumerate(scenarios[split]):
            for condition_index, condition in enumerate(CONDITIONS):
                gap_index = (condition_index // 3 + route_index + variant) % 3
                profile_index = (condition_index + route_index + route_index // 3 + variant) % 3
                seed = SEED_BASES[split] + variant * 1_000 + route_index * 20 + condition_index
                suffix = f"_v{variant}" if variant else ""
                cases.append({
                    "case_id": f"{scenario.name}_{condition}{suffix}",
                    "split": split,
                    "condition": condition,
                    "seed": seed,
                    "route": asdict(scenario),
                    "lead_start_index": START_INDICES[gap_index],
                    "lead_profile": LEAD_PROFILES[profile_index],
                })
    return cases


def choose_smoke_cases(cases):
    targets = (
        (10, "steady", "day_sunny", "straight"),
        (12, "pull_away", "sunset_cloudy", "turn"),
        (7, "brake_restart", "night_rainy", "bend"),
    )
    result = []
    for index, profile, condition, kind in targets:
        matches = [
            case for case in cases
            if case["split"] == "train" and case["lead_start_index"] == index
            and case["lead_profile"] == profile
        ]
        choice = min(
            matches,
            key=lambda case: (case["condition"] != condition, not case["route"]["name"].endswith(kind)),
        )
        result.append(choice["case_id"])
    return result


def make_plan(xodr):
    map_bytes = xodr.read_bytes()
    road_map = carla.Map("Town03", map_bytes.decode("utf-8"))
    scenarios = make_scenarios()
    routes, audit = audit_routes(road_map, scenarios)
    cases = make_cases(scenarios, variant=0)
    return {
        "map": "Town03",
        "xodr": str(xodr),
        "xodr_sha256": hashlib.sha256(map_bytes).hexdigest(),
        "sampling_method": (
            "Within each source route group, cross all nine weather/time presets with "
            "all specialists at common seeded starts. Rotate three lead starting indices "
            "and three deterministic lead profiles by route and condition. Development "
            "expansion uses alternate stressors and seeds on the same geometry partitions."
        ),
        "routes": {
            split: [asdict(scenario) for scenario in group]
            for split, group in scenarios.items()
        },
        "route_geometry_audit": audit,
        "legacy_archive_overlap": audit_archive_overlap(routes),
        "cases": cases,
        "smoke_case_ids": choose_smoke_cases(cases),
        "expansion_cases": make_cases(scenarios, variant=1) + make_cases(scenarios, variant=2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--xodr", type=Path, default=TOWN03_XODR)
    args = parser.parse_args()
    plan = make_plan(args.xodr)
    # An existing plan is the resume contract for all collection artifacts.
    contents = json.dumps(plan, indent=2, allow_nan=False) + "\n"
    if args.output.exists():
        if args.output.read_text() != contents:
            parser.error(f"Existing plan differs from current inputs: {args.output}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(contents)
    temporary.replace(args.output)
    print(f"Saved {len(plan['cases'])} matched cases to {args.output}")


if __name__ == "__main__":
    main()
