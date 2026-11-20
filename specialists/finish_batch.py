"""Verify the trained packages and write the comparison across all nine specialists."""
import argparse
import csv
import json
from pathlib import Path

from PIL import Image
from publish import checksum
from run_batch import PENDING


DAY = Path("experiments/day_cloudy_cnn_close_v3_2026-11-17")
SUNSET = Path("experiments/sunset_cloudy_specialist_2026-11-17")


def read_json(path):
    return json.loads(path.read_text())


def add(condition, experiment, frames, native, transfer, live):
    return dict(
        condition=condition, archived_test_frames=frames,
        native_archived_prediction_mae_m=native, daytime_transfer_mae_m=transfer,
        live_finishes=live["finishes"], live_episodes=live["episodes"],
        collisions=live["collisions"], distance_cutoffs=live["distance_violations"],
        route_violations=live["route_violations"], timeouts=live["timeouts"],
        max_live_true_gap_m=live["max_true_gap_m"],
        min_live_true_gap_m=live["min_true_gap_m"],
        mean_live_tracking_mae_m=live["mean_run_tracking_mae_m"],
        experiment=str(experiment),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    day_test = read_json(DAY / "finetune_v3/summary.json")["finetuned"]["legacy_test"]["overall"]
    day_live = read_json(DAY / "live_summary.json")["normal"]["v3"]
    day_live = dict(day_live, episodes=day_live["runs"], distance_violations=day_live["cutoffs"],
                    timeouts=day_live["runs"] - day_live["finishes"],
                    mean_run_tracking_mae_m=day_live["mean_per_run_tracking_mae_m"])
    results.append(add("day_cloudy", DAY, day_test["frames"], day_test["mae_m"],
                       day_test["mae_m"], day_live))
    for condition, e in (("sunset_cloudy", SUNSET),
                         *[(condition, args.output / condition) for condition in PENDING]):
        native = read_json(e / "specialist_evaluation/test_metrics.json")
        transfer = read_json(e / "day_transfer/test_metrics.json")
        live = read_json(e / "live/summary.json")
        assert live["settings"]["condition"] == condition and native["test_episodes"] == 80
        assert native["test_frames"] == transfer["test_frames"]
        results.append(add(condition, e, native["test_frames"], native["mae_m"], transfer["mae_m"],
                           live["by_mode"]["specialist"]))

    for result in results:
        condition = result["condition"]
        profile_path = Path("specialists") / condition / "pid.json"
        model = profile_path.parent / "cnn.pth"
        profile = read_json(profile_path)
        assert profile["condition"] == condition
        assert checksum(model) == profile["cnn_sha256"] == checksum(Path(profile["cnn_source"]))
        if condition in PENDING:
            e = args.output / condition
            assert read_json(e / "dataset_verification.json")["condition"] == condition
            plots = list(e.rglob("*.png"))
            assert len(plots) >= 9, (condition, plots)
            for path in plots:
                with Image.open(path) as image:
                    image.verify()
            success = (result["live_finishes"] == result["live_episodes"] and
                       all(result[key] == 0 for key in ("collisions", "distance_cutoffs",
                                                        "route_violations", "timeouts")))
            profile.update(status="validated" if success else "tested",
                           live_report=str(e / "live/summary.json"))
            temporary = profile_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(profile, indent=2) + "\n")
            temporary.replace(profile_path)
        result["status"] = profile["status"]

    with (args.output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda result: result["condition"]))
    print(f"Verified {len(results)} specialist checkpoints and PID profiles")


if __name__ == "__main__":
    main()
