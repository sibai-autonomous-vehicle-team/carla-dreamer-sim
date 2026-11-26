"""Apply the synchronized second stage to the saved 1,300-trajectory specialists."""

import argparse
import csv
import fcntl
from pathlib import Path
import shutil
import subprocess
import sys
import time

from analysis.run_braking_cnn_retrain import (
    ROOT as BASE, digest, invoke, read_json, save_or_reuse, validate_offline, write_json,
)
from specialists.run_batch import log
from train_distance_cnn import CONDITIONS


ROOT = Path("experiments/specialists_braking_1300_sync_2026-11-24")
EXPERIMENTS = Path("experiments")
EPOCHS = 20
BATCH_SIZE = 32


def original_inputs(condition):
    if condition == "day_cloudy":
        return (EXPERIMENTS / "day_cloudy_resnet18_pid_2026-11-17" / "best.pth",
                EXPERIMENTS / "day_cloudy_cnn_close_v3_2026-11-17" / "merged_split.json")
    historical = (EXPERIMENTS / "sunset_cloudy_specialist_2026-11-17" if condition == "sunset_cloudy"
                  else EXPERIMENTS / "all_specialists_2026-11-17" / condition)
    return historical / "archive" / "best.pth", historical / "merged_split.json"


def fingerprint(path):
    return {"path": str(path), "sha256": digest(path)}


def input_inventory(root, base):
    prior = read_json(base / "summary.json")
    conditions = {}
    for condition in CONDITIONS:
        initial = base / condition / "train" / "best.pth"
        deployed = base / condition / "before" / "cnn.pth"
        profile = read_json(deployed.with_name("pid.json"))
        old_base, captures = original_inputs(condition)
        condition_inputs = dict(
            base_800=fingerprint(old_base), deployed=fingerprint(deployed),
            base_1300=fingerprint(initial), synchronized_captures=fingerprint(captures),
        )
        if condition_inputs["base_1300"]["sha256"] != prior["training"][condition]["checkpoint_sha256"]:
            raise ValueError(f"Base 1300 checkpoint hash differs for {condition}")
        if condition_inputs["deployed"]["sha256"] != profile["cnn_sha256"]:
            raise ValueError(f"Frozen deployed checkpoint hash differs for {condition}")
        conditions[condition] = condition_inputs
    record = dict(epochs=EPOCHS, batch_size=BATCH_SIZE, seed=42, legacy_replay_frames=0,
                  max_gpu_workers=1, conditions=conditions)
    path = root / "inputs.json"
    if path.exists() and read_json(path) != record:
        raise ValueError(f"Checkpoint or capture inputs changed within {root}")
    if not path.exists():
        write_json(path, record)
    routes = root / "routes.json"
    if not routes.exists():
        shutil.copyfile(base / "routes.json", routes)
    elif digest(routes) != digest(base / "routes.json"):
        raise ValueError("The experiment-local route manifest changed")
    return conditions


def validate_sync(sync, base_checkpoint):
    summary = read_json(sync / "summary.json")
    with (sync / "epochs.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if [int(row["epoch"]) for row in rows] != list(range(1, EPOCHS + 1)):
        raise ValueError(f"Missing epochs in {sync}")
    if Path(summary["initial_checkpoint"]).resolve() != base_checkpoint.resolve():
        raise ValueError(f"Unexpected initial checkpoint for {sync}")
    if summary["legacy_replay_frames"] != 0:
        raise ValueError(f"Unexpected replay frames in {sync}")
    eligible = summary["selection_status"] == "eligible_checkpoint"
    expected = sync / ("best.pth" if eligible else "best_diagnostic.pth")
    selected = expected if eligible else base_checkpoint
    if (not expected.is_file()
            or expected.resolve() != Path(summary["evaluated_candidate_checkpoint"]).resolve()
            or selected.resolve() != Path(summary["selected_checkpoint"]).resolve()
            or not (sync / "test_predictions.pth").is_file()):
        raise ValueError(f"Missing or inconsistent fine-tuning outputs in {sync}")
    return summary, expected


def run_condition(root, base, condition, inputs):
    folder = root / condition
    sync = folder / "sync"
    marker = sync / ".complete"
    first = Path(inputs["base_1300"]["path"])
    if not marker.exists():
        if sync.exists():
            sync.rename(sync.with_name(f"sync-interrupted-{time.time_ns()}"))
        log(f"{condition}: synchronized fine-tuning started")
        command = [sys.executable, "-B", "-u", "-m", "analysis.finetune_distance_cnn_sync",
                   "--manifest", inputs["synchronized_captures"]["path"],
                   "--checkpoint", str(first), "--output", str(sync), "--close-aware",
                   "--epochs", str(EPOCHS), "--batch-size", str(BATCH_SIZE), "--seed", "42",
                   "--replay-frames", "0"]
        invoke(root, command, folder / "finetuning.log")
        validate_sync(sync, first)
        marker.touch()
    summary, candidate = validate_sync(sync, first)
    eligible = summary["selection_status"] == "eligible_checkpoint"
    log(f"{condition}: 20 epochs verified, {summary['selection_status']}, evaluated epoch {summary['best_epoch']}")

    if eligible:
        specialist = folder / "specialist"
        specialist.mkdir(exist_ok=True)
        local_weights = specialist / "cnn.pth"
        if not local_weights.exists():
            local_weights.symlink_to(Path("../sync/best.pth"))
        if local_weights.resolve() != candidate.resolve():
            raise ValueError(f"Unexpected specialist checkpoint link for {condition}")
        profile = read_json(base / condition / "before" / "pid.json")
        profile.update(cnn_source=str(candidate), cnn_sha256=digest(candidate),
                       status="offline_validated")
        write_json(specialist / "pid.json", profile)

    data = base / "data" / condition
    for source in ("original", "braking"):
        split = data / f"split_{source}.json"
        output = folder / "offline" / source
        command = [sys.executable, "-B", "-u", "evaluate_distance_cnn.py",
                   "--checkpoint", str(candidate), "--data", str(data), "--split", str(split),
                   "--batch-size", str(BATCH_SIZE), "--output", str(output)]
        save_or_reuse(root, output, command, folder / f"offline_{source}.log",
                      lambda path: validate_offline(path, split))
    log(f"{condition}: both held-out source evaluations saved")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=ROOT)
    parser.add_argument("--base-experiment", type=Path, default=BASE)
    args = parser.parse_args()
    args.experiment.mkdir(parents=True, exist_ok=True)
    with (args.experiment / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inventory = input_inventory(args.experiment, args.base_experiment)
        failed = {}
        order = ("day_cloudy", "night_cloudy", *(c for c in CONDITIONS if c not in
                                                ("day_cloudy", "night_cloudy")))
        for condition in order:
            try:
                run_condition(args.experiment, args.base_experiment, condition, inventory[condition])
            except (OSError, subprocess.CalledProcessError, ValueError) as error:
                failed[condition] = str(error)
                log(f"{condition}: {error}")
        write_json(args.experiment / "failures.json", failed)
        if failed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
