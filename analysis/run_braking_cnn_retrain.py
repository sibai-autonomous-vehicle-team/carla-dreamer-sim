"""Train and compare the nine expanded-data CNN–PID specialists."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from threading import Lock
import time

from specialists.run_batch import Simulator, log
from train_distance_cnn import CONDITIONS


ROOT = Path("experiments/specialists_braking_1300_2026-11-23")
THREADS = 2
STAGE_LOCK = Lock()
EPISODES = (42, 43, 44, 45, 46)


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def snapshot_baselines(root):
    routes = Path("specialists/routes.json")
    frozen_routes = root / "routes.json"
    if not frozen_routes.exists():
        shutil.copyfile(routes, frozen_routes)
    elif digest(routes) != digest(frozen_routes):
        raise ValueError("The shared route manifest differs from the saved benchmark")
    if len(read_json(frozen_routes)["test"]) != len(EPISODES):
        raise ValueError("This benchmark expects five held-out routes")

    for condition in CONDITIONS:
        deployed = Path("specialists") / condition
        baseline = root / condition / "before"
        baseline.mkdir(parents=True, exist_ok=True)
        profile_file = baseline / "pid.json"
        profile = read_json(profile_file if profile_file.exists() else deployed / "pid.json")
        settings = (profile["condition"], profile["target_gap"], profile["kp"], profile["ki"],
                    profile["kd"], profile["seed"], profile["episodes"], profile["steps"],
                    profile["warmup_steps"], profile["out_lane_thres"], profile["shared_route"])
        if settings != (condition, 6.0, 0.9, 0.05, 0.15, 42, 5, 360, 5, 5, True):
            raise ValueError(f"Unexpected baseline PID settings for {condition}: {settings}")
        path = baseline / "cnn.pth"
        if not path.exists():
            source = deployed / "cnn.pth"
            if digest(source) != profile["cnn_sha256"]:
                raise ValueError(f"Deployed checkpoint hash differs for {condition}")
            temporary = path.with_suffix(".pth.tmp")
            shutil.copyfile(source, temporary)
            temporary.replace(path)
        if digest(path) != profile["cnn_sha256"]:
            raise ValueError(f"Frozen baseline checkpoint hash differs for {condition}")
        if not profile_file.exists():
            profile.pop("live_report", None)
            profile.update(terminal_dist=10, route_manifest="../../routes.json", status="baseline")
            write_json(profile_file, profile)
        elif profile["terminal_dist"] != 10 or profile["route_manifest"] != "../../routes.json":
            raise ValueError(f"Frozen benchmark profile changed for {condition}")


def invoke(root, command, output_log):
    env = dict(os.environ, OMP_NUM_THREADS=str(THREADS), MKL_NUM_THREADS=str(THREADS),
               TORCH_HOME=str(Path(".local/torch").resolve()), MPLBACKEND="Agg")
    output_log.parent.mkdir(parents=True, exist_ok=True)
    with STAGE_LOCK, (root / "commands.jsonl").open("a") as history:
        history.write(json.dumps({"command": command, "log": str(output_log)}) + "\n")
    with output_log.open("a") as stream:
        subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def save_or_reuse(root, folder, command, logfile, validate):
    marker = folder / ".complete"
    if not marker.exists():
        if folder.exists():
            old = folder.with_name(f"{folder.name}-interrupted-{time.time_ns()}")
            folder.rename(old)
        folder.mkdir(parents=True)
        invoke(root, command, logfile)
        validate(folder)
        marker.touch()
        return False
    validate(folder)
    return True


def validate_training(folder, split_path, epochs):
    if not (folder / "best.pth").is_file() or read_json(folder / "split.json") != read_json(split_path):
        raise ValueError(f"Training split or checkpoint missing at {folder}")
    with (folder / "epochs.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if [int(row["epoch"]) for row in rows] != list(range(1, epochs + 1)):
        raise ValueError(f"Expected {epochs} completed epochs in {folder}")


def validate_offline(folder, split_path):
    metrics = read_json(folder / "test_metrics.json")
    if metrics["test_episodes"] != len(read_json(split_path)["test_ids"]):
        raise ValueError(f"Unexpected test episode count in {folder}")
    if not (folder / "test_predictions.pth").is_file() or metrics["test_frames"] < 1:
        raise ValueError(f"Missing saved test predictions in {folder}")


def fit_condition(root, condition, epochs, batch_size):
    result = root / condition
    data = root / "data" / condition
    split = data / "split.json"
    if [len(read_json(split)[name]) for name in ("train_ids", "val_ids", "test_ids")] != [1040, 130, 130]:
        raise ValueError(f"Unexpected combined source split for {condition}")
    trained = result / "train"
    trainer = [sys.executable, "-B", "-u", "train_distance_cnn.py", "--data", str(data),
               "--split", str(split), "--epochs", str(epochs), "--batch-size", str(batch_size),
               "--seed", "42", "--amp", "--output", str(trained)]
    start = time.monotonic()
    log(f"{condition}: CNN training started")
    reused = save_or_reuse(root, trained, trainer, result / "training.log",
                           lambda path: validate_training(path, split, epochs))
    if reused:
        log(f"{condition}: reusing {epochs} verified epochs")
    else:
        log(f"{condition}: {epochs} epochs finished in {(time.monotonic() - start) / 60:.1f} min")
    curve = trained / "training_curves.png"
    if not curve.is_file():
        invoke(root, [sys.executable, "-B", "-m", "analysis.plot_distance_cnn_curves", "--epochs",
                      str(trained / "epochs.csv"), "--output", str(curve)], result / "plots.log")

    # Keep the new CNN and runnable experiment-local PID profile beside the baseline.
    specialist = result / "specialist"
    specialist.mkdir(exist_ok=True)
    after = specialist / "cnn.pth"
    if not after.exists():
        after.symlink_to(Path("../train/best.pth"))
    if after.resolve() != (trained / "best.pth").resolve():
        raise ValueError(f"Unexpected retrained checkpoint link for {condition}")
    profile = read_json(result / "before" / "pid.json")
    profile.update(cnn_sha256=digest(after), cnn_source=str(trained / "best.pth"),
                   status="trained", training_split=str(split))
    write_json(specialist / "pid.json", profile)

    for variant, model in (("before", result / "before" / "cnn.pth"), ("after", after)):
        for source in ("original", "braking"):
            source_split = data / f"split_{source}.json"
            folder = result / "offline" / variant / source
            command = [sys.executable, "-B", "-u", "evaluate_distance_cnn.py", "--checkpoint",
                       str(model), "--data", str(data), "--split", str(source_split), "--batch-size",
                       str(batch_size), "--output", str(folder)]
            save_or_reuse(root, folder, command, result / f"offline_{variant}_{source}.log",
                          lambda path: validate_offline(path, source_split))
    log(f"{condition}: original and braking holdouts evaluated for both checkpoints")


def fit_all(root, workers, epochs, batch_size):
    failed = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        jobs = {executor.submit(fit_condition, root, cond, epochs, batch_size): cond
                for cond in CONDITIONS}
        for job in as_completed(jobs):
            cond = jobs[job]
            try:
                job.result()
            except (OSError, subprocess.CalledProcessError, ValueError) as error:
                failed[cond] = str(error)
                log(f"{cond}: parallel training/evaluation task failed: {error}")
    for condition in failed:
        # Retry without GPU contention and reuse any stages that already finished.
        log(f"{condition}: retrying with exclusive training-slot access")
        fit_condition(root, condition, epochs, batch_size)


def validate_live(folder):
    for seed in EPISODES:
        path = folder / f"seed_{seed}.json"
        run = read_json(path)
        if run["seed"] != seed or run["target_m"] != 6 or run["distance_limit_m"] != 10:
            raise ValueError(f"Unexpected live configuration at {path}")
        if not path.with_suffix(".csv").is_file():
            raise ValueError(f"Missing CARLA trace for {path}")


def run_live(root):
    server = Simulator(root)
    try:
        for condition in CONDITIONS:
            result = root / condition
            for variant, model_folder in (("before", "before"), ("after", "specialist")):
                folder = result / "live" / variant
                command = [sys.executable, "-B", "-u", "run_gap_pid.py", "--port", "2000",
                           "--pid-profile", str(result / "before" / "pid.json"),
                           "--checkpoint", str(result / model_folder / "cnn.pth"),
                           "--output", str(folder)]
                for attempt in range(2):
                    try:
                        if not (folder / ".complete").exists():
                            server.ensure()
                        save_or_reuse(root, folder, command, result / f"live_{variant}.log", validate_live)
                    except (OSError, subprocess.CalledProcessError, RuntimeError, ValueError):
                        if attempt:
                            raise
                        log(f"{condition} {variant}: restarting CARLA, then retrying")
                        server.stop()
                    else:
                        log(f"{condition} {variant}: five live rollouts verified")
                        break
    finally:
        server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=ROOT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--stage", choices=("all", "fit", "live"), default="all")
    args = parser.parse_args()
    if min(args.workers, args.epochs, args.batch_size) < 1:
        parser.error("Worker, epoch, and batch-size values must be positive")
    root = args.experiment
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        settings_file = root / "settings.json"
        settings = {"epochs": args.epochs, "batch_size": args.batch_size, "seed": 42,
                    "target_gap_m": 6.0, "distance_limit_m": 10,
                    "fine_tuning": "ImageNet initialization followed by mixed-source gap training"}
        if settings_file.exists() and read_json(settings_file) != settings:
            raise ValueError(f"Run settings changed in {root}; choose a new experiment directory")
        if not settings_file.exists():
            write_json(settings_file, settings)
        snapshot_baselines(root)
        if args.stage in ("all", "fit"):
            log(f"Starting {len(CONDITIONS)} specialist jobs with up to {args.workers} GPU workers")
            fit_all(root, args.workers, args.epochs, args.batch_size)
        if args.stage in ("all", "live"):
            run_live(root)


if __name__ == "__main__":
    main()
