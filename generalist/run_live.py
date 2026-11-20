"""Run the pooled CNN with each historical condition's PID settings."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from specialists.run_batch import Simulator, log


CHECKPOINT = Path("generalist/cnn.pth")


def run_condition(simulator, condition, output, *, checkpoint=CHECKPOINT,
                  profiles=Path("specialists")):
    path = output / condition
    marker = path / ".complete"
    if marker.exists():
        log(f"{condition}: five runs already saved")
        return
    profile = profiles / condition / "pid.json"
    for attempt in range(2):
        if path.exists():
            interrupted = output / "interrupted"
            interrupted.mkdir(exist_ok=True)
            path.rename(interrupted / f"{condition}-{time.time_ns()}")
        path.mkdir()
        run_log = path / "runner.log"
        command = [".venv/bin/python", "-B", "-u", "run_gap_pid.py", "--port", "2000",
                   "--pid-profile", str(profile), "--checkpoint", str(checkpoint),
                   "--output", str(path)]
        try:
            simulator.ensure()
            log(f"{condition}: started")
            with run_log.open("w") as stream:
                subprocess.run(
                    command, stdout=stream, stderr=subprocess.STDOUT,
                    env=dict(os.environ, OMP_NUM_THREADS="2"), check=True,
                )
        except (subprocess.CalledProcessError, OSError, RuntimeError):
            if attempt:
                raise
            log(f"{condition}: restarting CARLA after a runner error. See {run_log}.")
            simulator.stop()
            continue
        for seed in range(42, 47):
            summary = path / f"seed_{seed}.json"
            if not summary.is_file() or not summary.with_suffix(".csv").is_file():
                raise FileNotFoundError(f"Expected a rollout summary and trace for {condition}, seed {seed}")
        marker.touch()
        log(f"{condition}: five runs saved")
        return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    conditions = json.loads(CHECKPOINT.with_suffix(".json").read_text())["conditions"]
    if len(set(conditions)) != 9:
        raise ValueError("Expected the full nine-condition generalist")
    args.output.mkdir(exist_ok=True, parents=True)
    simulator = Simulator(args.output)
    try:
        for condition in sorted(conditions, key=lambda n: (
            ("day", "sunset", "night").index(n.split("_")[0]),
            ("sunny", "cloudy", "rainy").index(n.split("_")[1]),
        )):
            run_condition(simulator, condition, args.output)
    finally:
        simulator.stop()


if __name__ == "__main__":
    main()
