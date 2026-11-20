"""Train and test the pending condition specialists using the shared workflow."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
from threading import Event
import time


PENDING = ("day_sunny", "day_rainy", "sunset_sunny", "sunset_rainy",
           "night_sunny", "night_cloudy", "night_rainy")
PYTHON = ".venv/bin/python"


def log(message):
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def port_open():
    try:
        with socket.create_connection(("127.0.0.1", 2000), timeout=0.25):
            return True
    except OSError:
        return False


class Simulator:
    def __init__(self, root):
        self.root = root
        self.server = None

    def start(self):
        if port_open():
            raise RuntimeError("CARLA port 2000 is occupied by a different process")
        stamp = time.time_ns()
        path = self.root / f"carla-server-{stamp}.log"
        launch = [".local/carla-0.9.15/CarlaUE4.sh", "-RenderOffScreen",
                  "-quality-level=Epic", "-nosound", "-carla-rpc-port=2000"]
        env = dict(os.environ, VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/nvidia_icd.json")
        with path.open("w") as stream:
            self.server = subprocess.Popen(launch, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=True)
        for _ in range(60):
            if self.server.poll() is not None:
                break
            if port_open():
                log(f"CARLA ready ({path})")
                return
            time.sleep(1)
        raise RuntimeError(f"CARLA did not start, see {path}")

    def stop(self):
        if self.server is None:
            return
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(self.server.pid, sig)
            except ProcessLookupError:
                break
            try:
                self.server.wait(timeout=4)
            except subprocess.TimeoutExpired:
                continue
            if not port_open():
                break
        self.server = None

    def ensure(self):
        if self.server is None or self.server.poll() is not None or not port_open():
            self.stop()
            self.start()


def run_stage(condition, folder, stage, action=None):
    marker = folder / ".completed" / stage
    if marker.is_file():
        log(f"{condition} {stage} already complete")
        return
    log(f"{condition} {stage} started")
    if action is None:
        content = {"train_archive": "archive", "train_sync": "sync_model"}.get(stage)
        if content and (folder / content).exists():
            prior = folder / f"{content}-interrupted-{time.time_ns()}"
            (folder / content).rename(prior)
            log(f"{condition} preserved partial model stage at {prior}")
        subprocess.run(["bash", "specialists/workflow.sh", condition, stage, str(folder)],
                       env=dict(os.environ, DISTANCE_CNN_AMP="1"), check=True)
    else:
        action()
    marker.parent.mkdir(exist_ok=True)
    marker.touch()
    log(f"{condition} {stage} complete")


def report_live(condition, folder):
    command = [PYTHON, "-B", "-m", "analysis.summarize_specialist_live",
               "--experiment", str(folder), "--condition", condition,
               "--routes", "specialists/routes.json"]
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", choices=PENDING, nargs="+", default=PENDING)
    args = parser.parse_args()
    if len(set(args.conditions)) != len(args.conditions):
        parser.error("Conditions must be unique")
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / ".lock").open("a")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error(f"A batch is already active in {args.output}")

    folders = {}
    collected = {condition: Event() for condition in args.conditions}
    trained = {condition: Event() for condition in args.conditions}
    failures = {}
    for condition in args.conditions:
        folder = args.output / condition
        folder.mkdir(exist_ok=True)
        identity = folder / "condition.txt"
        if identity.exists() and identity.read_text().strip() != condition:
            raise ValueError(f"Wrong condition record in {folder}")
        identity.write_text(condition + "\n")
        folders[condition] = folder

    def error(condition, stage, exc):
        failures[f"{condition}/{stage}"] = str(exc)
        log(f"FAILED {condition} {stage} | {exc}")

    def training():
        for condition, folder in folders.items():
            try:
                run_stage(condition, folder, "train_archive")
                collected[condition].wait()
                if not (folder / ".completed" / "collect").is_file():
                    raise RuntimeError("Synchronized collection failed")
                for phase in ("train_sync", "publish", "evaluate", "plots"):
                    run_stage(condition, folder, phase)
            except (subprocess.CalledProcessError, OSError, RuntimeError) as exc:
                error(condition, "training", exc)
            finally:
                trained[condition].set()

    simulator = Simulator(args.output)

    def sim_stage(condition, folder, stage):
        if (folder / ".completed" / stage).is_file():
            return
        for attempt in range(2):
            try:
                simulator.ensure()
                run_stage(condition, folder, stage)
                return
            except (subprocess.CalledProcessError, OSError, RuntimeError) as exc:
                if attempt == 1:
                    raise
                log(f"{condition} {stage} retry after {exc}")
                simulator.stop()

    def sim_work():
        try:
            for condition, folder in folders.items():
                try:
                    sim_stage(condition, folder, "collect")
                except (subprocess.CalledProcessError, OSError, RuntimeError) as exc:
                    error(condition, "collect", exc)
                finally:
                    collected[condition].set()
            for condition, folder in folders.items():
                trained[condition].wait()
                if not (folder / ".completed" / "evaluate").is_file():
                    continue
                try:
                    sim_stage(condition, folder, "live")
                    run_stage(condition, folder, "report_live", lambda: report_live(condition, folder))
                except (subprocess.CalledProcessError, OSError, RuntimeError, ValueError) as exc:
                    error(condition, "live", exc)
        finally:
            simulator.stop()

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(training)
        b = pool.submit(sim_work)
        a.result()
        b.result()
    (args.output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    log(f"Batch ended with {len(failures)} failed stages")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
