"""Set a 9 m cutoff and replay the matched generalist and specialist benchmarks."""

import json
from pathlib import Path
import shutil

from generalist.run_live import CHECKPOINT, run_condition
from specialists.run_batch import Simulator, log


EXPERIMENT = Path("experiments/live_cutoff_9m_2026-11-18")
SPECIALISTS = Path("specialists")
BEFORE = EXPERIMENT / "profiles_10m"
AFTER = EXPERIMENT / "profiles_9m"


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def prepare_profiles():
    paths = sorted(SPECIALISTS.glob("*/pid.json"))
    if len(paths) != 9:
        raise ValueError("The 9 m benchmark requires nine profiles")
    for root in (BEFORE, AFTER):
        root.mkdir(exist_ok=True)
        routes = root / "routes.json"
        if not routes.exists():
            shutil.copyfile(SPECIALISTS / "routes.json", routes)
    for path in paths:
        condition = path.parent.name
        relative = Path(condition) / "pid.json"
        original = BEFORE / relative
        if not original.exists():
            previous = json.loads(path.read_text())
            if previous["condition"] != condition or previous["terminal_dist"] != 10:
                raise ValueError(f"Expected a 10 m starting profile at {path}")
            original.parent.mkdir(exist_ok=True)
            shutil.copyfile(path, original)
        if not (AFTER / relative).exists():
            settings = json.loads(original.read_text())
            settings.update(terminal_dist=9, status="pending_validation")
            settings.pop("live_report", None)
            write_json(AFTER / relative, settings)
        active = json.loads(path.read_text())
        trial = json.loads((AFTER / relative).read_text())
        if active["terminal_dist"] == 10:
            write_json(path, trial)
        elif active["terminal_dist"] != 9:
            raise ValueError(f"Unexpected active cutoff in {path}")
    log("All nine live PID profiles use a 9 m true-gap cutoff")
    return [path.parent.name for path in paths]


def main():
    conditions = prepare_profiles()
    outputs = {name: EXPERIMENT / "live" / name for name in ("generalist", "specialist")}
    for path in outputs.values():
        path.mkdir(parents=True, exist_ok=True)
    simulator = Simulator(EXPERIMENT)
    try:
        for condition in conditions:
            models = {"generalist": CHECKPOINT, "specialist": SPECIALISTS / condition / "cnn.pth"}
            for mode, checkpoint in models.items():
                log(f"Running {mode} in {condition}")
                run_condition(simulator, condition, outputs[mode], checkpoint=checkpoint, profiles=AFTER)
    finally:
        simulator.stop()


if __name__ == "__main__":
    main()
