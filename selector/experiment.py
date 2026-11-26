"""Collect, fit, and evaluate the CARLA logistic context selector."""

import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from selector.model import CONTROLLERS, Selector, encode_context
from specialists.run_batch import Simulator, log


FAILURE_FLAGS = {"is_collision", "terminal_dist", "out_of_lane"}
MODES = ("selector", "generalist", "weather_matched")
ROOT = Path("experiments/logistic_context_selector_2026-11-19")
SPECIALISTS = Path("specialists")


def atomic_text(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def save_json(path, content):
    atomic_text(path, json.dumps(content, indent=2, allow_nan=False) + "\n")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def checkpoints(specialists=SPECIALISTS):
    paths = {name: specialists / name / "cnn.pth" for name in CONTROLLERS}
    paths["generalist"] = Path("generalist/cnn.pth")
    manifest = {}
    for name, path in paths.items():
        info = path.with_name("cnn.json" if name == "generalist" else "pid.json")
        expected = json.loads(info.read_text())["cnn_sha256"]
        if digest(path) != expected:
            raise ValueError(f"Checkpoint checksum differs at {path}")
        manifest[name] = {"path": str(path), "sha256": expected}
    return paths, manifest


def check_trace(trial):
    rows = trial["rows"]
    decision_gaps = {row["decision_frame"]: row["true_gap_m"] for row in rows}
    for i, row in enumerate(rows, 1):
        if row["step"] != i or (i > 1 and
                                row["decision_frame"] != rows[i - 2]["decision_frame"] + 1):
            raise ValueError("Nonconsecutive CARLA decision frames")
        age = row["decision_frame"] - row["capture_frame"]
        if age < 0 or age != row["capture_age_steps"]:
            raise ValueError("A camera frame is future-dated or has an inconsistent age")
        if i > 1 and row["capture_frame"] < rows[i - 2]["capture_frame"]:
            raise ValueError("Camera capture frames went backwards within a rollout")
        if row["next_frame"] != row["decision_frame"] + 1 or not np.isclose(
            row["time_s"], (i - 1) * trial["dt_s"], atol=1e-4
        ):
            raise ValueError("A decision interval does not match the fixed CARLA timestep")
        if i > 1 and row["true_gap_m"] != rows[i - 2]["post_step_true_gap_m"]:
            raise ValueError("A post-step gap does not match the next decision")
        if row["capture_frame"] in decision_gaps and (
            row["capture_true_gap_m"] != decision_gaps[row["capture_frame"]]
        ):
            raise ValueError("Capture-frame gap does not match the simulator snapshot")
        if ("terminal_dist" in row["flags"]) != (row["post_step_true_gap_m"] > 10):
            raise ValueError("Distance termination disagrees with the strict 10 m cutoff")
        measurements = [
            row[field] for field in ("generalist_gap_m", "specialist_gap_m", "capture_true_gap_m",
                                     "true_gap_m", "post_step_true_gap_m", "pid_command")
            if row[field] is not None
        ]
        if not np.isfinite(measurements).all():
            raise ValueError("Non-finite measurement in the CARLA trace")


def examples(trial, case, controller):
    rows = trial["rows"]
    horizon = trial["horizon_steps"]
    count = max(1, (len(rows) + horizon - 1) // horizon)
    for index in range(count):
        window = rows[index * horizon:(index + 1) * horizon]
        first = window[0] if window else None
        flags = {flag for row in window for flag in row["flags"]} & FAILURE_FLAGS
        if flags:
            label, reason = 0, ", ".join(sorted(flags))
        elif len(window) == horizon:
            label, reason = 1, "full_violation_free_window"
        else:
            label, reason = None, trial["terminal_reason"]
        if first is not None and first["generalist_gap_m"] is None:
            label, reason = None, "missing_context_prediction"
        yield {
            "case_id": case["case_id"], "source_route": case["route"]["name"],
            "split": case["split"], "condition": case["condition"], "seed": case["seed"],
            "lead_profile": case["lead_profile"], "lead_start_index": case["lead_start_index"],
            "controller": controller, "window_index": index, "label": label, "reason": reason,
            "features": list(encode_context(case["condition"], first["generalist_gap_m"]))
            if first is not None and first["generalist_gap_m"] is not None else None,
            "initial_true_gap_m": first["true_gap_m"] if first else None,
            "context_capture_true_gap_m": first["capture_true_gap_m"] if first else None,
            "context_frame": first["capture_frame"] if first else None,
            "observed_steps": len(window), "horizon_steps": horizon, "dt_s": trial["dt_s"],
        }


def count_examples(records):
    counts = {name: Counter() for name in CONTROLLERS}
    for row in records:
        key = {0: "failure", 1: "success", None: "excluded"}[row["label"]]
        counts[row["controller"]][key] += 1
    return {
        name: {key: counts[name][key] for key in ("success", "failure", "excluded")}
        for name in CONTROLLERS
    }


def write_examples(path, records):
    atomic_text(path, "".join(json.dumps(row, allow_nan=False) + "\n" for row in records))


def pick_smoke_cases(plan):
    indexed = {case["case_id"]: case for case in plan["cases"]}
    return [indexed[key] for key in plan["smoke_case_ids"]]


class Experiment:
    def __init__(self, root, plan, options, *, specialists=SPECIALISTS):
        self.root = root
        self.plan = plan
        self.options = options
        self.engine = None
        self.simulator = Simulator(root)
        self.paths, fingerprints = checkpoints(specialists)
        settings = {
            "experiment": "adapted logistic context selector",
            "plan_sha256": digest(root / "plan.json"),
            "checkpoints": fingerprints,
            "horizon_s": options.horizon,
            "decision_interval_s": options.decision_interval,
            "collection_windows_per_trial": options.windows,
            "target_gap_m": 6.0,
            "distance_limit_m": 10.0,
            "lane_limit_m": 5.0,
            "warmup_steps": 5,
            "pid_gains": [0.9, 0.05, 0.15],
            "max_steps": 360,
        }
        path = root / "settings.json"
        if path.exists() and json.loads(path.read_text()) != settings:
            raise ValueError(f"Settings changed within {root}; use a new experiment directory")
        if not path.exists():
            save_json(path, settings)

    def start(self):
        from selector.rollout import RolloutEngine

        if self.engine is None:
            self.simulator.ensure()
            self.engine = RolloutEngine()
        return self.engine

    def close(self):
        try:
            if self.engine is not None:
                self.engine.close()
        finally:
            self.engine = None
            self.simulator.stop()

    def trial(self, case, mode, load_selector=None, collection=False):
        folder = "trials" if collection else "live"
        path = self.root / folder / case["case_id"] / f"{mode}.json"
        if path.exists():
            result = json.loads(path.read_text())
            if result["case"] != case or result["mode"] != mode:
                raise ValueError(f"Saved run metadata differs at {path}")
            check_trace(result)
            return result
        errors = []
        for attempt in range(2):
            engine = self.start()
            try:
                result = engine.run(
                    case, mode, self.paths, selector=load_selector,
                    horizon_s=self.options.horizon, decision_s=self.options.decision_interval,
                    max_steps=360, windows=self.options.windows if collection else None,
                )
            except (RuntimeError, OSError) as error:
                errors.append(f"{type(error).__name__}: {error}")
                log(f"{case['case_id']} {mode}: simulator attempt {attempt + 1} failed: {error}")
                self.close()
                if attempt == 0:
                    continue
            else:
                result.update(case=case, mode=mode, recoverable_errors=errors)
                check_trace(result)
                if result["target_gap_m"] != 6 or result["distance_limit_m"] != 10:
                    raise ValueError("Unexpected PID target or cutoff in the saved rollout")
                save_json(path, result)
                return result
        raise RuntimeError(f"Unable to obtain an auditable trajectory for {case['case_id']} {mode}: {errors}")

    def collect(self, cases, stage):
        records = []
        matches = []
        from selector.reporting import match_starts

        for index, case in enumerate(cases, 1):
            runs = {name: self.trial(case, name, collection=True) for name in CONTROLLERS}
            matched = match_starts(runs)
            group = [row for name, run in runs.items() for row in examples(run, case, name)]
            if not matched["matched"]:
                for row in group:
                    row.update(label=None, reason="unmatched_or_missing_start")
            records.extend(group)
            matches.append({"case_id": case["case_id"], **matched})
            if stage == "smoke" or index == 1 or index % 4 == 0 or index == len(cases):
                dispositions = Counter(row["label"] for row in group)
                log(f"{stage} {index}/{len(cases)} {case['case_id']}: "
                    f"{dispositions[1]} success, {dispositions[0]} failure, "
                    f"{dispositions[None]} excluded windows")
        return records, matches

    def smoke(self):
        cases = pick_smoke_cases(self.plan)
        records, matches = self.collect(cases, "smoke")
        counts = count_examples(records)
        rows = [row for case in cases for mode in CONTROLLERS for row in json.loads(
            (self.root / "trials" / case["case_id"] / f"{mode}.json").read_text()
        )["rows"]]
        passed = (all(match["matched"] for match in matches)
                  and all(t["success"] + t["failure"] > 0 for t in counts.values())
                  and any(t["success"] for t in counts.values())
                  and any(t["failure"] for t in counts.values()) and bool(rows))
        smoke = {
            "passed": passed, "case_ids": [case["case_id"] for case in cases],
            "trial_count": len(cases) * len(CONTROLLERS), "matches": matches,
            "counts_by_controller": counts, "rows": len(rows),
            "camera_age_steps": dict(Counter(row["capture_age_steps"] for row in rows)),
            "decision_frames_are_consecutive": True, "no_future_or_backward_capture_frames": True,
            "capture_gaps_match_recorded_decision_snapshots": True,
        }
        write_examples(self.root / "smoke_examples.jsonl", records)
        save_json(self.root / "smoke.json", smoke)
        for name, counts in counts.items():
            log(f"Smoke {name}: {counts}")
        if not passed:
            raise RuntimeError("Smoke checks did not pass; inspect smoke.json and saved traces")
        log(f"Smoke passed: {smoke['trial_count']} matched trials and {len(rows)} timestamped decisions")

    def fit(self):
        cases = list(self.plan["cases"])
        records, matches = self.collect(cases, "collection")
        if any(t["success"] == 0 or t["failure"] == 0
               for t in count_examples(row for row in records if row["split"] == "train").values()):
            extra = self.plan.get("expansion_cases", [])
            if extra:
                log("Some controller heads lack class variation; collecting planned development stress cases")
                more, additional_matches = self.collect(extra, "expansion")
                records.extend(more)
                matches.extend(additional_matches)
        save_json(self.root / "collection.json", {
            "matched_cases": matches,
            "counts_by_split": {
                split: count_examples(row for row in records if row["split"] == split)
                for split in ("train", "validation", "test")
            },
        })
        samples = self.root / "examples.jsonl"
        write_examples(samples, records)
        subprocess.run(
            [sys.executable, "-B", "-m", "selector.model", "--examples", str(samples),
             "--output", str(self.root / "model.json")], check=True,
        )

    def evaluate(self):
        from selector.reporting import match_starts, write_live_report

        fitted = Selector.load(self.root / "model.json")
        cases = [case for case in self.plan["cases"] if case["split"] == "test"]
        pairs = []
        for index, case in enumerate(cases, 1):
            runs = {mode: self.trial(case, mode, load_selector=fitted) for mode in MODES}
            match = match_starts(runs)
            pairs.append((case, runs, match))
            ends = ", ".join(f"{mode}={run['terminal_reason']}" for mode, run in runs.items())
            log(f"Live evaluation {index}/{len(cases)} {case['case_id']}: {ends}")
        write_live_report(self.root, pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=ROOT)
    parser.add_argument("--specialists", "--specialist-bank", type=Path, default=SPECIALISTS,
                        help="Directory containing each specialist's cnn.pth and pid.json")
    parser.add_argument("--stage", choices=("all", "smoke", "fit", "evaluate"), default="all")
    parser.add_argument("--horizon", type=float, default=2.0)
    parser.add_argument("--decision-interval", type=float, default=2.0)
    parser.add_argument("--windows", type=int, default=2)
    args = parser.parse_args()
    if args.windows < 1 or not all(np.isfinite(x) and x > 0 for x in
                                   (args.horizon, args.decision_interval)):
        parser.error("Use positive, finite intervals and at least one collection window")
    plan = json.loads((args.experiment / "plan.json").read_text())
    with (args.experiment / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        experiment = Experiment(args.experiment, plan, args, specialists=args.specialists)
        try:
            if args.stage in ("all", "smoke"):
                experiment.smoke()
            if args.stage in ("all", "fit"):
                smoke = args.experiment / "smoke.json"
                if not smoke.exists() or not json.loads(smoke.read_text())["passed"]:
                    raise RuntimeError("Run and pass the live smoke test before collecting")
                experiment.fit()
            if args.stage in ("all", "evaluate"):
                experiment.evaluate()
        finally:
            experiment.close()


if __name__ == "__main__":
    main()
