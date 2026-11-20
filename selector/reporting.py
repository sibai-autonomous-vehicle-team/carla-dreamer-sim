"""Matched-start checks and paired live CARLA selector reports."""

import csv
from collections import Counter
from itertools import combinations
import math
from pathlib import Path

import numpy as np

from selector.model import CONTROLLERS, write_json


START_TOLERANCES = {
    "position_m": 0.03,
    "rotation_deg": 1.0,
    "velocity_mps": 0.1,
    "gap_m": 0.03,
}


def state_deltas(states):
    poses = [np.asarray(state["state"], dtype=float).reshape(2, 6) for state in states]
    velocities = [np.asarray(state["velocities"], dtype=float).reshape(2, 3) for state in states]
    gaps = np.asarray([state["initial_gap_m"] for state in states], dtype=float)
    if not all(np.isfinite(values).all() for values in (*poses, *velocities, gaps)):
        raise ValueError("Nonfinite starting state")
    pose_pairs = list(combinations(poses, 2))
    return {
        "same_actor_types": all(state["actor_types"] == states[0]["actor_types"] for state in states),
        "position_m": max(
            (float(np.linalg.norm(a[:, :3] - b[:, :3], axis=1).max()) for a, b in pose_pairs),
            default=0.0,
        ),
        "horizontal_position_m": max(
            (float(np.linalg.norm(a[:, :2] - b[:, :2], axis=1).max()) for a, b in pose_pairs),
            default=0.0,
        ),
        "yaw_deg": max(
            (float(np.abs((a[:, 4] - b[:, 4] + 180.0) % 360.0 - 180.0).max())
             for a, b in pose_pairs),
            default=0.0,
        ),
        "rotation_deg": max(
            (float(np.abs((a[:, 3:] - b[:, 3:] + 180.0) % 360.0 - 180.0).max())
             for a, b in pose_pairs),
            default=0.0,
        ),
        "velocity_mps": max(
            (float(np.linalg.norm(a - b, axis=1).max())
             for a, b in combinations(velocities, 2)),
            default=0.0,
        ),
        "gap_m": float(np.ptp(gaps)),
    }


def match_starts(runs):
    """Compare every arm before and after the shared neutral warmup."""
    phases = {mode: run["initialization"] for mode, run in runs.items()}
    missing = [mode for mode, start in phases.items() if "control_start" not in start]
    reset = state_deltas(list(phases.values())) if phases else None
    control = None
    frame_deltas = {}
    time_deltas = {}
    if phases and not missing:
        control = state_deltas([start["control_start"] for start in phases.values()])
        frame_deltas = {
            mode: start["control_start"]["frame"] - start["frame"]
            for mode, start in phases.items()
        }
        time_deltas = {
            mode: start["control_start"]["simulation_time_s"] - start["simulation_time_s"]
            for mode, start in phases.items()
        }
    same_warmup = bool(frame_deltas) and len(set(frame_deltas.values())) == 1 and math.isclose(
        max(time_deltas.values()), min(time_deltas.values()), abs_tol=1e-4
    )
    # Reset may sample different vertical settling phases. Apply full pose,
    # velocity and gap tolerances at the handoff where control and labels begin.
    reset_layout = bool(reset and reset["same_actor_types"]
                        and reset["horizontal_position_m"] <= START_TOLERANCES["position_m"]
                        and reset["yaw_deg"] <= START_TOLERANCES["rotation_deg"])
    matched = same_warmup and reset_layout and control["same_actor_types"] and all(
        control[channel] <= limit for channel, limit in START_TOLERANCES.items()
    )
    return {
        "matched": matched,
        "matched_reset_layout": reset_layout,
        "criteria": "Same reset actor types, horizontal locations and yaw; same warmup; "
                    "full pose, velocity and gap tolerances at PID and label handoff",
        "reset": reset,
        "control_start": control,
        "missing_control_start": missing,
        "same_warmup": same_warmup,
        "warmup_ticks": frame_deltas,
        "warmup_seconds": time_deltas,
        "tolerances": START_TOLERANCES,
    }


def summarize_run(case, mode, run, matched):
    rows = run["rows"]
    flags = set(run["terminal_flags"])
    gaps = [row["post_step_true_gap_m"] for row in rows]
    target = run["target_gap_m"]
    decisions = [row for row in rows if row["probabilities"] is not None]
    tracking = [abs(gap - target) for gap in gaps]
    measurements = [rows[0]["true_gap_m"], *gaps] if gaps else []
    return {
        "case_id": case["case_id"],
        "condition": case["condition"],
        "source_route": case["route"]["name"],
        "seed": case["seed"],
        "lead_profile": case["lead_profile"],
        "lead_start_index": case["lead_start_index"],
        "mode": mode,
        "matched_start": matched,
        "initial_control_gap_m": run["initialization"].get("control_start", {}).get("initial_gap_m"),
        "observed_steps": len(rows),
        "duration_s": len(rows) * run["dt_s"],
        "terminal_reason": run["terminal_reason"],
        "clean_completion": flags == {"destination_reached"},
        "collision": "is_collision" in flags,
        "gap_limit_stop": "terminal_dist" in flags,
        "lane_or_route_stop": "out_of_lane" in flags,
        "step_or_time_limit": "time_exceeded" in flags or run["terminal_reason"] == "step_limit",
        "input_termination": run["terminal_reason"] in ("camera_unavailable", "invalid_prediction"),
        "warmup_terminal": run["warmup_terminal"],
        "min_true_gap_m": min(measurements) if measurements else None,
        "max_true_gap_m": max(measurements) if measurements else None,
        "tracking_mae_m": float(np.mean(tracking)) if tracking else None,
        "generalist_capture_gap_mae_m": float(np.mean([
            abs(row["generalist_gap_m"] - row["capture_true_gap_m"]) for row in rows
        ])) if rows else None,
        "active_capture_gap_mae_m": float(np.mean([
            abs(row["specialist_gap_m"] - row["capture_true_gap_m"]) for row in rows
        ])) if rows else None,
        "decisions": len(decisions),
        "switches": sum(row["controller_switched"] for row in rows),
        "below_advisory_threshold_decisions": sum(row["low_confidence"] for row in decisions),
        "fallback_use_steps": sum(row["fallback_used"] for row in rows),
    }


def aggregate(rows):
    result = {
        "runs": len(rows),
        "terminal_reasons": dict(Counter(row["terminal_reason"] for row in rows)),
    }
    counters = {
        "completions": "clean_completion",
        "collisions": "collision",
        "gap_limit_stops": "gap_limit_stop",
        "lane_or_route_stops": "lane_or_route_stop",
        "step_or_time_limits": "step_or_time_limit",
        "input_terminations": "input_termination",
        "switches": "switches",
        "decisions": "decisions",
        "below_advisory_threshold_decisions": "below_advisory_threshold_decisions",
        "fallback_use_steps": "fallback_use_steps",
    }
    result.update({name: sum(row[field] for row in rows) for name, field in counters.items()})
    result["fallback_runs"] = sum(bool(row["fallback_use_steps"]) for row in rows)
    tracking = [row["tracking_mae_m"] for row in rows if row["tracking_mae_m"] is not None]
    result["mean_run_tracking_mae_m"] = float(np.mean(tracking)) if tracking else None
    result["tracking_mae_runs"] = len(tracking)
    for key, operation in (("min_true_gap_m", min), ("max_true_gap_m", max)):
        result[key] = operation((row[key] for row in rows if row[key] is not None), default=None)
    return result


def aggregate_modes(rows):
    modes = ("selector", "generalist", "weather_matched")
    return {mode: aggregate([row for row in rows if row["mode"] == mode]) for mode in modes}


def paired_deltas(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["case_id"], {})[row["mode"]] = row
    result = {}
    for baseline in ("generalist", "weather_matched"):
        comparison = [(by_mode["selector"], by_mode[baseline]) for by_mode in grouped.values()]
        tracking_deltas = [
            first["tracking_mae_m"] - second["tracking_mae_m"]
            for first, second in comparison
            if first["tracking_mae_m"] is not None and second["tracking_mae_m"] is not None
        ]
        result[baseline] = {
            "pairs": len(comparison),
            "only_selector_finished": sum(
                first["clean_completion"] and not second["clean_completion"]
                for first, second in comparison
            ),
            "only_baseline_finished": sum(
                second["clean_completion"] and not first["clean_completion"]
                for first, second in comparison
            ),
            "both_finished": sum(
                first["clean_completion"] and second["clean_completion"]
                for first, second in comparison
            ),
            "mean_selector_minus_baseline_tracking_mae_m": (
                float(np.mean(tracking_deltas)) if tracking_deltas else None
            ),
            "pairs_with_tracking_mae": len(tracking_deltas),
        }
    return result


def plot_example(path, case, run, threshold):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = run["rows"]
    times = [row["time_s"] for row in rows]
    end_time = times[-1] + run["dt_s"]
    switch_times = [row["time_s"] for row in rows if row["controller_switched"]]
    present = [name for name in CONTROLLERS if any(row["controller"] == name for row in rows)]
    active = [present.index(row["controller"]) for row in rows]
    scores = [row["selected_probability"] for row in rows]
    true_gap = [row["true_gap_m"] for row in rows] + [rows[-1]["post_step_true_gap_m"]]
    fig, (gap_axis, selection_axis, score_axis) = plt.subplots(
        3, 1, sharex=True, figsize=(10, 6.8),
        gridspec_kw={"height_ratios": [3, 1, 1.5]}, layout="constrained",
    )
    gap_axis.plot(times + [end_time], true_gap, label="True gap", color="#152e53", linewidth=2.2)
    gap_axis.plot(times, [row["generalist_gap_m"] for row in rows],
                  label="Generalist estimate used for selection", color="#86548e", alpha=0.85)
    gap_axis.plot(times, [row["specialist_gap_m"] for row in rows],
                  label="Specialist estimate used by PID", color="#158276", linestyle="--")
    gap_axis.axhline(run["distance_limit_m"], color="#a03730", linestyle=":", label="Gap limit")
    gap_axis.axhline(run["target_gap_m"], color="#666666", linestyle=":", label="PID target")
    gap_axis.set_ylabel("Gap (m)")
    gap_axis.legend(fontsize=8, ncol=2)
    selection_axis.step(times + [end_time], active + [active[-1]], where="post", color="#158276")
    selection_axis.set_yticks(range(len(present)), [item.replace("_", " ") for item in present])
    selection_axis.set_ylim(-0.4, len(present) - 0.6)
    selection_axis.set_ylabel("Controller")
    score_axis.step(times + [end_time], scores + [scores[-1]], where="post", color="#86548e",
                    label="Selected specialist")
    if threshold is not None:
        score_axis.axhline(threshold, color="#666666", linestyle=":", label="Advisory threshold")
    score_axis.set_ylim(0, 1.02)
    score_axis.set_ylabel("Success probability")
    score_axis.set_xlabel("Simulation time after control handoff (s)")
    score_axis.legend(fontsize=8)
    for axis in (gap_axis, selection_axis, score_axis):
        axis.set_xlim(0, end_time)
        axis.grid(axis="x", color="#dddddd", linewidth=0.6)
        for time in switch_times:
            axis.axvline(time, color="#d18e40", linestyle="--", linewidth=1, alpha=0.8)
    fig.suptitle(
        f'CARLA logistic context selector · {case["condition"].replace("_", " ")}\n'
        f'{case["route"]["name"]}, seed {case["seed"]} · {run["terminal_reason"].replace("_", " ")}'
    )
    fig.savefig(path, dpi=175)
    plt.close(fig)


def write_live_report(root, pairs):
    root = Path(root)
    if not pairs:
        raise ValueError("No paired live rollouts to report")
    rows = [
        summarize_run(case, mode, run, match["matched"])
        for case, runs, match in pairs for mode, run in runs.items()
    ]
    with (root / "live_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    matched = [row for row in rows if row["matched_start"]]
    report = {
        "attempted_cases": len(pairs),
        "matched_cases": sum(match["matched"] for _, _, match in pairs),
        "unmatched_case_ids": [case["case_id"] for case, _, match in pairs if not match["matched"]],
        "method": "Adapted logistic context selector",
        "fallback_status": "Pending integration; scores below the advisory threshold still use argmax",
        "completion_rule": "Destination reached with no simultaneous failure flag",
        "tracking_mae_rule": "Within-run mean absolute post-step true-gap error from the PID target",
        "overall": aggregate_modes(matched),
        "all_attempts": aggregate_modes(rows),
        "by_condition": {
            name: aggregate_modes([row for row in matched if row["condition"] == name])
            for name in CONTROLLERS
        },
        "by_lead_profile": {
            name: aggregate_modes([row for row in matched if row["lead_profile"] == name])
            for name in sorted({row["lead_profile"] for row in rows})
        },
        "paired_comparisons": paired_deltas(matched),
        "start_checks": [{"case_id": case["case_id"], **match} for case, _, match in pairs],
    }
    candidates = [
        (case, runs["selector"]) for case, runs, match in pairs
        if match["matched"] and runs["selector"]["rows"]
    ]
    if candidates:
        case, run = max(
            candidates, key=lambda item: (
                sum(row["controller_switched"] for row in item[1]["rows"]), len(item[1]["rows"]),
            ),
        )
        from selector.model import Selector

        threshold = Selector.load(root / "model.json").threshold
        path = root / "selector_run.png"
        plot_example(path, case, run, threshold)
        report["example_run"] = {
            "case_id": case["case_id"], "plot": path.name,
            "trajectory": f'live/{case["case_id"]}/selector.json',
        }
    write_json(root / "live_report.json", report)
    return report
