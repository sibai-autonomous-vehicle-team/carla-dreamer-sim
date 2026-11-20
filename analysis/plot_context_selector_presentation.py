"""Render the paired CARLA pilot figures from saved artifacts."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np

from analysis.draw_gap_cnn_diagram import draw_cnn_diagram
from selector.model import CONTROLLERS, Selector
from selector.reporting import match_starts, summarize_run


EXPERIMENT = Path("experiments/logistic_context_selector_2026-11-19")
METHODS = ("generalist", "weather_matched", "selector")
NAMES = {
    "generalist": "Generalist CNN + PID",
    "weather_matched": "Direct weather/time match",
    "selector": "Learned logistic selector",
}
DISPLAY = {
    "generalist": "Generalist CNN\n+ PID",
    "weather_matched": "Direct weather/time\nmatch",
    "selector": "Learned logistic\nselector",
}
COLORS = {"generalist": "#28517A", "weather_matched": "#007F78", "selector": "#B94F35"}
INK = "#1B365D"
MUTED = "#4A5B6D"
GRID = "#DCE4EA"
BG = "#FFFFFF"
FLAG_FIELDS = {"is_collision": "collision", "terminal_dist": "gap_limit", "out_of_lane": "lane_route"}
CAUSES = ("collision", "gap_limit", "lane_route", "timeout_other")
CAUSE_LABELS = ("Collision", "True gap\nabove 10 m", "Lane/route\ndeparture", "Timeout or\nother stop")
INFRASTRUCTURE_REASONS = {
    "camera_unavailable", "invalid_prediction", "empty_camera", "simulator_error",
}


def read_json(path):
    return json.loads(path.read_text())


def write_csv(path, records):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def load_pairs(root):
    plan_path = root / "plan.json"
    plan = read_json(plan_path)
    settings = read_json(root / "settings.json")
    report = read_json(root / "live_report.json")
    if hashlib.sha256(plan_path.read_bytes()).hexdigest() != settings["plan_sha256"]:
        raise ValueError("Plan checksum differs from the recorded experiment settings")
    if (settings["target_gap_m"], settings["distance_limit_m"]) != (6.0, 10.0):
        raise ValueError("The publication comparison requires the 6 m / 10 m pilot settings")
    with (root / "live_results.csv").open(newline="") as handle:
        source_csv = {(row["case_id"], row["mode"]): row for row in csv.DictReader(handle)}

    pairs, records = [], []
    model = Selector.load(root / "model.json")
    for case in (row for row in plan["cases"] if row["split"] == "test"):
        case_id = case["case_id"]
        runs = {mode: read_json(root / "live" / case_id / f"{mode}.json") for mode in METHODS}
        reference = runs[METHODS[0]]
        if not match_starts(runs)["matched"]:
            raise ValueError(f"Starting states do not match in {case_id}")
        for mode, run in runs.items():
            if run["case"] != case or run["mode"] != mode:
                raise ValueError(f"Trace does not match the manifest for {case_id} {mode}")
            for key in ("dt_s", "horizon_steps", "target_gap_m", "distance_limit_m", "lane_limit_m",
                        "warmup_steps", "gains", "route_centerline_xy"):
                if run[key] != reference[key]:
                    raise ValueError(f"{key} differs between arms in {case_id}")
            for key in ("target_gap_m", "distance_limit_m", "lane_limit_m", "warmup_steps"):
                if run[key] != settings[key]:
                    raise ValueError(f"Unexpected {key} in {case_id}")
            if run["gains"] != settings["pid_gains"]:
                raise ValueError(f"Unexpected PID gains in {case_id}")
            base = summarize_run(case, mode, run, True)
            index = source_csv[case_id, mode]
            for key in ("terminal_reason", "clean_completion", "collision", "gap_limit_stop",
                        "lane_or_route_stop", "observed_steps", "matched_start", "switches"):
                if index[key] != str(base[key]):
                    raise ValueError(f"Run CSV disagrees on {key} in {case_id} {mode}")
            flags = set(run["terminal_flags"])
            violations = {field: flag in flags for flag, field in FLAG_FIELDS.items()}
            infrastructure = (run["terminal_reason"] in INFRASTRUCTURE_REASONS
                              or base["input_termination"])
            completion = flags == {"destination_reached"}
            any_violation = any(violations.values())
            max_gap = base["max_true_gap_m"]
            crossed_limit = max_gap is not None and max_gap > run["distance_limit_m"]
            if crossed_limit != violations["gap_limit"]:
                raise ValueError(f"True-gap history disagrees with terminal flag in {case_id}")
            records.append({
                "case_id": case_id, "condition": case["condition"], "route": case["route"]["name"],
                "seed": case["seed"], "lead_profile": case["lead_profile"],
                "lead_start_index": case["lead_start_index"], "method_id": mode, "method": NAMES[mode],
                "completion": completion, "any_specification_violation": any_violation,
                **violations, "timeout_other": not (completion or any_violation or infrastructure),
                "simulator_camera_error": infrastructure, "terminal_reason": run["terminal_reason"],
                "terminal_flags": " | ".join(run["terminal_flags"]),
                "target_gap_m": run["target_gap_m"], "upper_gap_m": run["distance_limit_m"],
                "duration_s": base["duration_s"],
                "source_trace": f"live/{case_id}/{mode}.json",
            })
        pairs.append((case, runs))
    if len(pairs) != report["matched_cases"] or len(source_csv) != len(records):
        raise ValueError("Paired run counts disagree with the report or run CSV")
    for mode in METHODS:
        group = [row for row in records if row["method_id"] == mode]
        summary = report["overall"][mode]
        for key, report_key in (("completion", "completions"), ("collision", "collisions"),
                                ("gap_limit", "gap_limit_stops"), ("lane_route", "lane_or_route_stops")):
            if sum(row[key] for row in group) != summary[report_key]:
                raise ValueError(f"Derived counts disagree with {report_key} for {mode}")
    return pairs, records, model


def rate_rows(records, grouping, fields):
    summaries = []
    for name, group in grouping:
        for mode in METHODS:
            subset = [row for row in group if row["method_id"] == mode]
            for key in fields:
                total = len(subset)
                count = sum(row[key] for row in subset) if total else None
                summaries.append({
                    "group": name, "method_id": mode, "method": NAMES[mode], "metric": key,
                    "count": count, "total": total or None, "percent": 100 * count / total if total else None,
                    "status": "Observed" if total else "Pending",
                })
    return summaries


def heading(fig, title, subtitle, caption, left=0.085):
    fig.text(left, 0.964, title, ha="left", va="top", fontsize=22, fontweight="bold", color=INK)
    fig.text(left, 0.901, subtitle, ha="left", va="top", fontsize=11.5, color=MUTED)
    fig.text(left, 0.031, caption, ha="left", va="bottom", fontsize=9.5, color=MUTED,
             linespacing=1.35)


def style_axis(ax, grid_axis="y"):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis=grid_axis, color=GRID, lw=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, pad=6)


def plot_main(rows, n):
    fig, ax = plt.subplots(figsize=(11, 5.9))
    fig.subplots_adjust(left=0.1, right=0.96, top=0.77, bottom=0.22)
    heading(fig, "Route completion and specification violations",
            "Full-route outcomes on matched CARLA Town03 starts",
            f"{n} paired cases per method ({3*n} rollouts) · PID target 6 m · True-gap cutoff >10 m.\n"
            "Violation = collision, true gap above the cutoff, or lane/route departure.\n"
            "Both specialist methods share seven fine-tuned checkpoints and two saved before fine-tuning.", left=0.1)
    style_axis(ax)
    centers = np.arange(len(METHODS))
    for row in rows:
        mode = row["method_id"]
        completed = row["metric"] == "completion"
        x = centers[METHODS.index(mode)] + (-0.182 if completed else 0.182)
        if row["percent"] is None:
            ax.text(x, 4, "Pending", ha="center", color=MUTED, fontsize=10)
            continue
        ax.bar(x, row["percent"], width=0.33, color=COLORS[mode] if completed else BG,
               edgecolor=COLORS[mode], hatch=None if completed else "///", lw=1.25)
        ax.text(x, row["percent"] + 2.4, f"{row['percent']:.1f}%\n{row['count']}/{row['total']}",
                ha="center", va="bottom", color=INK, fontweight="bold", fontsize=12)
    ax.set_xticks(centers, [DISPLAY[mode] for mode in METHODS], fontweight="bold", fontsize=11.5)
    for tick, mode in zip(ax.get_xticklabels(), METHODS):
        tick.set_color(COLORS[mode])
    ax.set(ylim=(0, 100), yticks=range(0, 101, 20), ylabel="Share of runs")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.legend(handles=[
        Patch(facecolor=INK, edgecolor=INK, label="Clean route completion"),
        Patch(facecolor=BG, edgecolor=INK, hatch="///", label="At least one specification violation"),
    ], ncol=2, loc="lower left", bbox_to_anchor=(0, 1.015), borderaxespad=0)
    return fig


def plot_failures(rows, n):
    fig, (ax, infra) = plt.subplots(ncols=2, figsize=(11, 5.9),
                                     gridspec_kw={"width_ratios": [3.5, 1.32]})
    fig.subplots_adjust(left=0.09, right=0.975, top=0.74, bottom=0.25, wspace=0.31)
    heading(fig, "Why episodes ended early",
            f"Terminal causes in the same {n} paired full-route cases per method",
            "Each simultaneous specification flag counts in its own category. No simultaneous flags occurred.\n"
            "Timeout or other denotes task endings without a specification flag. Infrastructure errors are shown separately.",
            left=0.09)
    style_axis(ax)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    for mode_index, mode in enumerate(METHODS):
        values = {row["metric"]: row["count"] for row in rows if row["method_id"] == mode}
        for i, cause in enumerate(CAUSES):
            x = i + (mode_index - 1) * 0.245
            value = values[cause]
            if value is None:
                ax.text(x, 0.43, "Pending", rotation=90, ha="center", color=MUTED, fontsize=8)
                continue
            ax.bar(x, value, width=0.22, color=COLORS[mode])
            ax.text(x, value + 0.28, str(value), ha="center", va="bottom", fontsize=10,
                    color=COLORS[mode], fontweight="bold")
    ax.set_xticks(range(len(CAUSES)), CAUSE_LABELS, fontsize=10.5)
    max_count = max((row["count"] or 0 for row in rows if row["metric"] in CAUSES), default=0)
    ax.set(ylim=(0, max_count + 2), ylabel="Episodes")
    handles = [Patch(facecolor=COLORS[mode], label=NAMES[mode]) for mode in METHODS]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.05), ncol=3, borderaxespad=0)
    infra.set_axis_off()
    infra.set_facecolor("#F1F6F8")
    infra.add_patch(plt.Rectangle((0, 0.07), 1, 0.88, facecolor="#F1F6F8", edgecolor="none",
                                  transform=infra.transAxes))
    infra.text(0.08, 0.87, "Simulator and\ncamera errors", transform=infra.transAxes, color=INK,
               fontsize=11.2, fontweight="bold", va="top", linespacing=1.15)
    for i, mode in enumerate(METHODS):
        value = next(row for row in rows if row["method_id"] == mode
                     and row["metric"] == "simulator_camera_error")
        label = "Pending" if value["total"] is None else f"{value['count']}/{value['total']}"
        y = 0.64 - 0.20 * i
        infra.text(0.08, y, NAMES[mode], transform=infra.transAxes, color=COLORS[mode],
                   fontsize=9.0, fontweight="bold", va="top")
        infra.text(0.08, y - 0.08, label, transform=infra.transAxes, color=INK, fontsize=10, va="top")
    return fig


def plot_heatmap(rows):
    fig, ax = plt.subplots(figsize=(9.8, 7.6))
    fig.subplots_adjust(left=0.22, right=0.88, top=0.78, bottom=0.12)
    heading(fig, "Route completion by lighting and weather",
            "Three paired held-out route cases in each condition",
            "Cells report clean completions / trials. The color scale is shared across all methods and conditions.",
            left=0.13)
    lookup = {(row["group"], row["method_id"]): row for row in rows}
    values = np.array([[lookup[condition, mode]["percent"] for mode in METHODS]
                       for condition in CONTROLLERS], dtype=float)
    cmap = LinearSegmentedColormap.from_list("completion", ("#F1F6F8", "#8CBCC7", "#246686", "#183653"))
    cmap.set_bad("#EBEEF1")
    columns = np.arange(len(METHODS) + 1) - 0.5
    row_edges = np.arange(len(CONTROLLERS) + 1) - 0.5
    img = ax.pcolormesh(columns, row_edges, np.ma.masked_invalid(values), cmap=cmap,
                        vmin=0, vmax=100, edgecolors=BG, linewidth=1.5)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(len(METHODS)), [DISPLAY[mode] for mode in METHODS], fontweight="bold", fontsize=11)
    ax.set_yticks(np.arange(len(CONTROLLERS)), [item.replace("_", " ").capitalize() for item in CONTROLLERS],
                  fontsize=11)
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", length=0, pad=8)
    for tick, mode in zip(ax.get_xticklabels(), METHODS):
        tick.set_color(COLORS[mode])
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, condition in enumerate(CONTROLLERS):
        for j, mode in enumerate(METHODS):
            row = lookup[condition, mode]
            label = ("Pending" if row["total"] is None
                     else f"{row['percent']:.0f}%  ({row['count']}/{row['total']})")
            ax.text(j, i, label, ha="center", va="center", fontsize=13, fontweight="bold",
                    color=BG if values[i, j] >= 64 else INK)
    colorbar = fig.colorbar(img, ax=ax, fraction=0.045, pad=0.025, ticks=[0, 25, 50, 75, 100])
    colorbar.solids.set_rasterized(False)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=0)
    colorbar.set_label("Completion rate", labelpad=10)
    return fig


def paired_trace(case, runs):
    records = []
    for mode, run in runs.items():
        for row in run["rows"]:
            records.append({
                "case_id": case["case_id"], "condition": case["condition"], "seed": case["seed"],
                "lead_profile": case["lead_profile"], "method_id": mode, "method": NAMES[mode],
                "time_s": row["time_s"], "true_gap_m": row["true_gap_m"],
                "target_gap_m": run["target_gap_m"], "upper_gap_m": run["distance_limit_m"],
                "selected_specialist": row["controller"] if mode == "selector" else None,
                "predicted_success_probability": row["selected_probability"],
                "selector_decision": row["probabilities"] is not None,
                "controller_switched": row["controller_switched"], "terminal_reason": None,
            })
        last = run["rows"][-1]
        records.append({
            **records[-1], "time_s": last["time_s"] + run["dt_s"],
            "true_gap_m": last["post_step_true_gap_m"], "selector_decision": False,
            "controller_switched": False, "terminal_reason": run["terminal_reason"],
        })
    return records


def plot_paired(case, records, threshold):
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11.1, 8.5),
                             gridspec_kw={"height_ratios": (3.3, 1.25, 1.55)})
    fig.subplots_adjust(left=0.15, right=0.95, top=0.795, bottom=0.13, hspace=0.27)
    gap, choice, probability = axes
    selector_rows = [row for row in records if row["method_id"] == "selector"]
    times = [row["time_s"] for row in selector_rows]
    switches = [row["time_s"] for row in selector_rows if row["controller_switched"]]
    heading(fig, "Gap tracking on a paired Town03 turn",
            f"{case['condition'].replace('_', ' ').capitalize()} · Seed {case['seed']} · Brake/restart lead",
            "Circles mark finishes. The cross marks the gap-limit stop. Vertical dashed lines mark specialist switches.\n"
            "The validation-selected threshold is advisory. Fallback is not implemented.", left=0.15)
    for ax in axes:
        style_axis(ax)
        for t in switches:
            ax.axvline(t, color="#B6A59E", linestyle=(0, (3, 3)), alpha=0.65, lw=0.9, zorder=1)
    actual_end = max(row["time_s"] for row in records)
    for value, label, color in ((6, "6 m target", MUTED), (10, "10 m limit", "#9D4534")):
        gap.axhline(value, color=color, linestyle=":", lw=1.25)
        gap.text(actual_end - 0.12, value + 0.11, label, color=color, ha="right", va="bottom", fontsize=9)
    for mode, style in zip(METHODS, ("--", "-.", "-")):
        data = [row for row in records if row["method_id"] == mode]
        end = data[-1]
        gap.plot([row["time_s"] for row in data], [row["true_gap_m"] for row in data],
                 label=NAMES[mode], color=COLORS[mode], linestyle=style, lw=2.0, zorder=3)
        marker = "o" if end["terminal_reason"] == "destination_reached" else "X"
        gap.scatter(end["time_s"], end["true_gap_m"], color=COLORS[mode], marker=marker,
                    edgecolor=BG, lw=0.6, s=56, zorder=4)
    gap.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, borderaxespad=0, fontsize=10)
    gap.set_ylabel("True gap (m)")
    gap.set_ylim(bottom=0, top=10.9)
    present = list(dict.fromkeys(row["selected_specialist"] for row in selector_rows))
    active = [present.index(row["selected_specialist"]) for row in selector_rows]
    choice.step(times, active, where="post", color=COLORS["selector"], lw=2.1)
    choice.set_yticks(range(len(present)), [name.replace("_", " ") for name in present], fontsize=10)
    choice.set(ylim=(-0.45, len(present) - 0.55), ylabel="Specialist")
    scores = [row["predicted_success_probability"] for row in selector_rows]
    probability.step(times, scores, where="post", color=COLORS["selector"], lw=2.1)
    probability.axhline(threshold, linestyle=":", lw=1.25, color=MUTED)
    probability.text(actual_end - 0.12, threshold + 0.012, f"Advisory threshold {threshold:.3f}",
                     ha="right", va="bottom", fontsize=9, color=MUTED)
    lower_score_limit = max(0, math.floor((min(min(scores), threshold) - 0.06) * 10) / 10)
    probability.set(ylim=(lower_score_limit, 1.02),
                    ylabel="Predicted success\nprobability", xlabel="Simulation time after control handoff (s)",
                    xlim=(0, actual_end + 0.18))
    probability.yaxis.set_major_locator(MaxNLocator(4))
    return fig


def fallback_report(pairs, threshold):
    decisions = [[row for row in runs["selector"]["rows"] if row["probabilities"] is not None]
                 for _, runs in pairs]
    low_scores = [[row for row in group if row["selected_probability"] < threshold] for group in decisions]
    if any(not row["low_confidence"] for group in low_scores for row in group):
        raise ValueError("Saved low-confidence flags disagree with the validation threshold")
    records = [
        {"scope": "Actual fallback", "metric": metric, "status": "Not implemented",
         "count": None, "total": None, "percent": None}
        for metric in ("Fallback activations", "Runs needing fallback (%)", "Driving time in fallback (%)")
    ]
    for metric, count, total in (
        ("Decision scores below threshold", sum(map(len, low_scores)), sum(map(len, decisions))),
        ("Runs with at least one advisory request", sum(bool(group) for group in low_scores), len(pairs)),
    ):
        records.append({"scope": "Advisory only", "metric": metric, "status": "Observed",
                        "count": count, "total": total, "percent": 100 * count / total})
    return records


def plot_fallback(records, threshold):
    fig = plt.figure(figsize=(10.8, 5.55))
    heading(fig, "Fallback integration status",
            "Intended rule · use a default controller if no specialist reaches the acceptance threshold",
            "Advisory threshold crossings indicate low model confidence. Specification violations are measured separately.")
    for i, row in enumerate(records[:3]):
        y = 0.77 - 0.09 * i
        fig.add_artist(plt.Rectangle((0.085, y - 0.057), 0.83, 0.076, facecolor="#F0F5F7",
                                     transform=fig.transFigure))
        fig.text(0.104, y - 0.017, row["metric"], color=INK, fontsize=12, va="center")
        fig.text(0.895, y - 0.017, row["status"], color=MUTED, fontweight="bold", fontsize=12,
                 ha="right", va="center")
    fig.text(0.085, 0.435, f"Advisory only · validation threshold {threshold:.3f}", color=INK,
             fontsize=13, fontweight="bold")
    for i, row in enumerate(records[3:]):
        y = 0.35 - i * 0.11
        fig.text(0.085, y, row["metric"], color=INK, fontsize=12, va="center")
        fig.text(0.895, y, f"{row['count']}/{row['total']}  ({row['percent']:.1f}%)",
                 color=COLORS["selector"], fontsize=13, fontweight="bold", ha="right", va="center")
    return fig


def window_summary(root, pairs):
    fit = read_json(root / "model_report.json")["test"]["matched"]
    records = []
    for mode, field in (("generalist", None), ("weather_matched", "weather_matched_successes"),
                        ("selector", "selected_successes")):
        count = fit[field] if field else None
        total = fit["complete_cases"] if field else None
        records.append({"scope": "Fixed-specialist matched-start test", "method": NAMES[mode],
                        "successes": count, "total": total, "status": "Observed" if field else "Pending"})
    for mode in METHODS:
        counts = Counter()
        for _, runs in pairs:
            run = runs[mode]
            steps = run["horizon_steps"]
            window = run["rows"][:steps]
            violation = any(set(row["flags"]) & FLAG_FIELDS.keys() for row in window)
            outcome = "failure" if violation else "success" if len(window) == steps else "excluded"
            counts[outcome] += 1
        records.append({"scope": "First 2 seconds of paired live runs", "method": NAMES[mode],
                        "successes": counts["success"], "total": counts["success"] + counts["failure"],
                        "status": "Observed"})
    return records


def save_figure(fig, stem):
    fig.savefig(stem.with_suffix(".svg"), facecolor=BG)
    fig.savefig(stem.with_suffix(".png"), dpi=360, facecolor=BG)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=EXPERIMENT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.experiment
    output = args.output or root / "presentation"
    pairs, episodes, model = load_pairs(root)
    overall = [("All nine conditions", episodes)]
    main_rows = rate_rows(episodes, overall, ("completion", "any_specification_violation"))
    causes = rate_rows(episodes, overall, (*CAUSES, "simulator_camera_error"))
    conditions = [(name, [row for row in episodes if row["condition"] == name]) for name in CONTROLLERS]
    by_condition = rate_rows(episodes, conditions, ("completion",))
    case, runs = max(pairs, key=lambda item: (
        sum(row["controller_switched"] for row in item[1]["selector"]["rows"]),
        len(item[1]["selector"]["rows"]),
    ))
    trace = paired_trace(case, runs)
    threshold = model.threshold
    fallback = fallback_report(pairs, threshold)
    output.mkdir(parents=True, exist_ok=True)
    for name, data in (
        ("paired_episode_summary", episodes), ("figure_1_overall", main_rows),
        ("figure_2_terminal_causes", causes), ("figure_3_by_condition", by_condition),
        ("figure_4_paired_trace", trace), ("fallback_status", fallback),
        ("two_second_outcomes", window_summary(root, pairs)),
    ):
        write_csv(output / f"{name}.csv", data)
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK, "axes.edgecolor": GRID,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "legend.frameon": False,
        "figure.facecolor": BG, "axes.facecolor": BG, "svg.fonttype": "none",
        "svg.hashsalt": "carla_context_selector_presentation",
    }):
        for name, render, inputs in (
            ("figure_1_overall", plot_main, (main_rows, len(pairs))),
            ("figure_2_terminal_causes", plot_failures, (causes, len(pairs))),
            ("figure_3_by_condition", plot_heatmap, (by_condition,)),
            ("figure_4_paired_run", plot_paired, (case, trace, threshold)),
            ("fallback_status", plot_fallback, (fallback, threshold)),
            ("cnn_gap_regressor", draw_cnn_diagram, ()),
        ):
            save_figure(render(*inputs), output / name)
    print(f"Rendered six figures from {len(pairs)} matched cases per method at {output}")


if __name__ == "__main__":
    main()
