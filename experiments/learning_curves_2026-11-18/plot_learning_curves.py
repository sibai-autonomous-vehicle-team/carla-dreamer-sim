"""Plot logged validation MAE and observed held-out test MAE."""

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


OUTPUT = Path(__file__).resolve().parent
EXPERIMENTS = OUTPUT.parent
REPO = EXPERIMENTS.parent
BATCH = EXPERIMENTS / "all_specialists_2026-11-17"
DAY = EXPERIMENTS / "day_cloudy_resnet18_pid_2026-11-17"
SUNSET = EXPERIMENTS / "sunset_cloudy_specialist_2026-11-17"
GENERALIST = EXPERIMENTS / "generalist_all_conditions_2026-11-18"
DAY_V2 = EXPERIMENTS / "day_cloudy_cnn_v2_2026-11-17" / "finetune"
DAY_V3 = EXPERIMENTS / "day_cloudy_cnn_close_v3_2026-11-17" / "finetune_v3"

CONDITIONS = tuple(f"{time}_{weather}" for time in ("day", "sunset", "night")
                   for weather in ("sunny", "cloudy", "rainy"))
VALIDATION = "#2563A0"
TEST = "#AE5914"
DIAGNOSTIC = "#A33958"


@dataclass
class Curve:
    title: str
    validation: dict[int, float]
    test: dict[int, float]
    diagnostic: dict[int, float]
    note: str = ""


def read_json(path):
    return json.loads(path.read_text())


def validation_history(folder):
    with (folder / "epochs.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {int(row["epoch"]): float(row["validation_mae_m"]) for row in rows}
    if list(result) != list(range(1, len(rows) + 1)) or not result:
        raise ValueError(f"Expected consecutive epochs in {folder / 'epochs.csv'}")
    return result


def archive_curve(condition):
    if condition == "day_cloudy":
        folder, test_folder = DAY, DAY
    else:
        parent = SUNSET if condition == "sunset_cloudy" else BATCH / condition
        folder, test_folder = parent / "archive", parent / "archive_evaluation"
    validation = validation_history(folder)
    score = read_json(test_folder / "test_metrics.json")
    epoch = score["checkpoint_epoch"]
    if epoch != min(validation, key=validation.get):
        raise ValueError(f"Archive test score and validation checkpoint differ for {condition}")
    return Curve(condition.replace("_", " ").title(), validation, {epoch: score["mae_m"]}, {})


def generalist_curve():
    validation = validation_history(GENERALIST / "training")
    score = read_json(GENERALIST / "summary.json")["pooled_generalist"]["mae_m"]
    epoch = read_json(REPO / "generalist" / "cnn.json")["selected_epoch"]
    test_epochs = {read_json(path)["checkpoint_epoch"]
                   for path in (GENERALIST / "evaluation").glob("*/test_metrics.json")}
    if test_epochs != {epoch} or epoch != min(validation, key=validation.get):
        raise ValueError("Generalist test score and selected validation checkpoint differ")
    return Curve("Generalist · all nine", validation, {epoch: score}, {})


def synchronized_curve(folder, title, extra=None):
    summary = read_json(folder / "summary.json")
    start, candidate = (summary[key] for key in ("original", "finetuned"))
    validation = {0: start["validation"]["overall"]["mae_m"], **validation_history(folder)}
    epoch = summary["best_epoch"]
    if not math.isclose(validation[epoch], candidate["validation"]["overall"]["mae_m"], abs_tol=1e-6):
        raise ValueError(f"Synchronized test checkpoint has a different validation MAE in {folder}")
    points = {0: start["test"]["overall"]["mae_m"]}
    diagnostics = {}
    note = ""
    if summary.get("selection_status") == "no_eligible_checkpoint":
        diagnostics[epoch] = candidate["test"]["overall"]["mae_m"]
        note = "Archive checkpoint retained"
    else:
        points[epoch] = candidate["test"]["overall"]["mae_m"]
    if extra is not None and extra.is_file():
        data = read_json(extra)
        if data["test_frames"] != candidate["test"]["overall"]["frames"] or data["epoch"] not in validation:
            raise ValueError(f"Extra diagnostic test is incompatible with {folder}")
        diagnostics[data["epoch"]] = data["mae_m"]
    return Curve(title, validation, points, diagnostics, note)


def draw_curves(curves, title, x_label, output):
    fig, axes = plt.subplots(2, 5, figsize=(17.8, 8.1), sharey=True, facecolor="#F5F7FA")
    maximum = max(max((*curve.validation.values(), *curve.test.values(), *curve.diagnostic.values()))
                  for curve in curves)
    top = math.ceil((maximum + 0.05) * 4) / 4
    has_diagnostic = any(curve.diagnostic for curve in curves)
    for ax, curve in zip(axes.flat, curves):
        ax.set_title(curve.title, loc="left", pad=11, fontsize=12, fontweight="semibold")
        ax.plot(list(curve.validation), list(curve.validation.values()), color=VALIDATION,
                marker="o", markersize=3.5, linewidth=1.8)
        ax.grid(axis="y", color="#DFE5ED", linewidth=0.85)
        ax.set_axisbelow(True)
        for points, color, marker in ((curve.test, TEST, "D"), (curve.diagnostic, DIAGNOSTIC, "X")):
            ax.scatter(list(points), list(points.values()), s=67, marker=marker, color=color,
                       edgecolor="white", linewidth=0.8, zorder=4)
            for epoch, error in points.items():
                tail = epoch >= (max(curve.validation) - 1)
                above = error >= curve.validation[epoch] and error + 0.15 < top
                ax.annotate(f"{error:.3f}", (epoch, error),
                            xytext=(-6 if tail else 6, 9 if above else -11),
                            textcoords="offset points", ha="right" if tail else "left",
                            va="bottom" if above else "top", color=color, fontsize=9.3,
                            fontweight="semibold",
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.84, pad=0.8))
        start, end = min(curve.validation), max(curve.validation)
        ticks = {10: (1, 4, 7, 10), 16: (0, 4, 8, 12, 16), 20: (0, 5, 10, 15, 20)}
        ax.set_xticks(ticks.get(end, (start, end)))
        ax.set(xlim=(start - 0.55, end + 0.55), ylim=(0, top))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.tick_params(length=3, labelsize=9, color="#808BA0", labelcolor="#344054")
        ax.spines[["top", "right"]].set_visible(False)
        for side in ("bottom", "left"):
            ax.spines[side].set_color("#BCC6D5")
        if curve.note:
            ax.text(0.04, 0.055, curve.note, transform=ax.transAxes, color=DIAGNOSTIC,
                    fontsize=8.8)

    fig.suptitle(title, x=0.064, y=0.949, ha="left", fontsize=24, fontweight="semibold", color="#172840")
    legend = [
        Line2D([0], [0], color=VALIDATION, marker="o", label="Validation MAE · every epoch"),
        Line2D([0], [0], color=TEST, marker="D", linestyle="none", label="Held-out test MAE · checkpoint"),
    ]
    if has_diagnostic:
        legend.append(Line2D([0], [0], color=DIAGNOSTIC, marker="X", linestyle="none",
                             label="Diagnostic checkpoint · held-out test"))
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.058, 0.884), frameon=False,
               ncol=len(legend), fontsize=11, handletextpad=0.5, columnspacing=2.6)
    fig.supylabel("Gap MAE (m)", x=0.022, fontsize=12, color="#344054")
    fig.supxlabel(x_label, y=0.038, fontsize=12, color="#344054")
    fig.subplots_adjust(left=0.065, right=0.98, bottom=0.095, top=0.807, wspace=0.24, hspace=0.39)
    fig.savefig(OUTPUT / output, dpi=165)
    plt.close(fig)


def main():
    archive = [archive_curve(condition) for condition in CONDITIONS]
    archive.append(generalist_curve())
    draw_curves(archive, "Archive training curves", "Archive training epoch",
                "archive_learning_curves.png")

    fine_tune = []
    for condition in CONDITIONS:
        title = condition.replace("_", " ").title()
        if condition == "day_cloudy":
            fine_tune.extend([synchronized_curve(DAY_V2, title + " · V2"),
                              synchronized_curve(DAY_V3, title + " · V3")])
        else:
            parent = SUNSET if condition == "sunset_cloudy" else BATCH / condition
            extra = OUTPUT / "sunset_sunny_diagnostic.json" if condition == "sunset_sunny" else None
            fine_tune.append(synchronized_curve(parent / "sync_model", title, extra))
    draw_curves(fine_tune, "Synchronized specialist fine-tuning",
                "Synchronized training epoch · 0 is the starting checkpoint",
                "synchronized_learning_curves.png")


if __name__ == "__main__":
    main()
