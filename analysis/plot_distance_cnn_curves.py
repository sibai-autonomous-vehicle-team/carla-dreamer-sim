"""Render training and validation curves from the distance CNN trainers."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.epochs.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No epochs in {args.epochs}")
    train_key = "train_huber" if "train_huber" in rows[0] else "train_huber_loss"
    epochs = [int(row["epoch"]) for row in rows]
    figure, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True, layout="constrained")
    axes[0].plot(epochs, [float(row[train_key]) for row in rows], marker="o")
    axes[0].set_ylabel("Training Huber loss")
    axes[1].plot(epochs, [float(row["validation_mae_m"]) for row in rows],
                 marker="o", label="Validation MAE")
    if "validation_score_m" in rows[0]:
        axes[1].plot(epochs, [float(row["validation_score_m"]) for row in rows],
                     marker=".", label="Selection score")
    axes[1].set(xlabel="Epoch", ylabel="Validation error (m)", xticks=epochs)
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
    figure.savefig(args.output, dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    main()
