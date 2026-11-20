"""Plot exact-frame close predictions for the old and new checkpoints."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

root = Path(__file__).resolve().parent / "finetune_v3"
predictions = torch.load(root / "test_predictions.pth", map_location="cpu", weights_only=True)
target = predictions["true_gap_m"]
close = target < 3
limits = (min(0, *(float(predictions[key][close].min()) for key in ("original", "finetuned"))),
          max(3, *(float(predictions[key][close].max()) for key in ("original", "finetuned"))))
figure, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9, 4.5), layout="constrained")
for ax, key, label in zip(axes, ("original", "finetuned"), ("CNN v2", "CNN v3")):
    gap = target[close]
    estimates = predictions[key][close]
    mae = float((estimates-gap).abs().mean())
    ax.scatter(gap, estimates, s=10, alpha=0.25)
    ax.plot([0, 3], [0, 3], "--", color="gray", linewidth=1)
    ax.axhline(0, color="tab:red", linestyle=":", linewidth=1)
    ax.set(title=f"{label} · MAE {mae:.2f} m", xlabel="Synchronized true gap (m)",
           xlim=(0, 3.05), ylim=limits)
axes[0].set_ylabel("Raw predicted gap (m)")
figure.suptitle(f"{int(close.sum())} held-out test frames below 3 m")
figure.savefig(root / "close_predictions.png", dpi=170)
plt.close(figure)

with (root / "epochs.csv").open(newline="") as handle:
    epochs = list(csv.DictReader(handle))
values = [int(row["epoch"]) for row in epochs]
figure, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 6), layout="constrained")
axes[0].plot(values, [float(row["train_huber"]) for row in epochs], marker="o")
axes[0].set_ylabel("Training Huber loss")
axes[1].plot(values, [float(row["validation_score_m"]) for row in epochs], marker="o")
axes[1].set(xlabel="Epoch", ylabel="Validation score (m)")
figure.savefig(root / "training_curves.png", dpi=155)
plt.close(figure)
