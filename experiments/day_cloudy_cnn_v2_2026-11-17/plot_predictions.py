"""Plot matched v1/v2 prediction errors and the v2 training history."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


root = Path(__file__).resolve().parent / "finetune"
data = torch.load(root / "test_predictions.pth", map_location="cpu", weights_only=True)
scores = json.loads((root / "summary.json").read_text())
target = data["true_gap_m"]
limits = [min(target.min(), data["original"].min(), data["finetuned"].min()),
          max(target.max(), data["original"].max(), data["finetuned"].max())]

fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True, layout="constrained")
for ax, key, label in zip(axes, ("original", "finetuned"), ("CNN v1", "CNN v2")):
    ax.scatter(target, data[key], s=8, alpha=0.3)
    ax.plot(limits, limits, "--", color="black", linewidth=1)
    ax.axhline(10, color="tab:red", linestyle=":", linewidth=1)
    ax.axvline(10, color="tab:red", linestyle=":", linewidth=1)
    ax.set(title=f"{label} · MAE {scores[key]['test']['overall']['mae_m']:.2f} m",
           xlabel="Synchronized true gap (m)", aspect="equal", xlim=limits, ylim=limits)
axes[0].set_ylabel("Predicted gap (m)")
fig.savefig(root / "held_out_predictions.png", dpi=165)
plt.close(fig)

with (root / "epochs.csv").open(newline="") as handle:
    epochs = list(csv.DictReader(handle))
x = [int(row["epoch"]) for row in epochs]
fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True, layout="constrained")
axes[0].plot(x, [float(row["train_huber"]) for row in epochs], marker="o")
axes[0].set_ylabel("Training Huber loss")
axes[1].plot(x, [float(row["validation_score_m"]) for row in epochs],
             marker="o", label="Weighted validation MAE")
axes[1].axhline(scores["original_validation_score_m"], color="gray", linestyle="--", label="CNN v1")
axes[1].axvline(scores["best_epoch"], color="tab:red", linestyle=":", label="Selected epoch")
axes[1].set(xlabel="Epoch", ylabel="Validation error (m)")
axes[1].legend()
fig.savefig(root / "training_curves.png", dpi=165)
plt.close(fig)
