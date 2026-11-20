"""Predict the recorded controller's finite-horizon minimum buffer margin."""
import argparse
import json
import math
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch.nn.functional import mse_loss

from .config import LOCAL_DIR, Settings, load_data, setup_device
from .dataset import risk_examples, safety_margin, split_episodes
from .models import RiskNet


def gap_forecast(history, cfg):
    """Extend the mean recent change per step and absorb the first predicted crossing."""
    gap = history[:, -1]
    change = torch.zeros_like(gap)
    if history.shape[1] > 1:
        change = (gap - history[:, 0]) / (history.shape[1] - 1)

    margin = safety_margin(gap, cfg.safety_buffer_m)
    minimum = margin.clone()
    for step in range(1, cfg.horizon_steps):
        next_gap = (gap + step * change).clamp_min(0)
        next_margin = safety_margin(next_gap, cfg.safety_buffer_m)
        minimum = torch.where(minimum > 0, torch.minimum(minimum, next_margin), minimum)
    return minimum


def fit_risk_model(model, train_x, train_y, val_x, val_y, cfg):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.risk_lr)
    losses = []
    best_epoch = 0
    best_loss = math.inf
    for epoch in range(1, cfg.risk_epochs + 1):
        model.train()
        squared_error = torch.zeros((), device=train_x.device)
        order = torch.randperm(len(train_x), device=train_x.device)
        for indices in order.split(cfg.batch_size):
            prediction = model(train_x[indices]).reshape(-1)
            loss = mse_loss(prediction, train_y[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            squared_error += loss.detach() * len(indices)

        train_loss = squared_error.item() / len(train_x)
        val_loss = mse_loss(predict(model, val_x, cfg.batch_size), val_y).item()
        if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
            raise ValueError("Nonfinite risk loss. Check the numerical data and learning rate.")
        losses.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss})
        print(f"Epoch {epoch:3d}   train MSE {train_loss:.5f}   val MSE {val_loss:.5f}", flush=True)
        if val_loss < best_loss:
            best_epoch, best_loss = epoch, val_loss
            best_weights = deepcopy(model.state_dict())
        elif epoch - best_epoch >= cfg.patience:
            break

    model.load_state_dict(best_weights)
    return best_epoch, losses


@torch.no_grad()
def predict(model, inputs, batch_size):
    model.eval()
    return torch.cat([model(batch).reshape(-1) for batch in inputs.split(batch_size)])


def score(target, prediction, boundary_band_m):
    """False-safe rate conditions on observed failures close to the buffer threshold."""
    near_failure = (target >= -boundary_band_m) & (target <= 0)
    near_count = int(near_failure.sum())
    false_safe = int((near_failure & (prediction > 0)).sum())
    return {
        "examples": len(target),
        "mae_m": (prediction - target).abs().mean().item(),
        "mse_m2": mse_loss(prediction, target).item(),
        "observed_failures_in_band": near_count,
        "false_safe_in_band": false_safe,
        "false_safe_rate_in_band": false_safe / near_count if near_count else None,
    }


def evaluate(target, predictions, conditions, boundary_band_m):
    overall = {name: score(target, pred, boundary_band_m) for name, pred in predictions.items()}
    by_condition = {}
    for condition in sorted(set(conditions)):
        mask = torch.tensor([c == condition for c in conditions], dtype=torch.bool)
        by_condition[condition] = {
            name: score(target[mask], pred[mask], boundary_band_m)
            for name, pred in predictions.items()
        }
    return {"overall": overall, "by_condition": by_condition}


def save_plots(losses, scores, folder):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for key, label in (("train_mse", "Training"), ("val_mse", "Validation")):
        ax.plot([row["epoch"] for row in losses], [row[key] for row in losses], label=label)
    ax.set(xlabel="Epoch", ylabel="Mean squared margin error (m²)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "loss.png")
    plt.close(fig)

    conditions = list(scores["by_condition"])
    fig, ax = plt.subplots(figsize=(9, 5))
    methods = list(scores["overall"])
    for i, method in enumerate(methods):
        positions = [j + (i - (len(methods) - 1) / 2) * .25 for j in range(len(conditions))]
        errors = [scores["by_condition"][c][method]["mae_m"] for c in conditions]
        ax.bar(positions, errors, width=.25, label=method)
    ax.set_xticks(range(len(conditions)), conditions, rotation=35, ha="right")
    ax.set_ylabel("Held-out mean absolute margin error (m)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "condition_mae.png")
    plt.close(fig)


def run(data_path, cfg):
    if min(cfg.risk_epochs, cfg.patience, cfg.batch_size) < 1 or cfg.boundary_band_m <= 0:
        raise ValueError("Epochs, patience, batch size and boundary band must be positive.")
    device = setup_device(cfg.seed)
    data = load_data(data_path)
    splits = split_episodes(data, cfg)
    samples = {}
    counts = {}
    for name, ids in splits.items():
        history, target, conditions, window_counts = risk_examples(data, ids, cfg)
        if not len(target):
            raise ValueError(f"No usable {name} windows. Check the split, history and horizon.")
        samples[name] = (history, target, conditions)
        counts[name] = {"episodes": len(ids), **window_counts}
        print(f"{name}: {len(ids)} episodes, {window_counts}", flush=True)

    train_x, train_y, _ = samples["train"]
    val_x, val_y, _ = samples["val"]
    test_x, test_y, test_conditions = samples["test"]
    # Fit on training windows only. Keep targets in meters for both loss and evaluation.
    mean = train_x.mean(0)
    scale = train_x.std(0, unbiased=False)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    model = RiskNet(train_x.shape[1], cfg.hidden_sizes).to(device)
    best_epoch, losses = fit_risk_model(
        model, ((train_x - mean) / scale).to(device), train_y.to(device),
        ((val_x - mean) / scale).to(device), val_y.to(device), cfg,
    )
    test_prediction = predict(model, ((test_x - mean) / scale).to(device), cfg.batch_size)
    predictions = {
        "MLP": test_prediction.cpu(),
        "Training mean": torch.full_like(test_y, train_y.mean().item()),
        "Gap trend": gap_forecast(test_x, cfg),
    }
    scores = evaluate(test_y, predictions, test_conditions, cfg.boundary_band_m)

    folder = LOCAL_DIR / f"risk_seed{cfg.seed}"
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.cpu().state_dict(), "input_mean": mean, "input_scale": scale,
        "history_steps": cfg.history_steps, "horizon_steps": cfg.horizon_steps,
        "safety_buffer_m": cfg.safety_buffer_m, "hidden_sizes": list(cfg.hidden_sizes),
        "best_epoch": best_epoch,
    }, folder / "best.pt")
    report = {
        "settings": asdict(cfg),
        "device": str(device),
        "data_path": str(Path(data_path).resolve()),
        "example_counts": counts,
        "count_convention": "Complete = full horizon without a buffer crossing. Failed = first crossing observed in horizon. Censored = short horizon without observed crossing. Insufficient histories and post-failure origins are excluded.",
        "boundary_definition": "False-safe = predicted margin > 0 among test windows with observed minimum margin in [-boundary_band_m, 0].",
        "best_epoch": best_epoch, "loss_history": losses, "test": scores,
    }
    (folder / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
    save_plots(losses, scores, folder)
    for method, result in scores["overall"].items():
        print(f"{method}: test MAE {result['mae_m']:.4f} m, boundary false-safe "
              f"{result['false_safe_in_band']}/{result['observed_failures_in_band']}")
    print(f"Best epoch {best_epoch}. Results and per-condition scores: {folder}")


def main():
    cfg = Settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=LOCAL_DIR / "numerical.pt")
    parser.add_argument("--epochs", type=int, default=cfg.risk_epochs)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    args = parser.parse_args()
    run(args.data, replace(cfg, risk_epochs=args.epochs, seed=args.seed))


if __name__ == "__main__":
    main()
