"""Offline action support and critic probes on a fixed validation or test split."""
import argparse
import json
import os
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from .config import LOCAL_DIR, Settings, load_data
from .dataset import HJ_FEATURES, episode_slices, hj_examples, observations
from .models import Actor, Critic


def distribution(x):
    if len(x) == 0:
        return None
    return dict(zip(("min", "p05", "median", "p95", "max"),
                    np.quantile(x.detach().cpu().numpy(), [0, .05, .5, .95, 1]).tolist()))


def closing_speed(state):
    # Change in center separation projected along the line between vehicle centers.
    displacement = state[:, 1:3]
    return -(displacement * state[:, 6:8]).sum(1) / displacement.norm(dim=1).clamp_min(1e-6)


def sample_indices(n, limit):
    return torch.randperm(n, generator=torch.Generator().manual_seed(31))[:limit]


def load_policy(path, historical):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if historical:
        if "objective" in saved or tuple(saved["features"]) != HJ_FEATURES:
            raise ValueError("Expected the historical eight-feature checkpoint")
        cfg = saved["settings"]
        actor = Actor(8, cfg["hidden_sizes"], cfg["action_low"], cfg["action_high"])
        critic = Critic(8, 2, cfg["hidden_sizes"])
        actor.load_state_dict(saved["actor"])
        critic.load_state_dict(saved["critic"])
        normalize = lambda x: (x - saved["state_mean"]) / saved["state_std"]
        return saved, normalize, actor.eval(), critic.eval()
    from .models import SafetyFilter
    policy = SafetyFilter(saved)
    return saved, policy.normalize, policy.actor, policy.critic


def action_stats(action):
    if not len(action):
        return {"rows": 0}
    return {"rows": len(action), "longitudinal": distribution(action[:, 0]),
            "steering": distribution(action[:, 1]), "brake_fraction": float((action[:, 0] < 0).float().mean()),
            "negative_steer_fraction": float((action[:, 1] < 0).float().mean()),
            "abs_steer_ge_095_fraction": float((action[:, 1].abs() >= .95).float().mean()),
            "abs_long_ge_285_fraction": float((action[:, 0].abs() >= 2.85).float().mean())}


def action_bins(values, limits, actions):
    output = []
    for lo, hi in zip(limits, limits[1:]):
        hit = (values >= lo) & (values < hi)
        output.append({"range": f"[{lo:g}, {hi:g})", **action_stats(actions[hit])})
    return output


def choose_probes(states):
    gap = states[:, 0]
    closing = closing_speed(states)
    # Select actual held-out observations using geometry alone.
    near = gap < 2
    return {"close and closing": int(torch.where(near, closing, -torch.inf).argmax()) if near.any() else int(gap.argmin()),
            "median gap": int((gap - gap.median()).abs().argmin()),
            "fast closing": int((closing - closing.quantile(.99)).abs().argmin()),
            "widest observed gap": int(gap.argmax())}


def q_sweeps(states, logged, normalize, critic, cfg):
    output = {}
    for label, row in choose_probes(states).items():
        detail = {"raw_state": states[row].tolist(), "recorded_request": logged[row].tolist()}
        for axis, name in enumerate(("longitudinal", "steering")):
            grid = torch.linspace(cfg.action_low[axis], cfg.action_high[axis], 41)
            actions = logged[row].repeat(len(grid), 1)
            actions[:, axis] = grid
            scores = critic(normalize(states[row].expand(len(grid), -1)), actions)
            detail[name] = {"commands": grid.tolist(), "q_m": scores.tolist(),
                            "maximizing_grid_command": float(grid[scores.argmax()])}
        output[label] = detail
    rows = sample_indices(len(states), 1024)
    n = len(rows)
    grid = torch.linspace(cfg.action_low[1], cfg.action_high[1], 21)
    actions = logged[rows, None, :].expand(n, len(grid), 2).clone()
    actions[:, :, 1] = grid
    scores = critic(normalize(states[rows])[:, None, :].expand(n, len(grid), 8), actions)
    maximum = grid[scores.argmax(1)]
    return output, {"sample_rows": n, "best_steer_minus_one_fraction": float((maximum == -1).float().mean()),
                    "best_steer_plus_one_fraction": float((maximum == 1).float().mean()),
                    "q_minus_one_minus_q_zero_m": distribution(scores[:, 0] - scores[:, len(grid)//2])}


def recovery_samples(data, ids, cfg):
    # Post-hit recovery poses are diagnostic only. One probe per clean recorded reopening.
    states, actions = [], []
    slices = episode_slices(data)
    for ep in ids:
        sl = slices[ep]
        gaps, costs = data["distances"][sl], data["costs"][sl]
        hits = (gaps <= cfg.safety_buffer_m).nonzero().flatten()
        if not len(hits) or costs.any() or len(gaps) < 5:
            continue
        clear = (gaps > cfg.safety_buffer_m + 1).unfold(0, 5, 1).all(1).nonzero().flatten()
        clear = clear[clear > hits[0]]
        if len(clear):
            i = int(clear[0]) - cfg.gap_offset
            j = i + cfg.action_offset
            if 1 <= i < len(gaps) and 0 <= j < len(gaps):
                states.append(observations(data["states"][sl], gaps, torch.tensor([i]), cfg)[0])
                actions.append(data["actions"][sl][j].clamp(torch.tensor(cfg.action_low), torch.tensor(cfg.action_high)))
    if not states:
        return torch.empty(0, 8), torch.empty(0, 2)
    return torch.stack(states), torch.stack(actions)


def local_action_support(query, logged, train, train_actions, normalize, actor, cfg):
    if not len(query):
        return {"eligible": 0, "sampled": 0}
    ids = sample_indices(len(query), 96)
    q = query[ids]
    reference = normalize(train)
    count = min(64, len(reference))
    ranges = torch.tensor(cfg.action_high) - torch.tensor(cfg.action_low)
    records = []
    for raw in q.split(8):
        neighbors = torch.cdist(normalize(raw), reference).topk(count, largest=False)
        neighbor_actions = train_actions[neighbors.indices]
        proposal = actor(normalize(raw))
        records.append((neighbors.values[:, -1], neighbor_actions, proposal))
    radius, neighbors, proposed = (torch.cat(values) for values in zip(*records))
    def summarize(actions):
        joint = ((neighbors - actions[:, None]) / ranges).norm(dim=2).amin(1)
        outside = ((actions < neighbors.amin(1)) | (actions > neighbors.amax(1))).any(1)
        return {"nearest_joint_action_distance": distribution(joint),
                "outside_local_action_box_fraction": float(outside.float().mean())}
    return {"eligible": len(query), "sampled": len(ids), "neighbors": count, "state_radius": distribution(radius),
            "actor": summarize(proposed), "recorded": summarize(logged[ids])}


def plot_results(output, result, actor_actions, logged, history):
    os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_DIR / "mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for col, label in enumerate(("Longitudinal request", "Steering request")):
        axis = axes[col, 0]
        bins = np.linspace(-3 if col == 0 else -1, 3 if col == 0 else 1, 41)
        for actions, name in ((logged, "Recorded"), (actor_actions, "Actor")):
            axis.hist(actions[:, col].numpy(), bins=bins, histtype="step", density=True, label=name, linewidth=1.6)
        axis.set(xlabel=label, ylabel="Density")
        axis.legend()
        for j, kind in enumerate(("gap_m", "closing_mps"), 1):
            axis = axes[col, j]
            key = ("longitudinal", "steering")[col]
            for name in ("recorded", "actor"):
                b = result["by_band"][kind][name]
                y = [item[key]["median"] if item["rows"] else float("nan") for item in b]
                axis.plot(range(len(b)), y, marker="o", label=name)
            axis.set_xticks(range(len(b)), [z["range"] for z in b], rotation=25, ha="right")
            axis.set(xlabel="Gap (m)" if j == 1 else "Closing speed (m/s)", ylabel="Median " + label.lower())
            axis.legend()
    fig.tight_layout()
    fig.savefig(output / "actions.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for label, detail in result["q_sweeps"].items():
        for axis, name in zip(axes, ("longitudinal", "steering")):
            sweep = detail[name]
            axis.plot(sweep["commands"], sweep["q_m"], label=label)
            axis.set(xlabel=f"Requested {name}", ylabel="Predicted discounted margin (m)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "critic_sweeps.png", dpi=140)
    plt.close(fig)
    if history:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for axis, key, label in zip(axes, ("critic_mse", "actor_loss"), ("Critic MSE", "Actor loss (−mean Q)")):
            axis.plot([r["step"] for r in history], [r[key] for r in history])
            axis.set(xlabel="Gradient updates", ylabel=label)
        fig.tight_layout()
        fig.savefig(output / "training_progress.png", dpi=140)
        plt.close(fig)


@torch.inference_mode()
def evaluate(args):
    saved_header = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if saved_header.get("architecture") == "resnet18_rgb":
        from .visual_ddpg import evaluate_visual
        return evaluate_visual(args)
    del saved_header
    torch.set_num_threads(2)
    data = load_data(args.data)
    saved, normalize, actor, critic = load_policy(args.checkpoint, args.historical)
    if not args.historical and (saved["dataset_id"], saved["split_id"]) != (data["dataset_id"], data["split_id"]):
        raise ValueError("Use the cache and split identified by this checkpoint")
    cfg = Settings(**{k:v for k,v in saved["settings"].items() if k in {f.name for f in fields(Settings)}})
    ids = data["splits"][args.split]
    train_ids = data["splits"]["train"]
    selection = saved.get("dataset_selection", "old")
    if selection in ("new", "old"):
        train_ids = [i for i in train_ids if data["sources"][i] == selection]
    x_parts, u_parts, hit_parts, sources = [], [], [], []
    for source in ("old", "new"):
        x, u, _, _, _, hit = hj_examples(data, [i for i in ids if data["sources"][i] == source], cfg)
        x_parts.append(x)
        u_parts.append(u)
        hit_parts.append(hit)
        sources.extend([source] * len(x))
    x, u, hit = torch.cat(x_parts), torch.cat(u_parts), torch.cat(hit_parts)
    train, train_actions, *_ = hj_examples(data, train_ids, cfg)
    actions = actor(normalize(x))
    q_logged = critic(normalize(x), u)
    q_actor = critic(normalize(x), actions)
    if not all(torch.isfinite(t).all() for t in (x, u, actions, q_logged, q_actor)):
        raise ValueError("Nonfinite evaluation output")
    low, high = train_actions.amin(0), train_actions.amax(0)
    sweeps, grid_summary = q_sweeps(x, u, normalize, critic, cfg)
    result = {"dataset_id":data["dataset_id"], "split_id":data["split_id"], "split":args.split, "step":saved["step"],
              "training_selection":selection, "historical":args.historical, "actor":action_stats(actions),
              "recorded":action_stats(u), "q_actor_m":distribution(q_actor), "q_recorded_m":distribution(q_logged),
              "outside_global_train_box_fraction":float(((actions < low) | (actions > high)).any(1).float().mean()),
              "q_sweeps":sweeps, "steer_grid_summary":grid_summary,
              "interpretation":"Open-loop scores on recorded states. Actual filter interventions and safety outcomes require simulation.",
              "by_source":{}, "by_band":{}}
    for source in ("old", "new"):
        mask = torch.tensor([s == source for s in sources])
        section = {"actor":action_stats(actions[mask]), "recorded":action_stats(u[mask])}
        if not args.historical:
            section["recorded_request_trigger_fraction"] = float((q_logged[mask] <= cfg.switching_threshold).float().mean())
        result["by_source"][source] = section
    closing = closing_speed(x)
    for label, value, bins in (("gap_m", x[:, 0], [-float("inf"), 2, 5, 10, 20, cfg.max_gap_m, float("inf")]),
                               ("closing_mps", closing, [-float("inf"), -2, 0, 2, 4, float("inf")])):
        # Keep the gap bins ordered when changing the far boundary.
        result["by_band"][label] = {name:action_bins(value, sorted(set(bins)), a) for name,a in (("actor", actions), ("recorded",u))}
    if not args.historical:
        result["inference_contract"] = {
            "checkpoint": args.checkpoint.name, "feature_order": list(HJ_FEATURES), "dt_s": cfg.dt,
            "d_min_m": cfg.safety_buffer_m, "d_max_m": cfg.max_gap_m, "first_failure": "h <= 0",
            "seed": cfg.seed, "switch": "Q(raw_state, effective_nominal_request) <= switching_threshold",
            "switching_threshold": cfg.switching_threshold, "action_low": cfg.action_low,
            "action_high": cfg.action_high, "action_order": ["longitudinal_request", "steering_request"],
            "throttle": "clip(longitudinal_request / 3, 0, 1)",
            "brake": "clip(-longitudinal_request / 3, 0, 1)", "applied_steer": "-steering_request",
            "normalization": "use state_mean and state_std from checkpoint on raw features",
            "archive_gap_offset": cfg.gap_offset, "archive_action_offset": cfg.action_offset,
            "live_gap": "current synchronized measurement; PID until previous pose is available"}
        triggered = q_logged <= cfg.switching_threshold
        result["switch_on_recorded_requests"] = {}
        for label, mask in (("all", torch.ones_like(hit)), ("near_2m", x[:, 0] <= 2),
                            ("observed_first_failure_next", hit)):
            result["switch_on_recorded_requests"][label] = {
                "evaluated": int(mask.sum()), "triggered": int((mask & triggered).sum())}
        masks = {"all":torch.ones(len(x), dtype=torch.bool), "small_gap":x[:,0] < cfg.safety_buffer_m + 1,
                 "fast_closing":closing > 2, "near_far_boundary":x[:,0] >= cfg.max_gap_m - 2}
        result["local_support"] = {label:local_action_support(x[mask], u[mask], train, train_actions, normalize, actor, cfg)
                                   for label,mask in masks.items()}
        rx, ru = recovery_samples(data, ids, cfg)
        result["local_support"]["recorded_post_hit_reopening"] = local_action_support(rx, ru, train, train_actions, normalize, actor, cfg)
        result["support_method"] = "Up to 96 held-out rows per stratum, 64 nearest training transitions in train-standardized eight-feature Euclidean distance. Joint action distance scales each component by its full range. Radius reports distance to the 64th neighbor. Neighboring rows can share an episode. Empirical coverage diagnostic only."
    history_file = args.checkpoint.parent / "metrics.json"
    history = json.loads(history_file.read_text(encoding="utf8"))["history"] if history_file.is_file() else []
    args.output.mkdir(parents=True, exist_ok=True)
    if not getattr(args, "no_plots", False):
        plot_results(args.output, result, actions, u, [row for row in history if row["step"] <= saved["step"]])
    (args.output / "evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf8")
    print(args.output, result["actor"], grid_summary, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=LOCAL_DIR / "hj_nine.pt")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--historical", action="store_true", help="explicitly inspect the original, one-sided 200-update checkpoint")
    parser.add_argument("--no-plots", action="store_true", help="Write JSON diagnostics without requiring Matplotlib")
    parser.add_argument("--image-root", type=Path, help="Override RGB checkpoint image root")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
