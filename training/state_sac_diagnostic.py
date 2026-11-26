"""Compare numerical-state HJ-SAC with the RGB pilot using identical SAC updates."""
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path

import torch
from torch import nn

from .config import Settings, load_data, setup_device
from .dataset import HJ_FEATURES, hj_examples
from .models import Critic, SafetyFilter, mlp
from .visual_algorithms import HJSAC, LearnerOptions
from .visual_models import ResNetActor


class StateGaussianActor(nn.Module):
    # Share the RGB actor's squashing, sampling, and normalized log-probability.
    forward = ResNetActor.forward
    sample = ResNetActor.sample

    def __init__(self, cfg):
        super().__init__()
        self.network = mlp(len(HJ_FEATURES), cfg.hidden_sizes, 4)
        nn.init.uniform_(self.network[-1].weight, -.003, .003)
        nn.init.zeros_(self.network[-1].bias)
        low, high = torch.tensor(cfg.action_low), torch.tensor(cfg.action_high)
        self.register_buffer("midpoint", (low + high) / 2)
        self.register_buffer("half_range", (high - low) / 2)

    def logits(self, state):
        return self.network(state)


class StateHJSAC(HJSAC):
    def __init__(self, cfg, device, options):
        self.actor = StateGaussianActor(cfg).to(device)
        self.critic = Critic(len(HJ_FEATURES), 2, cfg.hidden_sizes).to(device)
        self.critic2 = Critic(len(HJ_FEATURES), 2, cfg.hidden_sizes).to(device)
        self.target_critic = deepcopy(self.critic).eval().requires_grad_(False)
        self.target_critic2 = deepcopy(self.critic2).eval().requires_grad_(False)
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.AdamW(
            list(self.critic.parameters()) + list(self.critic2.parameters()), lr=cfg.critic_lr)
        self.gamma, self.tau, self.alpha = cfg.gamma, cfg.tau, options.sac_alpha


class StateSACPolicy(SafetyFilter):
    """Deterministic mean-action evaluation of the numerical-state SAC diagnostic."""

    def __init__(self, checkpoint, device="cpu"):
        if (checkpoint.get("architecture") != "state_sac_diagnostic" or
                tuple(checkpoint.get("features", ())) != HJ_FEATURES):
            raise ValueError("Expected the numerical-state SAC feature contract")
        self.settings = checkpoint["settings"]
        cfg = Settings(**self.settings)
        self.device = torch.device(device)
        self.state_mean = checkpoint["state_mean"].to(self.device)
        self.state_std = checkpoint["state_std"].to(self.device)
        if (self.state_mean.shape != (len(HJ_FEATURES),) or
                self.state_std.shape != self.state_mean.shape or
                not self.state_mean.isfinite().all() or
                not self.state_std.isfinite().all() or not (self.state_std > 0).all()):
            raise ValueError("Invalid numerical-state normalization")
        self.threshold = cfg.switching_threshold
        self.saturation = cfg.longitudinal_saturation
        self.actor = StateGaussianActor(cfg).to(self.device)
        self.critic = Critic(len(HJ_FEATURES), 2, cfg.hidden_sizes).to(self.device)
        self.critic2 = Critic(len(HJ_FEATURES), 2, cfg.hidden_sizes).to(self.device)
        for name in ("actor", "critic", "critic2"):
            model = getattr(self, name)
            model.load_state_dict(checkpoint[name])
            model.eval().requires_grad_(False)
            if any(not p.isfinite().all() for p in model.parameters()):
                raise ValueError("Nonfinite numerical-state SAC weights")

    @torch.inference_mode()
    def score_action(self, state, action):
        state = self.normalize(state)
        command = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if (state.ndim != 1 or command.shape != (2,) or
                not command.isfinite().all() or command[1].abs() > 1):
            raise ValueError("Expected one state and a finite bounded steering command")
        command = torch.stack((command[0].clamp(-self.saturation, self.saturation), command[1]))
        return float(torch.minimum(self.critic(state, command), self.critic2(state, command)))


@torch.inference_mode()
def diagnose(trainer, batch, raw_state):
    state, action, _, margin, *_ = batch
    actor_action = trainer.actor(state)
    grid = torch.tensor([-3., 0., 3.], device=state.device)
    candidates = torch.stack((grid.expand(len(state), -1),
                              action[:, 1, None].expand(-1, 3)), dim=-1)
    expanded_state = state[:, None, :].expand(-1, 3, -1)
    q_grid = torch.minimum(trainer.critic(expanded_state, candidates),
                           trainer.critic2(expanded_state, candidates))
    logged_q = torch.minimum(trainer.critic(state, action), trainer.critic2(state, action))
    correlation = torch.corrcoef(torch.stack((logged_q, margin)))[0, 1]
    report = {
        "current_margin_correlation": float(correlation) if torch.isfinite(correlation) else None,
        "critic_score_std_m": float(logged_q.std()),
        "longitudinal_quantiles": torch.quantile(
            actor_action[:, 0], torch.tensor([0., .1, .5, .9, 1.], device=state.device)).tolist(),
        "steering_quantiles": torch.quantile(
            actor_action[:, 1], torch.tensor([0., .1, .5, .9, 1.], device=state.device)).tolist(),
        "groups": {},
    }
    gap, relative_speed = raw_state[:, 0], raw_state[:, 6]
    for name, mask in (
        ("close_closing", (gap <= 5) & (relative_speed <= -1)),
        ("wide_widening", (gap >= 6) & (relative_speed >= 1)),
    ):
        if mask.any():
            report["groups"][name] = {
                "rows": int(mask.sum()),
                "median_actor_longitudinal": float(actor_action[mask, 0].median()),
                "preferred_brake_coast_throttle_counts": torch.bincount(
                    q_grid[mask].argmax(1), minlength=3).tolist(),
                "median_accel_minus_brake_m": float((q_grid[mask, 2] - q_grid[mask, 0]).median()),
            }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, help="Override the reference HJ update count")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    saved = torch.load(args.reference, map_location="cpu", weights_only=True)
    cfg = Settings(**saved["settings"])
    if args.updates is not None:
        if args.updates < 1:
            parser.error("--updates must be positive")
        cfg = replace(cfg, updates=args.updates)
    options = LearnerOptions(**saved["learner_options"])
    data = load_data(args.data)
    if (saved["dataset_id"], saved["split_id"]) != (data["dataset_id"], data["split_id"]):
        raise ValueError("Dataset or split differs from reference checkpoint")
    torch.set_num_threads(4)
    device = setup_device(cfg.seed)
    train = hj_examples(data, data["splits"]["train"], cfg)
    validation = hj_examples(data, data["splits"]["val"], cfg)
    mean, std = train[0].mean(0), train[0].std(0).clamp_min(1e-3)
    raw_validation = validation[0].to(device)
    train = tuple(t.to(device) for t in train)
    validation = tuple(t.to(device) for t in validation)
    mean, std = mean.to(device), std.to(device)
    train = ((train[0] - mean) / std, train[1], (train[2] - mean) / std, *train[3:])
    validation = ((validation[0] - mean) / std, validation[1],
                  (validation[2] - mean) / std, *validation[3:])
    report = {
        "settings": asdict(cfg), "learner_options": asdict(options),
        "dataset_id": data["dataset_id"], "split_id": data["split_id"],
        "features": HJ_FEATURES, "train_rows": len(train[0]),
        "validation_rows": len(validation[0]), "runs": {},
        "interpretation": "Offline state-input diagnostic. Grouping uses relative forward velocity from recorded poses.",
    }
    for warmup in (0, options.critic_margin_warmup):
        torch.manual_seed(cfg.seed)
        trainer = StateHJSAC(cfg, device, options)
        sampler = torch.Generator().manual_seed(cfg.seed)
        def sample():
            ids = torch.randint(len(train[0]), (cfg.batch_size,), generator=sampler).to(device)
            return tuple(t[ids] for t in train)
        for _ in range(warmup):
            trainer.warmup_critic(sample())
        trainer.target_critic.load_state_dict(trainer.critic.state_dict())
        trainer.target_critic2.load_state_dict(trainer.critic2.state_dict())
        trainer.critic_optimizer.state.clear()
        history = []
        for step in range(cfg.updates + 1):
            if step:
                critic_loss, actor_loss = trainer.update(sample())
            if step % 50 == 0:
                metrics = diagnose(trainer, validation, raw_validation)
                metrics["step"] = step
                if step:
                    metrics.update(critic_batch_mse=float(critic_loss), actor_batch_loss=float(actor_loss))
                history.append(metrics)
                if step % 250 == 0:
                    print(json.dumps({"warmup": warmup, **metrics}), flush=True)
        report["runs"][str(warmup)] = history
        payload = {
            "architecture": "state_sac_diagnostic", "settings": asdict(cfg),
            "learner_options": asdict(options), "features": HJ_FEATURES,
            "state_mean": mean.cpu(), "state_std": std.cpu(),
            "dataset_id": data["dataset_id"], "split_id": data["split_id"],
            "warmup_steps": warmup,
        }
        payload.update({key: getattr(trainer, key).state_dict() for key in trainer.state_keys})
        torch.save(payload, args.output / f"state_sac_warmup_{warmup}.pt")
        (args.output / "diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
