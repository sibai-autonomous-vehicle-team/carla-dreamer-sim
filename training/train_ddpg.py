"""Train a two-sided, first-hit HJ critic and deterministic continuous-control actor."""

import argparse
import json
import math
import os
import tempfile
import time
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch.nn import functional as F

from .config import LOCAL_DIR, Settings, load_data, setup_device
from .dataset import HJ_FEATURES, hj_examples
from .models import Actor, Critic


OBJECTIVE = "discounted_two_sided_first_hit"
ACTION_CONVENTION = "effective_requested_longitudinal_and_requested_steer"
MILESTONES = {0, 200, 1000, 5000, 10000}


@torch.no_grad()
def hj_target(margin, next_margin, failed, next_q, gamma):
    """After the first hit, the observed successor's own margin is absorbing."""
    continuation = torch.where(failed, next_margin, next_q)
    return (1 - gamma) * margin + gamma * torch.minimum(margin, continuation)


@torch.no_grad()
def soft_update(target, online, tau):
    for frozen, learned in zip(target.parameters(), online.parameters()):
        frozen.lerp_(learned, tau)


class HJDDPG:
    """Networks and optimizers for one actor-critic update."""

    def __init__(self, state_dim, cfg, device):
        self.actor = Actor(state_dim, cfg.hidden_sizes, cfg.action_low, cfg.action_high).to(device)
        self.critic = Critic(state_dim, len(cfg.action_low), cfg.hidden_sizes).to(device)
        self.target_actor = deepcopy(self.actor).requires_grad_(False)
        self.target_critic = deepcopy(self.critic).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.gamma = cfg.gamma
        self.tau = cfg.tau

    def update(self, batch):
        state, action, next_state, margin, next_margin, failed = batch
        with torch.no_grad():
            next_q = self.target_critic(next_state, self.target_actor(next_state))
            target = hj_target(margin, next_margin, failed, next_q, self.gamma)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss = F.mse_loss(self.critic(state, action), target)
        critic_loss.backward()
        self.critic_optimizer.step()

        # Holding the critic weights fixed still lets dQ/da reach the actor.
        self.critic.requires_grad_(False)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss = -self.critic(state, self.actor(state)).mean()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic.requires_grad_(True)

        soft_update(self.target_actor, self.actor, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)
        return critic_loss.detach(), actor_loss.detach()


def training_normalization(train_states):
    mean = train_states.mean(0)
    std = train_states.std(0, unbiased=False)
    return mean, torch.where(std < 1e-6, torch.ones_like(std), std)


def describe(values):
    values = values.detach().float().cpu()
    quantiles = torch.quantile(values, torch.tensor([0., 0.1, 0.5, 0.9, 1.]))
    return dict(zip(("min", "p10", "median", "p90", "max"), quantiles.tolist()))


@torch.no_grad()
def diagnostics(trainer, states, logged_actions, support_low, support_high):
    actions = trainer.actor(states)
    outside = (actions < support_low) | (actions > support_high)
    return {
        "longitudinal": describe(actions[:, 0]),
        "steering": describe(actions[:, 1]),
        "braking_fraction": (actions[:, 0] < 0).float().mean().item(),
        "outside_train_action_box_fraction": outside.any(dim=1).float().mean().item(),
        "q_actor_m": describe(trainer.critic(states, actions)),
        "q_logged_m": describe(trainer.critic(states, logged_actions)),
    }


def gradient_norm(network):
    return torch.linalg.vector_norm(torch.stack([
        parameter.grad.detach().norm() for parameter in network.parameters()
    ])).item()


@torch.no_grad()
def inference_probes(trainer, raw_states, commands, mean, std, threshold):
    # Evaluate the in-memory weights on CPU before serializing them. A separate
    # process can repeat these individual decisions using only the saved model.
    actor = deepcopy(trainer.actor).cpu().eval()
    critic = deepcopy(trainer.critic).cpu().eval()
    probes = []
    for raw, nominal in zip(raw_states, commands):
        state = (raw - mean) / std
        score = float(critic(state, nominal))
        safe = actor(state).tolist()
        intervene = score <= threshold
        probes.append({"state": raw.tolist(), "nominal": nominal.tolist(), "actor": safe,
                       "score": score, "intervened": intervene,
                       "executed": safe if intervene else nominal.tolist()})
    return probes


def make_checkpoint(trainer, cfg, step, mean, std, data, selection, sampler, history,
                    sampled, elapsed_seconds):
    return {
        "format_version": 2, "objective": OBJECTIVE, "action_convention": ACTION_CONVENTION,
        "actuator_mapping": {"command_order": ["longitudinal", "requested_steering"],
                             "throttle": "clip(longitudinal / 3, 0, 1)",
                             "brake": "clip(-longitudinal / 3, 0, 1)",
                             "applied_steering": "-requested_steering"},
        "actor": trainer.actor.state_dict(), "critic": trainer.critic.state_dict(),
        "target_actor": trainer.target_actor.state_dict(),
        "target_critic": trainer.target_critic.state_dict(),
        "actor_optimizer": trainer.actor_optimizer.state_dict(),
        "critic_optimizer": trainer.critic_optimizer.state_dict(),
        "features": HJ_FEATURES, "state_mean": mean, "state_std": std, "settings": asdict(cfg),
        "step": step, "dataset_selection": selection, "dataset_id": data["dataset_id"],
        "split_id": data["split_id"], "sampler_state": sampler.get_state(),
        "history": history, "sampled": sampled, "elapsed_seconds": elapsed_seconds,
    }


def restore(checkpoint, trainer, cfg, data, selection, mean, std, sampler):
    if (checkpoint.get("format_version") != 2 or checkpoint.get("objective") != OBJECTIVE
            or checkpoint.get("action_convention") != ACTION_CONVENTION):
        raise ValueError("Resume requires a checkpoint of the two-sided first-hit objective.")
    if (checkpoint["dataset_id"], checkpoint["split_id"], checkpoint["dataset_selection"]) != (
            data["dataset_id"], data["split_id"], selection):
        raise ValueError("Dataset, split or source selection differs from the checkpoint.")
    current = asdict(cfg)
    for key, value in checkpoint["settings"].items():
        if key != "updates" and current[key] != value:
            raise ValueError(f"Resume settings differ for {key}.")
    if (tuple(checkpoint["features"]) != HJ_FEATURES or not torch.equal(mean, checkpoint["state_mean"])
            or not torch.equal(std, checkpoint["state_std"])):
        raise ValueError("Checkpoint feature order or training normalization differs.")
    for name in ("actor", "critic", "target_actor", "target_critic", "actor_optimizer", "critic_optimizer"):
        getattr(trainer, name).load_state_dict(checkpoint[name])
    sampler.set_state(checkpoint["sampler_state"])


def save_checkpoint(path, checkpoint):
    # Publish only a complete snapshot; an existing checkpoint is never replaced.
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(checkpoint, tmp_path)
        if os.name == "nt":
            tmp_path.rename(path)  # Windows rename fails when the destination exists.
        else:
            os.link(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def train(data_path, cfg, output, selection="combined", resume=None, require_cuda=False,
          checkpoint_interval=None):
    if checkpoint_interval is not None and (isinstance(checkpoint_interval, bool) or
            not isinstance(checkpoint_interval, int) or checkpoint_interval < 1):
        raise ValueError("checkpoint_interval must be a positive integer")
    if cfg.updates < 1 or cfg.batch_size < 1 or not 0 <= cfg.gamma <= 1 or not 0 < cfg.tau <= 1:
        raise ValueError("Check updates, batch size, gamma and tau.")
    if selection not in ("combined", "new"):
        raise ValueError("Choose combined or new training episodes.")
    cfg = replace(cfg, dataset_selection=selection)
    # Deterministic GPU matrix products require a reproducible cuBLAS workspace.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = setup_device(cfg.seed)
    if require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA was requested but is unavailable in this Python environment.")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    data = load_data(data_path)
    if "splits" not in data:
        raise ValueError("Prepare the two-source cache with the fixed grouped split first.")
    train_ids = [i for i in data["splits"]["train"] if selection == "combined" or data["sources"][i] == "new"]
    state, action, next_state, margin, next_margin, failed = hj_examples(data, train_ids, cfg)
    val_state, val_action = hj_examples(data, data["splits"]["val"], cfg)[:2]
    if not len(state) or not len(val_state):
        raise ValueError("Training and validation both require eligible transitions.")
    mean, std = training_normalization(state)
    replay = ((state - mean) / std, action, (next_state - mean) / std, margin, next_margin, failed)

    # Both source experiments use the same held-out probes, independent of training seed.
    selected = torch.randperm(len(val_state), generator=torch.Generator().manual_seed(42))[:2048]
    validation = ((val_state[selected] - mean) / std).to(device)
    logged = val_action[selected].to(device)
    support_low, support_high = action.amin(0).to(device), action.amax(0).to(device)
    sampler = torch.Generator().manual_seed(cfg.seed)
    trainer = HJDDPG(len(HJ_FEATURES), cfg, device)

    output = Path(output)
    history = []
    sampled = {"braking": 0, "first_failures": 0, "transitions": 0}
    start_step, elapsed_before = 0, 0.
    if resume is None:
        output.mkdir(parents=True, exist_ok=False)
    else:
        resume = Path(resume)
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        restore(checkpoint, trainer, cfg, data, selection, mean, std, sampler)
        start_step = checkpoint["step"]
        if cfg.updates <= start_step:
            raise ValueError("--updates must exceed the resumed update count.")
        if resume.parent.resolve() == output.resolve():
            if any(int(p.stem.split("_")[-1]) > start_step for p in output.glob("checkpoint_[0-9]*.pt")):
                raise FileExistsError("Later checkpoints already exist here. Use a fresh --run-dir to branch.")
        else:
            output.mkdir(parents=True, exist_ok=False)
        history, sampled = checkpoint["history"], checkpoint["sampled"]
        elapsed_before = checkpoint["elapsed_seconds"]

    started = time.perf_counter()
    def save(step):
        elapsed = elapsed_before + time.perf_counter() - started
        probes = []
        if step == cfg.updates:
            ids = selected[:3]
            probes = inference_probes(trainer, val_state[ids], val_action[ids], mean, std,
                                      cfg.switching_threshold)
        checkpoint = make_checkpoint(trainer, cfg, step, mean, std, data, selection, sampler,
                                     history, sampled, elapsed)
        save_checkpoint(output / f"checkpoint_{step:05d}.pt", checkpoint)
        metrics = {
            "objective": OBJECTIVE, "dataset_selection": selection, "dataset_id": data["dataset_id"],
            "split_id": data["split_id"], "settings": asdict(cfg), "step": step,
            "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "torch_version": str(torch.__version__), "train_episodes": len(train_ids),
            "train_transitions": len(state), "train_braking": int((action[:, 0] < 0).sum()),
            "train_first_failures": int(failed.sum()), "sampling": "uniform training transitions with replacement",
            "sampled": sampled, "validation_transitions": len(val_state), "validation_probe_count": len(selected),
            "history": history, "elapsed_seconds": elapsed, "updates_per_second": step / elapsed,
            "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.,
            "cpu_inference_probes": probes, "cpu_threads": torch.get_num_threads(),
        }
        temporary = output / "metrics.json.tmp"
        temporary.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf8")
        temporary.replace(output / "metrics.json")

    if resume is None:
        save(0)
    losses = []
    for step in range(start_step + 1, cfg.updates + 1):
        indices = torch.randint(len(state), (cfg.batch_size,), generator=sampler)
        batch = tuple(column.index_select(0, indices) for column in replay)
        sampled["braking"] += int((batch[1][:, 0] < 0).sum())
        sampled["first_failures"] += int(batch[-1].sum())
        sampled["transitions"] += len(indices)
        losses.append(torch.stack(trainer.update(tuple(column.to(device, non_blocking=True) for column in batch))))
        if step == 1 or step % 50 == 0 or step == cfg.updates:
            critic_loss, actor_loss = torch.stack(losses).mean(0).tolist()
            losses.clear()
            record = diagnostics(trainer, validation, logged, support_low, support_high)
            record.update(step=step, critic_mse=critic_loss, actor_loss=actor_loss,
                          actor_grad_l2=gradient_norm(trainer.actor), critic_grad_l2=gradient_norm(trainer.critic))
            # Any nonfinite batch loss survives averaging and invalidates the run.
            if not all(math.isfinite(record[k]) for k in ("critic_mse", "actor_loss", "actor_grad_l2", "critic_grad_l2")):
                raise FloatingPointError(f"Nonfinite training diagnostic at update {step}")
            json.dumps(record, allow_nan=False)
            history.append(record)
            if step == 1 or step % 500 == 0 or step == cfg.updates:
                print(f"{step:5d}  critic MSE {critic_loss:.5f}  actor loss {actor_loss:.5f}  "
                      f"validation braking {record['braking_fraction']:.1%}", flush=True)
        if (step in MILESTONES or step == cfg.updates or
                (checkpoint_interval is not None and step % checkpoint_interval == 0)):
            save(step)

    return output / f"checkpoint_{cfg.updates:05d}.pt"


def main():
    cfg = Settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=LOCAL_DIR / "hj_nine.pt")
    parser.add_argument("--updates", type=int, default=cfg.updates)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--dataset", choices=("combined", "new"), default=cfg.dataset_selection)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="start from a saved two-sided checkpoint")
    parser.add_argument("--require-cuda", action="store_true", help="fail if the selected device is CPU")
    parser.add_argument("--checkpoint-interval", type=int,
                        help="Save every N total updates, plus existing milestones and the final update; N must be positive")
    parser.add_argument('--architecture', choices=('resnet18', 'state-mlp'), default='resnet18',
                        help='New runs default to the RGB ResNet-18 architecture')
    parser.add_argument('--image-root', type=Path, help='Combined RGB dataset root; default uses cached source paths')
    parser.add_argument('--batch-size', type=int, help='Default 8 for temporal ResNet-18, 32 for single-frame ResNet-18, 256 for state MLP')
    parser.add_argument('--history-frames', type=int, default=5, help='Ordered consecutive RGB frames; 1 restores single-frame architecture')
    parser.add_argument('--no-pretrained', action='store_true', help='Initialize RGB ResNet-18 randomly instead of ImageNet DEFAULT')
    parser.add_argument('--validation-probes', type=int, default=256)
    parser.add_argument('--algorithm', choices=('bc','hj-td3-bc','ddpg','hj-sac'), default='hj-td3-bc')
    parser.add_argument('--sac-alpha', type=float, default=.1, help='HJ-SAC actor entropy weight')
    parser.add_argument('--critic-margin-warmup', type=int, default=0,
                        help='HJ-SAC initialization updates fitting measured current margins before HJ training')
    parser.add_argument('--bc-target', choices=('teacher','recorded'), default='teacher')
    parser.add_argument('--bc-alpha', type=float, default=.1, help='Q influence relative to normalized imitation loss')
    parser.add_argument('--actor-warmup', type=int, default=1000, help='Initial updates use only imitation in the actor loss')
    parser.add_argument('--policy-delay', type=int, default=2)
    parser.add_argument('--policy-noise', type=float, default=.2, help='Target noise std in normalized action units')
    parser.add_argument('--noise-clip', type=float, default=.5)
    parser.add_argument('--teacher-gap', type=float, default=5.)
    parser.add_argument('--teacher-kp', type=float, default=.5)
    parser.add_argument('--teacher-kd', type=float, default=.5)
    parser.add_argument('--teacher-steer-gain', type=float, default=.75)
    parser.add_argument('--recorded-tolerance', type=float, default=.15)
    parser.add_argument('--actor-init', type=Path, help='Initialize from a supervised BC actor checkpoint')
    parser.add_argument('--critic-lr', type=float, help='Default 1e-4 for new RGB algorithms, 1e-3 for legacy DDPG')
    args = parser.parse_args()
    if args.algorithm == 'hj-sac' and args.actor_init:
        parser.error('HJ-SAC uses a stochastic actor and cannot initialize from a BC checkpoint')
    if args.architecture == 'state-mlp' and (args.algorithm != 'ddpg' or args.actor_init):
        parser.error('state-mlp supports --algorithm ddpg only')
    if args.checkpoint_interval is not None and args.checkpoint_interval < 1:
        parser.error("--checkpoint-interval must be positive")
    if args.run_dir is None:
        args.run_dir = args.resume.parent if args.resume else LOCAL_DIR / f"{args.dataset}_data_{args.architecture}_{args.algorithm}_s{args.seed}"
    batch_size = args.batch_size if args.batch_size is not None else ((8 if args.history_frames > 1 else 32) if args.architecture == 'resnet18' else cfg.batch_size)
    critic_lr = args.critic_lr if args.critic_lr is not None else (1e-4 if args.algorithm != 'ddpg' else cfg.critic_lr)
    if not math.isfinite(critic_lr) or critic_lr <= 0:
        parser.error('--critic-lr must be positive')
    if args.history_frames < 1:
        parser.error("--history-frames must be positive")
    cfg = replace(cfg, updates=args.updates, seed=args.seed, batch_size=batch_size, critic_lr=critic_lr,
                  visual_history_frames=args.history_frames if args.architecture == "resnet18" else 1)
    if args.architecture == 'resnet18':
        from .visual_ddpg import train_visual
        from .visual_algorithms import LearnerOptions
        options = LearnerOptions(algorithm=args.algorithm,bc_target=args.bc_target,bc_alpha=args.bc_alpha,
                                 actor_warmup=args.actor_warmup,policy_delay=args.policy_delay,
                                 policy_noise=args.policy_noise,noise_clip=args.noise_clip,
                                 teacher_gap=args.teacher_gap,teacher_kp=args.teacher_kp,teacher_kd=args.teacher_kd,
                                 teacher_steer_gain=args.teacher_steer_gain,recorded_tolerance=args.recorded_tolerance,
                                 sac_alpha=args.sac_alpha,critic_margin_warmup=args.critic_margin_warmup)
        checkpoint = train_visual(args.data, cfg, args.run_dir, args.dataset, args.resume,
                                  args.require_cuda, args.checkpoint_interval, args.image_root,
                                  not args.no_pretrained, args.validation_probes, options, args.actor_init)
    else:
        checkpoint = train(args.data, cfg, args.run_dir, args.dataset, args.resume,
                           args.require_cuda, args.checkpoint_interval)
    print(f"Saved {checkpoint}", flush=True)


if __name__ == "__main__":
    main()

