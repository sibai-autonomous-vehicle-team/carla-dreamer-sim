from dataclasses import dataclass
import math

import torch
from torch import nn

from .dataset import HJ_FEATURES


def mlp(input_dim, hidden_sizes, output_dim):
    widths = [input_dim, *hidden_sizes]
    layers = []
    for before, after in zip(widths, widths[1:]):
        layers.extend([nn.Linear(before, after), nn.ReLU()])
    layers.append(nn.Linear(widths[-1], output_dim))
    return nn.Sequential(*layers)


class RiskNet(nn.Module):
    def __init__(self, input_dim, hidden_sizes):
        super().__init__()
        self.network = mlp(input_dim, hidden_sizes, 1)

    def forward(self, history):
        return self.network(history).squeeze(-1)


class Actor(nn.Module):
    def __init__(self, state_dim, hidden_sizes, action_low, action_high):
        super().__init__()
        low = torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.as_tensor(action_high, dtype=torch.float32)
        if low.ndim != 1 or high.shape != low.shape or not torch.all(high > low):
            raise ValueError("Each control needs an ordered lower and upper bound")
        self.network = mlp(state_dim, hidden_sizes, len(low))
        self.register_buffer("midpoint", (high + low) / 2)
        self.register_buffer("half_range", (high - low) / 2)

    def forward(self, state):
        return self.midpoint + self.half_range * torch.tanh(self.network(state))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes):
        super().__init__()
        self.network = mlp(state_dim + action_dim, hidden_sizes, 1)

    def forward(self, state, action):
        return self.network(torch.cat((state, action), dim=-1)).squeeze(-1)


@dataclass(frozen=True)
class FilterDecision:
    action: tuple
    nominal_q: float
    intervened: bool


class SafetyFilter:
    """Frozen two-sided HJ policy. Public decision methods take one *raw* state."""

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("architecture") == "resnet18_rgb":
            from .visual_models import RGBSafetyFilter
            return RGBSafetyFilter(checkpoint, device)
        if checkpoint.get("architecture") == "state_sac_diagnostic":
            from .state_sac_diagnostic import StateSACPolicy
            return StateSACPolicy(checkpoint, device)
        return cls(checkpoint, device)

    def __init__(self, checkpoint, device="cpu"):
        if (checkpoint.get("format_version") != 2 or
            checkpoint.get("objective") != "discounted_two_sided_first_hit" or
            checkpoint.get("action_convention") != "effective_requested_longitudinal_and_requested_steer" or
            tuple(checkpoint.get("features", ())) != HJ_FEATURES):
            raise ValueError("Expected a two-sided HJ checkpoint with the supported feature and command conventions")
        self.settings = checkpoint["settings"]
        self.device = torch.device(device)
        self.state_mean = checkpoint["state_mean"].to(self.device)
        self.state_std = checkpoint["state_std"].to(self.device)
        state_dim = len(HJ_FEATURES)
        if (self.state_mean.shape != (state_dim,) or self.state_std.shape != (state_dim,)
                or not torch.isfinite(self.state_mean).all() or not torch.isfinite(self.state_std).all()
                or not (self.state_std > 0).all()):
            raise ValueError("Invalid state normalization in checkpoint")
        self.threshold = float(self.settings["switching_threshold"])
        self.saturation = float(self.settings["longitudinal_saturation"])
        if not math.isfinite(self.threshold) or self.saturation != 3.0:
            raise ValueError("Expected a finite switching threshold and CARLA longitudinal saturation of 3")
        actor = Actor(state_dim, self.settings["hidden_sizes"], self.settings["action_low"], self.settings["action_high"])
        critic = Critic(state_dim, 2, self.settings["hidden_sizes"])
        for model, key in ((actor, "actor"), (critic, "critic")):
            model.load_state_dict(checkpoint[key])
            model.to(self.device).eval().requires_grad_(False)
            if any(not parameter.isfinite().all() for parameter in model.parameters()):
                raise ValueError(f"Nonfinite {key} parameters in checkpoint")
        low, high = actor.midpoint - actor.half_range, actor.midpoint + actor.half_range
        for bounds, name in ((low, "action_low"), (high, "action_high")):
            expected = torch.as_tensor(self.settings[name], dtype=bounds.dtype, device=self.device)
            if not torch.allclose(bounds, expected):
                raise ValueError("Actor's stored scaling disagrees with configured action bounds")
        if low[0] < -self.saturation or high[0] > self.saturation or low[1] < -1 or high[1] > 1:
            raise ValueError("Actor bounds exceed the CARLA actuator domain")
        self.actor, self.critic = actor, critic

    def normalize(self, state):
        values = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if values.shape[-1:] != (len(HJ_FEATURES),) or not values.isfinite().all():
            raise ValueError(f"Expected finite raw features ordered as {HJ_FEATURES}")
        if (values[..., 0] < 0).any():
            raise ValueError("Current geometric clearance must be nonnegative")
        return (values - self.state_mean) / self.state_std

    @torch.inference_mode()
    def predict_safe_action(self, state):
        state = self.normalize(state)
        if state.ndim != 1:
            raise ValueError("Expected a single decision state")
        return tuple(self.actor(state).tolist())

    @torch.inference_mode()
    def score_action(self, state, action):
        state = self.normalize(state)
        command = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if state.ndim != 1 or command.shape != (2,) or not command.isfinite().all() or abs(command[1]) > 1:
            raise ValueError("Expected one state and a finite [longitudinal, steering request] with |steer| <= 1")
        # Different longitudinal requests past saturation give the same vehicle control.
        effective = torch.stack((command[0].clamp(-self.saturation, self.saturation), command[1]))
        score = float(self.critic(state, effective))
        if not math.isfinite(score):
            raise ValueError("Nonfinite nominal safety estimate")
        return score

    def filter_action(self, state, nominal_action):
        score = self.score_action(state, nominal_action)
        intervene = score <= self.threshold
        action = self.predict_safe_action(state) if intervene else tuple(float(x) for x in nominal_action)
        return FilterDecision(action, score, intervene)
