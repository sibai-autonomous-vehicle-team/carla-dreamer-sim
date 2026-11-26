"""RGB ResNet-18 models matching train_distance_cnn.py preprocessing."""
import math

import torch
from torch import nn

from .models import FilterDecision, mlp

ARCHITECTURE = 'resnet18_rgb'
PREPROCESS = {'shape': [224, 224, 3], 'dtype': 'uint8', 'channel_order': 'RGB',
              'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}


class RGBEncoder(nn.Module):
    def __init__(self, pretrained=False, freeze_batch_norm=False):
        super().__init__()
        self.freeze_batch_norm = freeze_batch_norm
        from torchvision.models import ResNet18_Weights, resnet18
        self.network = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.register_buffer('image_mean', torch.tensor(PREPROCESS['mean']).view(1,3,1,1))
        self.register_buffer('image_std', torch.tensor(PREPROCESS['std']).view(1,3,1,1))

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_batch_norm:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def preprocess(self, frames):
        if frames.dtype != torch.uint8 or frames.ndim != 4 or tuple(frames.shape[1:]) != (224,224,3):
            raise ValueError('Expected uint8 RGB images [B,224,224,3]')
        x = frames.permute(0,3,1,2).to(dtype=torch.float32)/255.
        return (x-self.image_mean)/self.image_std


def temporal_features(encoder, frames):
    """Encode oldest-to-newest frames with one shared backbone; retain order."""
    if frames.ndim != 5 or frames.shape[1] != encoder.history_frames:
        raise ValueError(f'Expected [B,{encoder.history_frames},224,224,3] history')
    batch, history = frames.shape[:2]
    features = encoder.network(encoder.preprocess(frames.reshape(-1,*frames.shape[2:])))
    return features.reshape(batch,history*features.shape[-1])


class ResNetActor(RGBEncoder):
    def __init__(self, action_low=(-3.,-1.), action_high=(3.,1.), pretrained=False, freeze_batch_norm=False, history_frames=1, stochastic=False):
        super().__init__(pretrained, freeze_batch_norm)
        self.history_frames = history_frames
        if history_frames < 1:
            raise ValueError('history_frames must be positive')
        width = self.network.fc.in_features
        self.stochastic = stochastic
        outputs = 4 if stochastic else 2
        self.network.fc = nn.Linear(width, outputs) if history_frames == 1 else nn.Identity()
        if history_frames > 1:
            self.temporal_head = nn.Sequential(nn.Linear(width*history_frames,256), nn.ReLU(), nn.Linear(256,outputs))
        if stochastic:
            head = self.network.fc if history_frames == 1 else self.temporal_head[-1]
            nn.init.uniform_(head.weight, -0.003, 0.003)
            nn.init.zeros_(head.bias)
        low, high = torch.tensor(action_low), torch.tensor(action_high)
        if low.shape != (2,) or high.shape != (2,) or not torch.isfinite(low).all() or not torch.isfinite(high).all() or not (high>low).all():
            raise ValueError('Expected two ordered finite action bounds')
        self.register_buffer('midpoint', (low+high)/2)
        self.register_buffer('half_range', (high-low)/2)

    def logits(self, frames):
        if self.history_frames == 1:
            return self.network(self.preprocess(frames))
        return self.temporal_head(temporal_features(self, frames))

    def forward(self, frames):
        return self.midpoint + self.half_range * self.logits(frames)[..., :2].tanh()

    def sample(self, frames):
        mean, log_std = self.logits(frames).chunk(2, dim=-1)
        distribution = torch.distributions.Normal(mean, log_std.clamp(-5., 2.).exp())
        raw = distribution.rsample()
        # Entropy is measured in normalized action coordinates.
        log_jacobian = 2 * (math.log(2) - raw - nn.functional.softplus(-2 * raw))
        log_prob = (distribution.log_prob(raw) - log_jacobian).sum(dim=-1)
        return self.midpoint + self.half_range * raw.tanh(), log_prob


class ResNetCritic(RGBEncoder):
    def __init__(self, hidden_sizes=(128,128), pretrained=False, freeze_batch_norm=False, history_frames=1):
        super().__init__(pretrained, freeze_batch_norm)
        width = self.network.fc.in_features
        self.network.fc = nn.Identity()
        self.history_frames = history_frames
        if history_frames < 1:
            raise ValueError("history_frames must be positive")
        self.head = mlp(width*history_frames+2, hidden_sizes, 1)

    def forward(self, frames, action):
        features = (self.network(self.preprocess(frames)) if self.history_frames == 1
                    else temporal_features(self, frames))
        return self.head(torch.cat((features,action),-1)).squeeze(-1)


class RGBSafetyFilter:
    observation_kind = 'rgb'

    def __init__(self, checkpoint, device='cpu'):
        self.algorithm = checkpoint.get('algorithm', 'ddpg')
        if self.algorithm not in ('bc','hj-td3-bc','ddpg','hj-sac'):
            raise ValueError('Unsupported checkpoint algorithm')
        expected_objective = 'controller_imitation' if self.algorithm == 'bc' else 'discounted_two_sided_first_hit'
        if (checkpoint.get('format_version') not in (3,4) or checkpoint.get('architecture') != ARCHITECTURE or
                checkpoint.get('objective') != expected_objective or
                checkpoint.get('action_convention') != 'effective_requested_longitudinal_and_requested_steer' or
                checkpoint.get('preprocessing') != PREPROCESS):
            raise ValueError('Expected an RGB ResNet-18 HJ checkpoint with the supported image contract')
        self.settings = checkpoint['settings']
        self.device = torch.device(device)
        self.threshold = float(self.settings['switching_threshold'])
        self.saturation = float(self.settings['longitudinal_saturation'])
        if not math.isfinite(self.threshold) or self.saturation != 3.:
            raise ValueError('Invalid switching settings')
        self.history_frames = self.settings.get('visual_history_frames',1)
        self.actor = ResNetActor(self.settings['action_low'],self.settings['action_high'],
                                 history_frames=self.history_frames, stochastic=self.algorithm == 'hj-sac')
        self.critic = ResNetCritic(self.settings['hidden_sizes'],history_frames=self.history_frames) if 'critic' in checkpoint else None
        self.critic2 = ResNetCritic(self.settings['hidden_sizes'],history_frames=self.history_frames) if 'critic2' in checkpoint else None
        self.has_safety_critic = self.critic is not None
        models = [(self.actor,'actor')]
        if self.critic is not None: models.append((self.critic,'critic'))
        if self.critic2 is not None: models.append((self.critic2,'critic2'))
        for network,key in models:
            network.load_state_dict(checkpoint[key])
            network.to(self.device).eval().requires_grad_(False)
            if any(not p.isfinite().all() for p in network.parameters()):
                raise ValueError('Nonfinite model weights')
        expected_low = self.actor.midpoint-self.actor.half_range
        expected_high = self.actor.midpoint+self.actor.half_range
        if (not torch.allclose(expected_low,torch.tensor(self.settings['action_low'],device=self.device)) or
                not torch.allclose(expected_high,torch.tensor(self.settings['action_high'],device=self.device)) or
                expected_low[0] < -3 or expected_high[0] > 3 or expected_low[1] < -1 or expected_high[1] > 1):
            raise ValueError('Stored action scaling is inconsistent')

    def conservative_q(self, images, actions):
        if self.critic is None:
            raise ValueError('The supervised BC actor has no safety critic; use actor_only mode')
        q = self.critic(images,actions)
        return torch.minimum(q,self.critic2(images,actions)) if self.critic2 is not None else q

    def image(self, frame):
        image = torch.as_tensor(frame,device=self.device)
        expected = (224,224,3) if self.history_frames == 1 else (self.history_frames,224,224,3)
        if image.dtype != torch.uint8 or tuple(image.shape) != expected:
            raise ValueError(f'Expected uint8 RGB observation {expected}')
        return image.unsqueeze(0)

    @torch.inference_mode()
    def predict_safe_action(self, frame):
        return tuple(self.actor(self.image(frame))[0].tolist())

    @torch.inference_mode()
    def score_action(self, frame, action):
        command = torch.as_tensor(action,dtype=torch.float32,device=self.device)
        if command.shape != (2,) or not command.isfinite().all() or abs(command[1]) > 1:
            raise ValueError('Expected finite longitudinal and steering requests')
        command = torch.stack((command[0].clamp(-3,3),command[1]))
        score = float(self.conservative_q(self.image(frame),command.unsqueeze(0))[0])
        if not math.isfinite(score):
            raise ValueError('Nonfinite safety score')
        return score

    def filter_action(self, frame, nominal_action):
        score = self.score_action(frame,nominal_action)
        intervene = score <= self.threshold
        action = self.predict_safe_action(frame) if intervene else tuple(float(x) for x in nominal_action)
        return FilterDecision(action,score,intervene)
