"""Supervised controller imitation and an HJ adaptation of TD3+BC.

TD3 mechanisms follow Fujimoto & Gu's TD3+BC; the return target remains this
project's HJ first-hit backup, and teacher relabeling is an explicit extension.
"""
from copy import deepcopy
from dataclasses import dataclass, asdict
import math

import torch
from torch.nn import functional as F

from .train_ddpg import hj_target
from .visual_models import ResNetActor, ResNetCritic


@dataclass(frozen=True)
class LearnerOptions:
    algorithm: str = 'hj-td3-bc'
    bc_target: str = 'teacher'
    bc_alpha: float = .1
    actor_warmup: int = 1000
    policy_delay: int = 2
    policy_noise: float = .2
    noise_clip: float = .5
    teacher_gap: float = 5.
    teacher_kp: float = .5
    teacher_kd: float = .5
    teacher_steer_gain: float = .75
    recorded_tolerance: float = .15
    sac_alpha: float = .1
    critic_margin_warmup: int = 0

    def validate(self):
        if self.algorithm not in ('bc','hj-td3-bc','ddpg','hj-sac') or self.bc_target not in ('teacher','recorded'):
            raise ValueError('Unsupported algorithm or imitation target')
        values = asdict(self)
        for key in ('bc_alpha','policy_noise','noise_clip','teacher_gap','teacher_kp',
                    'teacher_kd','teacher_steer_gain','recorded_tolerance','sac_alpha'):
            if not math.isfinite(values[key]) or values[key] < 0:
                raise ValueError(f'{key} must be finite and nonnegative')
        if self.teacher_gap <= 0 or self.policy_delay < 1 or self.actor_warmup < 0:
            raise ValueError('Invalid teacher gap, delay or warmup')
        if self.critic_margin_warmup < 0 or (self.critic_margin_warmup and self.algorithm != 'hj-sac'):
            raise ValueError('Critic margin warmup requires HJ-SAC and a nonnegative step count')
        return self


def soft_update(target,online,tau):
    with torch.no_grad():
        source = online.state_dict()
        for name,destination in target.state_dict().items():
            if destination.is_floating_point(): destination.lerp_(source[name],tau)
            else: destination.copy_(source[name])


def imitation_loss(action,target,low,high):
    """Dimensionless MSE: acceleration and steering have equal normalized scale."""
    scale = (high-low)/2
    return F.mse_loss(action/scale,target/scale)


def smoothed_action(action,low,high,noise_std,noise_clip,noise=None):
    """Noise settings are expressed in normalized [-1,1] action coordinates."""
    noise = torch.randn_like(action) if noise is None else noise
    perturbation = (noise*noise_std).clamp(-noise_clip,noise_clip)*(high-low)/2
    return torch.maximum(low,torch.minimum(high,action+perturbation))


class ControllerBC:
    def __init__(self,cfg,device,pretrained,options):
        self.actor = ResNetActor(cfg.action_low,cfg.action_high,pretrained,freeze_batch_norm=True,history_frames=cfg.visual_history_frames).to(device)
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(),lr=cfg.actor_lr)
        self.critic = None
        self.options = options
        self.step = 0
        self.last_metrics = {}
        self.actor_updated = False
        self.low = torch.tensor(cfg.action_low,device=device)
        self.high = torch.tensor(cfg.action_high,device=device)

    @property
    def state_keys(self): return ('actor','actor_optimizer')

    def update(self,batch,target,mask):
        self.step += 1
        self.actor_updated = bool(mask.any())
        zero = torch.zeros((),device=self.low.device)
        if not self.actor_updated:
            self.last_metrics = {'bc_rows':0}
            return zero,zero
        self.actor.train()
        action = self.actor(batch[0][mask])
        loss = imitation_loss(action,target[mask],self.low,self.high)
        self.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.actor_optimizer.step()
        self.last_metrics = {'bc_loss':float(loss.detach()),'bc_rows':int(mask.sum())}
        return zero,loss.detach()


class HJTD3BC(ControllerBC):
    def __init__(self,cfg,device,pretrained,options):
        super().__init__(cfg,device,pretrained,options)
        self.critic = ResNetCritic(cfg.hidden_sizes,pretrained,freeze_batch_norm=True,history_frames=cfg.visual_history_frames).to(device)
        self.critic2 = ResNetCritic(cfg.hidden_sizes,pretrained,freeze_batch_norm=True,history_frames=cfg.visual_history_frames).to(device)
        self.target_actor = deepcopy(self.actor).eval().requires_grad_(False)
        self.target_critic = deepcopy(self.critic).eval().requires_grad_(False)
        self.target_critic2 = deepcopy(self.critic2).eval().requires_grad_(False)
        self.critic_optimizer = torch.optim.AdamW(list(self.critic.parameters())+list(self.critic2.parameters()),lr=cfg.critic_lr)
        self.gamma,self.tau = cfg.gamma,cfg.tau

    @property
    def state_keys(self):
        return ('actor','critic','critic2','target_actor','target_critic','target_critic2',
                'actor_optimizer','critic_optimizer')

    def update(self,batch,bc_target,bc_mask):
        self.step += 1
        image,action,successor,h,nh,failed = batch
        self.actor.train(); self.critic.train(); self.critic2.train()
        with torch.no_grad():
            target_action = smoothed_action(self.target_actor(successor),self.low,self.high,
                                           self.options.policy_noise,self.options.noise_clip)
            nq = torch.minimum(self.target_critic(successor,target_action),
                               self.target_critic2(successor,target_action))
            target = hj_target(h,nh,failed,nq,self.gamma)
        self.critic_optimizer.zero_grad(set_to_none=True)
        q1,q2 = self.critic(image,action),self.critic2(image,action)
        mse1,mse2 = F.mse_loss(q1,target),F.mse_loss(q2,target)
        (mse1+mse2).backward()
        self.critic_optimizer.step()
        self.actor_updated = self.step % self.options.policy_delay == 0 and bool(bc_mask.any())
        actor_loss = torch.zeros((),device=image.device)
        self.last_metrics = dict(critic1_mse=float(mse1.detach()),critic2_mse=float(mse2.detach()),
                                 critic_disagreement=float((q1-q2).detach().abs().mean()),
                                 bc_rows=int(bc_mask.sum()),actor_updated=self.actor_updated)
        if self.actor_updated:
            self.critic.requires_grad_(False).eval()
            self.critic2.requires_grad_(False).eval()
            self.actor_optimizer.zero_grad(set_to_none=True)
            proposed = self.actor(image[bc_mask])
            bc = imitation_loss(proposed,bc_target[bc_mask],self.low,self.high)
            q = self.critic(image[bc_mask],proposed)
            # Detach the scale: it must not introduce a gradient that cancels Q.
            weight = self.options.bc_alpha/q.detach().abs().mean().clamp_min(1e-3)
            warmup = self.step <= self.options.actor_warmup
            q_loss = torch.zeros_like(bc) if warmup else -weight*q.mean()
            actor_loss = bc+q_loss
            actor_loss.backward()
            self.actor_optimizer.step()
            self.critic.requires_grad_(True); self.critic2.requires_grad_(True)
            for target_model,online in ((self.target_actor,self.actor),(self.target_critic,self.critic),
                                        (self.target_critic2,self.critic2)):
                soft_update(target_model,online,self.tau)
            self.last_metrics.update(bc_loss=float(bc.detach()),q_actor_loss=float(q_loss.detach()),
                                     q_weight=float(weight),actor_bc_warmup=warmup)
        return ((mse1+mse2)/2).detach(),actor_loss.detach()


class HJSAC:
    state_keys = ('actor', 'critic', 'critic2', 'target_critic', 'target_critic2',
                  'actor_optimizer', 'critic_optimizer')
    actor_updated = True

    def __init__(self, cfg, device, pretrained, options):
        self.actor = ResNetActor(cfg.action_low, cfg.action_high, pretrained, freeze_batch_norm=True,
                                 history_frames=cfg.visual_history_frames, stochastic=True).to(device)
        self.critic = ResNetCritic(cfg.hidden_sizes, pretrained, freeze_batch_norm=True,
                                   history_frames=cfg.visual_history_frames).to(device)
        self.critic2 = ResNetCritic(cfg.hidden_sizes, pretrained, freeze_batch_norm=True,
                                    history_frames=cfg.visual_history_frames).to(device)
        self.target_critic = deepcopy(self.critic).eval().requires_grad_(False)
        self.target_critic2 = deepcopy(self.critic2).eval().requires_grad_(False)
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.AdamW(
            list(self.critic.parameters()) + list(self.critic2.parameters()), lr=cfg.critic_lr)
        self.gamma, self.tau, self.alpha = cfg.gamma, cfg.tau, options.sac_alpha

    def warmup_critic(self, batch):
        images, actions, _, margin, *_ = batch
        self.critic.train()
        self.critic2.train()
        loss = (F.mse_loss(self.critic(images, actions), margin) +
                F.mse_loss(self.critic2(images, actions), margin)) / 2
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.critic_optimizer.step()
        return loss.detach()

    def update(self, batch):
        image, action, successor, margin, next_margin, failed = batch
        self.actor.train()
        self.critic.train()
        self.critic2.train()
        with torch.no_grad():
            next_action, _ = self.actor.sample(successor)
            continuation = torch.minimum(self.target_critic(successor, next_action),
                                         self.target_critic2(successor, next_action))
            target = hj_target(margin, next_margin, failed, continuation, self.gamma)
        q1, q2 = self.critic(image, action), self.critic2(image, action)
        critic_loss = (F.mse_loss(q1, target) + F.mse_loss(q2, target)) / 2
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.eval().requires_grad_(False)
        self.critic2.eval().requires_grad_(False)
        proposed, log_prob = self.actor.sample(image)
        value = torch.minimum(self.critic(image, proposed), self.critic2(image, proposed))
        actor_loss = (self.alpha * log_prob - value).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic.requires_grad_(True)
        self.critic2.requires_grad_(True)
        soft_update(self.target_critic, self.critic, self.tau)
        soft_update(self.target_critic2, self.critic2, self.tau)
        self.last_metrics = {'sampled_policy_entropy': float(-log_prob.detach().mean())}
        return critic_loss.detach(), actor_loss.detach()
