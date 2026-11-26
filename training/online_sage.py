"""SAC performance actor-critic paired with the online HJ safety actor-critic."""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F

from .train_online_hj import OnlineHJ, Actor, Critic, tensors
from .visual_algorithms import soft_update


def following_reward(gap, relative_speed, action, previous_action, target_gap):
    """Simulator-only task feedback; geometry never enters the actor inputs."""
    return (1.-abs(gap-target_gap)/target_gap-.2*abs(relative_speed)/5.
            -.05*((action-previous_action)/3.)**2)


def sac_backup(reward, terminated, continuation, log_probability, gamma, alpha):
    return reward+gamma*(~terminated).float()*(continuation-alpha*log_probability)


def finalize_performance(rows, contacts, departures, args):
    """True task terminations stop reward bootstraps; time limits do not."""
    if not rows: return rows
    lo,hi=rows[0].observation.frame_ids[-1],rows[-1].successor.frame_ids[-1]
    contact_frames=[f for f,_ in contacts if lo<=f<=hi]
    lane_frames=[f for f in departures if lo<=f<=hi]
    if contact_frames or lane_frames:
        first=min(contact_frames+lane_frames)
        index=next(i for i,row in enumerate(rows) if row.successor.frame_ids[-1]>=first)
        rows=rows[:index+1]
        rows[-1].terminated=True
        rows[-1].reward-=args.collision_penalty if first in contact_frames else args.lane_penalty
    return rows


class OnlineSAGE(OnlineHJ):
    architecture = 'online_sage_rgb_longitudinal_v2'
    model_names = OnlineHJ.model_names+('goal_actor','goal_critics','goal_targets')
    snapshot_names = model_names+('actor_optimizer','critic_optimizer','goal_actor_optimizer','goal_critic_optimizer')

    def __init__(self,args,device):
        super().__init__(args,device)
        contract=(args.history_frames,args.include_speed,not args.no_pretrained,args.freeze_encoder)
        self.goal_actor=Actor(*contract).to(device)
        self.goal_critics=nn.ModuleList([Critic(*contract),Critic(*contract)]).to(device)
        self.goal_targets=deepcopy(self.goal_critics).eval().requires_grad_(False)
        self.goal_actor_optimizer=torch.optim.Adam(self.goal_actor.parameters(),lr=args.actor_lr)
        self.goal_critic_optimizer=torch.optim.Adam(self.goal_critics.parameters(),lr=args.critic_lr)

    @torch.no_grad()
    def goal_action(self,observation,deterministic=False):
        self.goal_actor.eval()
        images,speed=tensors([observation],self.device)
        return float(self.goal_actor(images,speed,deterministic)[0][0,0])

    def update(self,batch):
        metrics=super().update(batch)
        image,speed,executed_action,successor,next_speed=batch[:5]
        reward,terminated=batch[8:10]
        self.goal_actor.train(); self.goal_critics.train()
        with torch.no_grad():
            proposed,log_probability=self.goal_actor(successor,next_speed)
            continuation=torch.minimum(*(q(successor,next_speed,proposed) for q in self.goal_targets))
            target=sac_backup(reward,terminated,continuation,log_probability,
                              self.args.performance_gamma,self.args.performance_alpha)
        values=[q(image,speed,executed_action) for q in self.goal_critics]
        critic_loss=sum(F.mse_loss(q,target) for q in values)/2
        if not torch.isfinite(critic_loss): raise FloatingPointError('Nonfinite SAC performance critic loss')
        self.goal_critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward(); nn.utils.clip_grad_norm_(self.goal_critics.parameters(),10.)
        self.goal_critic_optimizer.step()
        parameters=[p for p in self.goal_critics.parameters() if p.requires_grad]
        for p in parameters: p.requires_grad_(False)
        self.goal_critics.eval()
        try:
            proposed,log_probability=self.goal_actor(image,speed)
            q=torch.minimum(*(critic(image,speed,proposed) for critic in self.goal_critics))
            action_gradient=torch.autograd.grad(q.sum(),proposed,retain_graph=True)[0].detach().abs().mean()
            actor_loss=(self.args.performance_alpha*log_probability-q).mean()
            if not torch.isfinite(actor_loss): raise FloatingPointError('Nonfinite SAC performance actor loss')
            self.goal_actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward(); nn.utils.clip_grad_norm_(self.goal_actor.parameters(),10.)
            self.goal_actor_optimizer.step()
        finally:
            for p in parameters: p.requires_grad_(True)
        soft_update(self.goal_targets,self.goal_critics,self.args.performance_tau)
        metrics.update(goal_critic_mse=float(critic_loss.detach()),goal_actor_loss=float(actor_loss.detach()),
                       goal_target_mean=float(target.mean()),goal_q_mean=float(values[0].detach().mean()),
                       goal_entropy=float(-log_probability.detach().mean()),reward_mean=float(reward.mean()),
                       goal_action_gradient=float(action_gradient))
        return metrics


class GoalPolicy:
    """Adapter lets synchronized rollout/evaluation code use the learned proposer."""
    is_goal = True
    def __init__(self,learner,target_gap):
        self.learner=learner
        self.profile=dict(target_gap=target_gap)
    def reset(self): pass
    def propose(self,observation,dt,deterministic=False):
        return self.learner.goal_action(observation,deterministic),None

