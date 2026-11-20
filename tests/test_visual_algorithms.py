"""Algorithm invariants independent of large CNN checkpoints."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from training.config import Settings
from training.visual_algorithms import HJTD3BC,ControllerBC,LearnerOptions,imitation_loss,smoothed_action
from training.visual_models import ResNetActor


class TinyActor(nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__()
        self.layer=nn.Linear(3,2)
        self.register_buffer('scale',torch.tensor([3.,1.]))
    def forward(self,x): return self.layer(x).tanh()*self.scale


class TinyCritic(nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__()
        self.layer=nn.Linear(5,1)
    def forward(self,x,u): return self.layer(torch.cat((x,u),1)).squeeze(-1)


class AlgorithmTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(7);torch.set_num_threads(2)

    def learner(self,algorithm='hj-td3-bc',**changes):
        options=replace(LearnerOptions(),algorithm=algorithm,**changes)
        with patch('training.visual_algorithms.ResNetActor',TinyActor),patch('training.visual_algorithms.ResNetCritic',TinyCritic):
            cls=ControllerBC if algorithm=='bc' else HJTD3BC
            return cls(Settings(),torch.device('cpu'),False,options)

    def batch(self):
        return (torch.ones(4,3),torch.zeros(4,2),torch.zeros(4,3),torch.ones(4),
                torch.full((4,),-.5),torch.ones(4,dtype=torch.bool))

    def test_frozen_batchnorm_has_identical_train_eval_behavior(self):
        model=ResNetActor(freeze_batch_norm=True)
        image=torch.zeros(2,224,224,3,dtype=torch.uint8)
        counter=model.network.bn1.num_batches_tracked.clone()
        model.train();a=model(image)
        self.assertFalse(model.network.bn1.training)
        a.sum().backward()
        self.assertIsNotNone(model.network.bn1.weight.grad)
        torch.testing.assert_close(counter,model.network.bn1.num_batches_tracked)
        model.eval();b=model(image)
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_normalized_loss_and_noise_scaling(self):
        low,high=torch.tensor([-3.,-1.]),torch.tensor([3.,1.])
        loss=imitation_loss(torch.tensor([[3.,1.]]),torch.zeros(1,2),low,high)
        self.assertEqual(loss.item(),1.)
        action=smoothed_action(torch.zeros(1,2),low,high,.2,.5,torch.tensor([[100.,-100.]]))
        torch.testing.assert_close(action,torch.tensor([[1.5,-.5]]))
        clipped=smoothed_action(high.unsqueeze(0),low,high,.2,.5,torch.ones(1,2)*100)
        torch.testing.assert_close(clipped,high.unsqueeze(0))

    def test_twin_hj_target_absorbs_and_actor_is_delayed(self):
        learner=self.learner(actor_warmup=0)
        for critic in (learner.critic,learner.critic2):
            with torch.no_grad(): critic.layer.weight.zero_();critic.layer.bias.zero_()
        original=[p.clone() for p in learner.actor.parameters()]
        targets=[p.clone() for p in learner.target_actor.parameters()]
        learner.update(self.batch(),torch.zeros(4,2),torch.ones(4,dtype=torch.bool))
        self.assertFalse(learner.actor_updated)
        self.assertAlmostEqual(learner.last_metrics['critic1_mse'],.485**2,places=5)
        self.assertAlmostEqual(learner.last_metrics['critic2_mse'],.485**2,places=5)
        for p,old in zip(learner.actor.parameters(),original):torch.testing.assert_close(p,old)
        for p,old in zip(learner.target_actor.parameters(),targets):torch.testing.assert_close(p,old)
        learner.update(self.batch(),torch.zeros(4,2),torch.ones(4,dtype=torch.bool))
        self.assertTrue(learner.actor_updated)
        self.assertTrue(any(not torch.equal(p,old) for p,old in zip(learner.actor.parameters(),original)))
        self.assertFalse(learner.last_metrics['actor_bc_warmup'])
        self.assertTrue(all(not p.requires_grad for p in learner.target_critic2.parameters()))

    def test_nonterminal_backup_uses_smaller_target_critic(self):
        learner=self.learner()
        learner.gamma=1.
        for model,bias in ((learner.target_critic,2.),(learner.target_critic2,-3.),
                           (learner.critic,0.),(learner.critic2,0.)):
            with torch.no_grad():
                model.layer.weight.zero_();model.layer.bias.fill_(bias)
        batch=list(self.batch());batch[-1]=torch.zeros(4,dtype=torch.bool)
        learner.update(batch,torch.zeros(4,2),torch.ones(4,dtype=torch.bool))
        self.assertAlmostEqual(learner.last_metrics['critic1_mse'],9.)
        self.assertAlmostEqual(learner.last_metrics['critic2_mse'],9.)

    def test_warmup_uses_only_imitation_and_empty_mask_skips_actor(self):
        learner=self.learner(actor_warmup=10,policy_delay=1)
        learner.update(self.batch(),torch.zeros(4,2),torch.ones(4,dtype=torch.bool))
        self.assertEqual(learner.last_metrics['q_actor_loss'],0.)
        old=[p.clone() for p in learner.actor.parameters()]
        learner.update(self.batch(),torch.zeros(4,2),torch.zeros(4,dtype=torch.bool))
        self.assertFalse(learner.actor_updated)
        for p,previous in zip(learner.actor.parameters(),old):torch.testing.assert_close(p,previous)

    def test_bc_loss_decreases_on_fixed_labels(self):
        learner=self.learner('bc')
        target=torch.tensor([[.6,-.2]]).expand(4,2)
        first=imitation_loss(learner.actor(self.batch()[0]),target,learner.low,learner.high).item()
        for _ in range(20):learner.update(self.batch(),target,torch.ones(4,dtype=torch.bool))
        last=imitation_loss(learner.actor(self.batch()[0]),target,learner.low,learner.high).item()
        self.assertLess(last,first)
        self.assertIsNone(learner.critic)


if __name__=='__main__':unittest.main()
