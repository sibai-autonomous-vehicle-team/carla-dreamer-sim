"""Independent distribution, gradient, Bellman and cached-update checks."""
from copy import deepcopy
import math
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.distributions import Normal, TransformedDistribution, TanhTransform, AffineTransform

from training.online_sage import sac_backup
from training.train_online_hj import Actor, OnlineHJ, VisualNetwork
from training.train_offline_generalist_hj import feature_target, update_features


def tiny_network(self,history,include_speed,pretrained,freeze_encoder):
    nn.Module.__init__(self)
    self.width=8
    self.freeze_batch_norm=True
    self.network=nn.Linear(2,2).requires_grad_(False)


def args(**changes):
    values=dict(history_frames=4,include_speed=False,no_pretrained=True,freeze_encoder=True,
                safety_objective='collision',event_failure_margin=1.,max_gap=9.,
                actor_lr=1e-3,critic_lr=1e-3,gamma_start=.85,gamma_final=.99,
                anneal_updates=10,alpha=.05,alpha_final=0.,tau=.25)
    values.update(changes)
    return NS(**values)


class SACImplementationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(123)

    def learner(self,**changes):
        with patch.object(VisualNetwork,'__init__',tiny_network):
            return OnlineHJ(args(**changes),'cpu')

    def batch(self):
        return (torch.randn(8,8),torch.linspace(-3,3,8)[:,None],torch.randn(8,8),
                torch.linspace(1,4,8),torch.tensor([-1.,2.,1.,-1.,1.,2.,1.,2.]),
                torch.tensor([True,False,False,True,False,False,False,False]))

    def test_actor_log_probability_matches_independent_transformed_distribution(self):
        with patch.object(VisualNetwork,'__init__',tiny_network):actor=Actor(4,False,False,True)
        actor.head=nn.Linear(8,2)
        with torch.no_grad():
            actor.head.weight.zero_();actor.head.bias.copy_(torch.tensor([.3,-.7]))
        features=torch.randn(16,8)
        action,logp=actor.sample_features(features)
        distribution=Normal(torch.full((16,1),.3),torch.full((16,1),math.exp(-.7)))
        normalized=TransformedDistribution(distribution,[TanhTransform()])
        physical=TransformedDistribution(distribution,[TanhTransform(),AffineTransform(0.,3.)])
        torch.testing.assert_close(logp,normalized.log_prob(action/3).sum(-1),atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(logp,physical.log_prob(action).sum(-1)+math.log(3),atol=2e-6,rtol=2e-6)
        self.assertTrue((action.abs()<=3).all())

    def test_saturated_actor_has_finite_log_probabilities_and_gradients(self):
        with patch.object(VisualNetwork,'__init__',tiny_network):actor=Actor(4,False,False,True)
        actor.head=nn.Linear(8,2)
        with torch.no_grad():
            actor.head.weight.zero_();actor.head.bias.copy_(torch.tensor([30.,-5.]))
        action,logp=actor.sample_features(torch.zeros(4,8))
        self.assertTrue(torch.isfinite(logp).all())
        (logp.mean()+action.mean()).backward()
        self.assertTrue(torch.isfinite(actor.head.bias.grad).all())

    def test_actor_gradient_moves_toward_braking_when_true_q_prefers_braking(self):
        learner=self.learner(alpha=0.,alpha_final=0.)
        with torch.no_grad():
            learner.actor.head[-1].weight.zero_()
            learner.actor.head[-1].bias.copy_(torch.tensor([0.,-4.]))
        batch=self.batch()
        before=learner.actor.sample_features(batch[0],True)[0].detach()
        def true_value(self,features,action):
            # Known Q=-a: reducing acceleration must increase value.
            return -action.squeeze(-1)+next(self.head.parameters()).sum()*0
        with patch('training.train_online_hj.Critic.value',true_value):
            update_features(learner,batch)
        after=learner.actor.sample_features(batch[0],True)[0].detach()
        self.assertTrue((after<before).all())

    def test_pixel_and_cached_updates_match_parameters_and_optimizer_states(self):
        online=self.learner();cached=deepcopy(online)
        batch=self.batch()
        f,a,nf,h,nh,failed=batch
        pixel_batch=(f,torch.zeros(8,1),a,nf,torch.zeros(8,1),h,nh,failed)
        with patch.object(VisualNetwork,'features',lambda self,image,speed:image):
            torch.manual_seed(91);online.update(pixel_batch)
        torch.manual_seed(91);update_features(cached,batch)
        for name in online.model_names:
            for key,value in getattr(online,name).state_dict().items():
                torch.testing.assert_close(value,getattr(cached,name).state_dict()[key],atol=1e-7,rtol=1e-6)
        for name in ('actor_optimizer','critic_optimizer'):
            left=getattr(online,name).state_dict()['state']
            right=getattr(cached,name).state_dict()['state']
            self.assertEqual(left.keys(),right.keys())
            for parameter,state in left.items():
                for key,value in state.items():torch.testing.assert_close(value,right[parameter][key])

    def test_safety_target_uses_twin_min_and_detaches_terminal_bootstrap(self):
        learner=self.learner(gamma_start=.9,gamma_final=.9)
        batch=(torch.zeros(2,8),torch.zeros(2,1),torch.zeros(2,8,requires_grad=True),
               torch.tensor([3.,3.]),torch.tensor([1.,-1.]),torch.tensor([False,True]))
        for critic,value in zip(learner.targets,(2.,4.)):
            critic.value=lambda features,action,value=value:torch.full((len(features),),value)
        # SAGE safety backup excludes entropy intentionally; actor loss retains it.
        learner.actor.sample_features=lambda f:(torch.zeros(len(f),1),torch.full((len(f),),-100.))
        target=feature_target(learner,batch)
        torch.testing.assert_close(target,torch.tensor([2.1,-.6]))
        self.assertFalse(target.requires_grad)

    def test_standard_sac_reward_backup_has_entropy_and_terminal_mask(self):
        result=sac_backup(torch.tensor([1.,1.]),torch.tensor([False,True]),
                          torch.tensor([2.,2.]),torch.tensor([-.7,-.7]),.9,.2)
        torch.testing.assert_close(result,torch.tensor([2.926,1.]))

    def test_target_update_moves_by_tau_and_has_no_gradients(self):
        learner=self.learner(actor_lr=0.,critic_lr=0.)
        original=[deepcopy(q.state_dict()) for q in learner.targets]
        with torch.no_grad():
            for q in learner.critics:q.head[-1].bias.add_(1.)
        update_features(learner,self.batch())
        for previous,target,critic in zip(original,learner.targets,learner.critics):
            expected=previous['head.4.bias']*.75+critic.state_dict()['head.4.bias']*.25
            torch.testing.assert_close(target.state_dict()['head.4.bias'],expected)
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in target.parameters()))


if __name__=='__main__':unittest.main()
