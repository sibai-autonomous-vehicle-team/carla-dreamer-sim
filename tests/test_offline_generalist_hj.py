"""Logged transition alignment and cached-feature HJ-SAC gradients."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from torch import nn

from training.train_online_hj import Actor, OnlineHJ, VisualNetwork
from training.train_offline_generalist_hj import FeatureReplay, transition_rows, update_features


class OfflineTests(unittest.TestCase):
    def test_original_uses_pre_action_images_and_post_action_labels(self):
        actions=torch.tensor([[0.,0.],[1.,0.],[2.,0.],[3.,0.]])
        gap=torch.tensor([8.,7.,6.,5.])
        rows,a,h,nh,failed=transition_rows(actions,gap,torch.zeros(4))
        self.assertEqual(rows.tolist(),[1,2])
        self.assertEqual(a.flatten().tolist(),[1.,2.])
        self.assertEqual(h.tolist(),[8.,7.])
        self.assertEqual(nh.tolist(),[7.,6.])
        self.assertFalse(failed.any())

    def test_braking_retains_terminal_collision_and_truncates_later_rows(self):
        meta=dict(frames=torch.arange(10,15),timestamps=torch.arange(5,dtype=torch.float64)*.1,
                  dt=.1,initial_gap=torch.tensor(8.),collision_events=[(13,7)])
        rows,a,h,nh,failed=transition_rows(torch.zeros(4,2),torch.tensor([7.,6.,5.,4.]),
                                         torch.zeros(4),metadata=meta)
        self.assertEqual(rows.tolist(),[0,1,2])
        self.assertEqual(h.tolist(),[8.,7.,6.])
        self.assertEqual(nh.tolist(),[7.,6.,-1.])
        self.assertEqual(failed.tolist(),[False,False,True])
        meta['frames'][2]+=2
        with self.assertRaisesRegex(ValueError,'Nonconsecutive'):
            transition_rows(torch.zeros(4,2),torch.ones(4),torch.zeros(4),metadata=meta)

    def test_cached_features_train_heads_and_preserve_encoders(self):
        torch.set_num_threads(2)
        def tiny(self,history,include_speed,pretrained,freeze_encoder):
            nn.Module.__init__(self)
            self.width=8
            self.freeze_batch_norm=True
            self.network=nn.Linear(2,2).requires_grad_(False)
        args=NS(history_frames=4,include_speed=False,no_pretrained=True,freeze_encoder=True,
                safety_objective='collision',event_failure_margin=1.,max_gap=9.,
                actor_lr=1e-3,critic_lr=1e-3,gamma_start=.85,gamma_final=.99,
                anneal_updates=10,alpha=.05,alpha_final=0.,tau=.05)
        with patch.object(VisualNetwork,'__init__',tiny):
            learner=OnlineHJ(args,'cpu')
        before={k:v.clone() for k,v in learner.actor.network.state_dict().items()}
        original=learner.actor.head[-1].weight.detach().clone()
        batch=(torch.randn(16,8),torch.zeros(16,1),torch.randn(16,8),
               torch.ones(16)*2,torch.ones(16),torch.zeros(16,dtype=torch.bool))
        result=update_features(learner,batch)
        self.assertEqual(learner.updates,1)
        self.assertTrue(torch.isfinite(torch.tensor(result['critic_mse'])))
        self.assertFalse(torch.equal(original,learner.actor.head[-1].weight))
        for key,value in learner.actor.network.state_dict().items():
            torch.testing.assert_close(value,before[key])
        self.assertTrue(all(p.grad is None for p in learner.actor.network.parameters()))
        self.assertTrue(all(not p.requires_grad for p in learner.targets.parameters()))

    def test_replay_respects_validation_partition(self):
        data=dict(features=torch.arange(4.).reshape(4,1).expand(4,2),
                  refs=torch.tensor([[[0]*4,[1]*4],[[2]*4,[3]*4]]),
                  action=torch.tensor([[1.],[2.]]),margin=torch.ones(2),
                  next_margin=torch.ones(2),failed=torch.zeros(2,dtype=torch.bool),
                  validation=torch.tensor([False,True]))
        replay=FeatureReplay([data]*9)
        self.assertTrue((replay.sample(20,'train','cpu')[1]==1).all())
        self.assertTrue((replay.sample(20,'val','cpu')[1]==2).all())


if __name__ == '__main__':
    unittest.main()
