"""Counterfactual action scans must not alter training randomness or controls."""
import math
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from training.train_online_hj import OnlineHJ, Observation, VisualNetwork
from tests.test_online_hj import tiny_encoder


class ActionDiagnosticsTests(unittest.TestCase):
    def test_scan_matches_direct_scores_and_preserves_rng_parameters_and_modes(self):
        torch.set_num_threads(2)
        args=NS(history_frames=4,include_speed=True,no_pretrained=True,freeze_encoder=True,
            actor_lr=1e-4,critic_lr=1e-4)
        with patch.object(VisualNetwork,'__init__',tiny_encoder):
            learner=OnlineHJ(args,'cpu')
        with torch.no_grad():
            learner.actor.head[-1].weight.zero_()
            learner.actor.head[-1].bias.copy_(torch.tensor([0.,math.log(.5)]))
        image=torch.ones(224,224,3,dtype=torch.uint8)
        observation=Observation((image,)*4,(1,2,3,4),15.,0)
        learner.actor.train(); learner.critics.train(); learner.critics[0].eval()
        models=[learner.actor,learner.critics]
        modes=[m.training for model in models for m in model.modules()]
        parameters=[p.detach().clone() for model in models for p in model.parameters()]
        rng=torch.get_rng_state().clone()
        diagnostic=learner.action_diagnostics(observation,.6)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(modes,[m.training for model in models for m in model.modules()])
        for before,after in zip(parameters,[p for model in models for p in model.parameters()]):
            self.assertTrue(torch.equal(before,after))
        self.assertEqual(diagnostic['deterministic_action'],0.)
        self.assertAlmostEqual(diagnostic['latent_std'],.5)
        for key,action in [('brake_q',-3.),('coast_q',0.),('throttle_q',3.),
                           ('deterministic_q',0.),('executed_q',.6)]:
            self.assertAlmostEqual(diagnostic[key],learner.score(observation,action),places=5)
        self.assertGreaterEqual(diagnostic['best_grid_action'],-3.)
        self.assertLessEqual(diagnostic['best_grid_action'],3.)
        self.assertGreaterEqual(diagnostic['actor_q_gap'],0.)


if __name__ == '__main__': unittest.main()
