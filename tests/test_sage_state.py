"""Exercise the real state networks through production replay and SAGE updates."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest

import torch

from training.test_sage_state import StateHistory, StateSAGE
from training.train_online_hj import Observation, Transition, Replay


class StateDiagnosticTests(unittest.TestCase):
    def test_state_snapshot_and_replay_use_normalized_geometry(self):
        from unittest.mock import patch
        world = NS(get_snapshot=lambda: NS(frame=10, find=lambda actor: NS(
            get_velocity=lambda: NS(x=15. if actor == 1 else 12., y=0., z=0.))))
        env = NS(world=world, ego=NS(id=1), nonego=NS(id=2))
        history = StateHistory(1, 0, .1)
        with patch('training.closed_loop.read_snapshot', return_value=(10, 1., None, 6.)):
            obs = history.capture(env)
            torch.testing.assert_close(obs.frames[0], torch.tensor([.6, .3, .5]))
            with self.assertRaisesRegex(RuntimeError, 'skipped'):
                history.capture(env)

    def test_actual_state_networks_update_and_checkpoint_roundtrip(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        args = NS(history_frames=1, include_speed=True, no_pretrained=True,
            freeze_encoder=False, actor_lr=1e-4, critic_lr=1e-4,
            gamma_start=.99, gamma_final=.99, alpha=0., alpha_final=0.,
            anneal_updates=20000, tau=.05, performance_gamma=.99,
            performance_alpha=.2, performance_tau=.005, safety_objective='collision',
            event_failure_margin=1., max_gap=9.)
        learner = StateSAGE(args, 'cpu')
        obs = Observation((torch.tensor([.6, .1, .5]),), (10,), 15., 0)
        nxt = Observation((torch.tensor([.59, .1, .5]),), (11,), 15., 0)
        replay = Replay(8)
        replay.extend([Transition(obs, -2., nxt, 6., 5.9, False, reward=.9),
                       Transition(obs, 2., nxt, .1, -1., True, reward=-10., terminated=True)])
        before = {name: next(getattr(learner, name).parameters()).detach().clone()
                  for name in ('actor', 'critics', 'goal_actor', 'goal_critics')}
        for _ in range(3):
            metrics = learner.update(replay.sample(8, 'cpu'))
            self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics.values()))
        for name, original in before.items():
            self.assertFalse(torch.equal(original, next(getattr(learner, name).parameters())))
        self.assertTrue(-1 <= learner.score(obs, -3.) <= 9)
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'state.pt'
            learner.save(path, 3)
            loaded = StateSAGE.from_checkpoint(path)
            self.assertAlmostEqual(loaded.goal_action(obs, True), learner.goal_action(obs, True))


if __name__ == '__main__':
    unittest.main()
