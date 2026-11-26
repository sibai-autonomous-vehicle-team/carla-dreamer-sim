"""Observed HJ returns, grouped holdout, and causal RGB history checks."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch

from training.config import Settings
from training.dataset import start_groups
from training.train_braking_critic import BrakingExamples, episode_scores, grouped_split, observed_returns, scores


class CriticPilotTests(unittest.TestCase):
    def test_returns_absorb_first_buffer_hit(self):
        cfg = replace(Settings(), gamma=1.)
        target = observed_returns(torch.tensor([3., 2., .5, .1]), cfg)
        self.assertTrue(torch.equal(target, torch.tensor([-.5, -.5])))

    def test_discounted_returns_reuse_the_hj_backup(self):
        cfg = replace(Settings(), gamma=.5)
        target = observed_returns(torch.tensor([3., 2., .5]), cfg)
        self.assertTrue(torch.equal(target, torch.tensor([1.125, .25])))

    def test_successful_cutoff_uses_last_observed_margin(self):
        cfg = replace(Settings(), gamma=1.)
        self.assertEqual(observed_returns(torch.tensor([3., 2., 3.]), cfg).tolist(), [1., 1.])

    def test_grouped_holdout_keeps_matching_starts_together(self):
        states = []
        for position in (0., .1, 10., 20., 30., 40.):
            state = torch.zeros(2, 12)
            state[:, 0], state[:, 6] = position, position + 6.
            states.append(state)
        split = grouped_split(states, 4, 42)
        self.assertEqual(split, grouped_split(states, 4, 42))
        groups = start_groups(dict(states=torch.cat(states), lengths=torch.tensor([2] * 6)), .25)
        self.assertFalse({groups[i] for i in split['train']} & {groups[i] for i in split['test']})
        self.assertEqual(set(split['train']) | set(split['test']), set(range(6)))

    def test_history_padding_has_no_future_frames(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'obses').mkdir()
            frames = torch.arange(4, dtype=torch.uint8).reshape(4, 1, 1, 1).expand(4, 224, 224, 3).clone()
            torch.save(frames, folder / 'obses/episode_000.pth')
            data = BrakingExamples(folder, dict(actions=[torch.zeros(4, 2)],
                                               distances=[torch.full((4,), 3.)]),
                                   [dict(initial_gap=torch.tensor(3.))], [0],
                                   replace(Settings(), visual_history_frames=3))
            self.assertEqual(data[0][0][:, 0, 0, 0].tolist(), [0, 0, 0])
            self.assertEqual(data[2][0][:, 0, 0, 0].tolist(), [0, 1, 2])

    def test_metrics_expose_constant_unsafe_predictions(self):
        result = scores(torch.tensor([-1., 1.]), torch.tensor([-1., -1.]))
        self.assertEqual(result['false_safe_rate'], 0.)
        self.assertEqual(result['false_alarm_rate'], 1.)

    def test_completed_buffer_hit_is_separate_from_false_alert_on_safe_episode(self):
        summary = dict(ending='braking_complete', collision_before_braking=False,
                       initial_gap_m=3., requested_speed_mps=15.)
        metadata = [dict(initial_gap=torch.tensor(3.), frames=torch.arange(3), collision_events=[], summary=summary)
                    for _ in range(2)]
        data = SimpleNamespace(refs=[(0, 0), (0, 1), (1, 0), (1, 1)], metadata=metadata,
                               columns=dict(distances=[torch.tensor([2., .9]), torch.tensor([2., 1.5])]))
        result = episode_scores(data, torch.tensor([-1., -1., -1., -1.]), Settings())
        self.assertEqual(result['completed_braking_with_alert'], 2)
        self.assertEqual(result['buffer_safe_completed_episodes'], 1)
        self.assertEqual(result['buffer_safe_completed_with_alert'], 1)


if __name__ == '__main__':
    unittest.main()
