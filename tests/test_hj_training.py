"""Invariants for the first-hit HJ backup, optimizers, normalization and resume."""

from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from training.config import LOCAL_DIR, Settings
from training.dataset import HJ_FEATURES, hj_examples
from training.train_ddpg import HJDDPG, hj_target, train, training_normalization


class HJTrainingTests(unittest.TestCase):
    def test_invalid_checkpoint_interval(self):
        for interval in (0, -1, 1.5, True):
            with self.assertRaisesRegex(ValueError, "checkpoint_interval"):
                train(Path("unused"), Settings(), Path("unused"), checkpoint_interval=interval)

    def test_absorbing_targets_and_discounted_sign(self):
        h = torch.tensor([2., 2., 2., 2.], requires_grad=True)
        next_h = torch.tensor([-.5, -.01, 3., 3.])
        failed = torch.tensor([True, True, False, False])
        next_q = torch.tensor([100., -100., 3., 1.], requires_grad=True)
        target = hj_target(h, next_h, failed, next_q, .99)
        torch.testing.assert_close(target, torch.tensor([-.475, .0101, 2., 1.01]))
        self.assertFalse(target.requires_grad)
        self.assertGreater(float(target[1]), 0.)

    def test_actor_ascent_and_soft_updates(self):
        cfg = replace(Settings(), hidden_sizes=(), critic_lr=0., tau=.25)
        trainer = HJDDPG(8, cfg, torch.device("cpu"))
        # Q increases linearly in longitudinal command, so the ascent direction is known.
        with torch.no_grad():
            trainer.actor.network[0].weight.zero_()
            trainer.actor.network[0].bias.zero_()
            trainer.critic.network[0].weight.zero_()
            trainer.critic.network[0].weight[0, -2] = 1.
        original_targets = {name: [p.clone() for p in getattr(trainer, "target_" + name).parameters()]
                            for name in ("actor", "critic")}
        state = torch.ones(10, 8)
        action = torch.ones(10, 2)
        margin = torch.ones(10)
        trainer.update((state, action, state, margin, margin, margin.bool()))
        self.assertGreater(float(trainer.actor(state)[0, 0]), 0.)
        self.assertGreater(float(trainer.actor.network[0].bias.grad.abs().sum()), 0.)
        self.assertIsNotNone(trainer.critic.network[0].weight.grad)
        for name in ("actor", "critic"):
            for prior, online, target in zip(original_targets[name], getattr(trainer, name).parameters(),
                                              getattr(trainer, "target_" + name).parameters()):
                torch.testing.assert_close(target, .75 * prior + .25 * online)
                self.assertFalse(target.requires_grad)
                self.assertIsNone(target.grad)

    def test_resume_and_training_statistics(self):
        LOCAL_DIR.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=LOCAL_DIR) as tmp:
            root = Path(tmp)
            lengths = torch.tensor([8, 8, 8])
            generator = torch.Generator().manual_seed(344)
            states = torch.randn(24, 12, generator=generator)
            data = {
                "format_version": 1, "lengths": lengths, "states": states,
                "actions": torch.rand(24, 2, generator=generator), "costs": torch.zeros(24),
                "distances": torch.tensor([2.] * 8 + [7.] * 8 + [24.] * 8),
                "sources": ["old", "new", "new"], "conditions": ["day_sunny"] * 3,
                "episode_ids": torch.tensor([0, 0, 1]), "splits": {"train": [0, 1], "val": [2], "test": []},
                "dataset_id": "fixture-data", "split_id": "fixture-split",
            }
            data_path = root / "cache.pt"
            torch.save(data, data_path)
            cfg = replace(Settings(), max_gap_m=30., batch_size=4, hidden_sizes=(8,), updates=4)
            full = train(data_path, cfg, root / "continuous", checkpoint_interval=3)
            self.assertEqual(sorted(p.name for p in (root / "continuous").glob("checkpoint_*.pt")),
                             ["checkpoint_00000.pt", "checkpoint_00003.pt", "checkpoint_00004.pt"])
            early = train(data_path, replace(cfg, updates=2), root / "resumed", checkpoint_interval=1)
            self.assertTrue((root / "resumed" / "checkpoint_00001.pt").exists())
            resumed = train(data_path, cfg, root / "resumed", resume=early, checkpoint_interval=3)
            self.assertTrue((root / "resumed" / "checkpoint_00003.pt").exists())

            def load(path):
                return torch.load(path, map_location="cpu", weights_only=True)

            verification = """
import json, sys, torch
from pathlib import Path
from training.models import SafetyFilter
path = Path(sys.argv[1])
model = SafetyFilter.from_checkpoint(path)
metrics = json.loads((path.parent / 'metrics.json').read_text())
torch.set_num_threads(metrics['cpu_threads'])
for probe in metrics['cpu_inference_probes']:
    decision = model.filter_action(probe['state'], probe['nominal'])
    assert list(model.predict_safe_action(probe['state'])) == probe['actor']
    assert decision.nominal_q == probe['score']
    assert decision.intervened == probe['intervened']
    assert list(decision.action) == probe['executed']
"""
            subprocess.run([sys.executable, "-B", "-c", verification, str(full)], check=True,
                           cwd=Path(__file__).resolve().parents[1])
            full, resumed = load(full), load(resumed)
            for name in ("actor", "critic", "target_actor", "target_critic"):
                for key in full[name]:
                    torch.testing.assert_close(full[name][key], resumed[name][key], rtol=0, atol=0)
            torch.testing.assert_close(full["sampler_state"], resumed["sampler_state"], rtol=0, atol=0)
            self.assertEqual(full["sampled"], resumed["sampled"])
            self.assertEqual(full["features"], HJ_FEATURES)
            independent_states = hj_examples(data, [0, 1], cfg)[0]
            mean, std = training_normalization(independent_states)
            torch.testing.assert_close(mean, full["state_mean"])
            torch.testing.assert_close(std, full["state_std"])
            self.assertAlmostEqual(float(mean[0]), 4.5)
            self.assertGreater(float(hj_examples(data, [2], cfg)[0][:, 0].mean()), 20.)
            with self.assertRaises(FileExistsError):
                train(data_path, cfg, root / "continuous")
            with self.assertRaisesRegex(ValueError, "selection differs"):
                train(data_path, cfg, root / "rejected", selection="new", resume=early)

if __name__ == "__main__":
    unittest.main()

